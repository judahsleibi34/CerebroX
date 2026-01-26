import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from matplotlib import colors as mcolors
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def get_col_groups(df: pd.DataFrame):
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(exclude=["number"]).columns.tolist()

    # remove ID-like numeric columns (optional but recommended)
    def looks_like_id(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return False
        # high uniqueness => likely id
        return (s.nunique() / len(s)) > 0.95

    numeric_clean = []
    for c in num:
        if not looks_like_id(df[c]):
            numeric_clean.append(c)

    # good categorical: not too many unique values
    categorical_good = []
    for c in cat:
        nunique = df[c].nunique(dropna=False)
        if 2 <= nunique <= 30:
            categorical_good.append(c)

    return numeric_clean, categorical_good


def choose_xy(df: pd.DataFrame):
    """Pick 2 numeric columns with most variance + non-null."""
    nums = df.select_dtypes(include=["number"])
    stats = []
    for c in nums.columns:
        s = nums[c].dropna()
        if len(s) < 10:
            continue
        stats.append((c, s.notna().mean(), float(s.var())))
    stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in stats[:2]]


def choose_group(df: pd.DataFrame):
    """Pick a categorical column with reasonable group sizes."""
    _, cats = get_col_groups(df)
    if not cats:
        return None
    # prefer lowest cardinality
    cats_sorted = sorted(cats, key=lambda c: df[c].nunique(dropna=False))
    return cats_sorted[0]


class ColumnResolver:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cols = list(df.columns)

    def _match(self, patterns, prefer_exact=True):
        cols = self.cols

        # exact match first
        if prefer_exact:
            lower_map = {c.strip().lower(): c for c in cols}
            for p in patterns:
                key = p.strip().lower()
                if key in lower_map:
                    return lower_map[key]

        # regex contains match
        for c in cols:
            name = c.strip().lower()
            for p in patterns:
                if re.search(p, name):
                    return c
        return None

    def tls_col(self):
        return self._match([
            r"\btls\b",
            r"teaching\s*location",
            r"school\s*location",
            r"\blocation\b",
            r"\bsite\b",
            r"\bcampus\b",
        ])

    def grade_col(self):
        return self._match([
            r"\bgrade\b",
            r"grade\s*level",
            r"\bclass\b",
            r"\blevel\b",
            r"\byear\b",
        ])

    def gender_col(self):
        return self._match([
            r"\bgender\b",
            r"\bsex\b",
        ])

    def score_cols(self, k=3):
        numeric = self.df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric:
            return []

        # prefer columns that look like scores/marks
        scored = []
        for c in numeric:
            name = c.lower()
            if re.search(r"score|mark|result|total|arabic|english|math|science", name):
                scored.append(c)

        if scored:
            return scored[:k]

        # fallback: best numeric by non-null + variance
        return best_numeric_cols(self.df, k=k)


def best_numeric_cols(df: pd.DataFrame, k=2):
    nums = df.select_dtypes(include=["number"])
    if nums.shape[1] == 0:
        return []

    stats = []
    for c in nums.columns:
        s = nums[c].dropna()
        if len(s) < 3:
            continue
        stats.append((c, float(s.notna().mean()), float(s.var())))

    stats.sort(key=lambda x: (x[1], x[2]), reverse=True)  # non-null then variance
    return [c for c, _, _ in stats[:k]]


def best_categorical_cols(df: pd.DataFrame, max_unique=25):
    cats = df.select_dtypes(exclude=["number"]).columns.tolist()
    if not cats:
        return []

    good = []
    for c in cats:
        nunique = df[c].nunique(dropna=False)
        if 2 <= nunique <= max_unique:
            good.append((c, nunique))

    # smaller cardinality usually works better visually
    good.sort(key=lambda x: x[1])
    return [c for c, _ in good]


class VisualizationConfig:
    DARK_BLUE = "#132249"
    LIGHT_BLUE = "#358cbf"
    PIE_PALETTE_NAME = "Blues"
    FONT_FAMILY = "Times New Roman"
    TITLE_SIZE = 24
    LABEL_SIZE = 18
    TICK_SIZE = 14
    VALUE_LABEL_SIZE = 12
    DEFAULT_SIZE = (14, 8)
    DPI_SCREEN = 300
    DPI_PRINT = 600

    @classmethod
    def get_palette(cls, n: int) -> List:
        cmap = LinearSegmentedColormap.from_list("alnayzak", [cls.LIGHT_BLUE, cls.DARK_BLUE], N=n)
        return [cmap(i / max(1, n - 1)) for i in range(n)]

    @classmethod
    def pie_palette(cls, labels, emphasize=None):
        n = len(labels)

        raw = sns.color_palette(cls.PIE_PALETTE_NAME, n)
        pal = [mcolors.to_hex(c) for c in raw]

        if emphasize is not None and emphasize in labels:
            idx = labels.index(emphasize)
            pal[idx] = cls.DARK_BLUE

        return pal

    @staticmethod
    def _text_color_for(bg_hex: str) -> str:
        r, g, b = mcolors.to_rgb(bg_hex)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance > 0.6 else "#ffffff"

    @classmethod
    def get_font_config(cls, custom: Optional[Dict] = None) -> Dict:
        config = {
            "family": cls.FONT_FAMILY,
            "title_size": cls.TITLE_SIZE,
            "label_size": cls.LABEL_SIZE,
            "tick_size": cls.TICK_SIZE,
            "value_label_size": cls.VALUE_LABEL_SIZE,
        }
        if custom:
            config.update(custom)
        return config


class DataManager:
    def __init__(
        self,
        dataset_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        default_xlabel: Optional[str] = None,
        default_ylabel: Optional[str] = "Count",
        ylim_padding: float = 0.1,
        df: Optional[pd.DataFrame] = None,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else Path("in_memory.csv")
        self.dataset_base = self._strip_suffixes(self.dataset_path.name)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            parent = self.dataset_path.parent
            self.output_dir = parent / f"{self.dataset_base}_run"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_frame = None
        self.default_xlabel = default_xlabel
        self.default_ylabel = default_ylabel
        self.ylim_padding = float(ylim_padding)

        if df is not None:
            self.set_dataframe(df)

    @staticmethod
    def _strip_suffixes(filename: str) -> str:
        base = filename.split(".", 1)[0].strip()
        return base or "dataset"

    def load_data(self, clean_columns: bool = True):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        self.data_frame = pd.read_csv(self.dataset_path)
        if clean_columns:
            self.data_frame.columns = (
                self.data_frame.columns.str.strip().str.replace("\n", " ", regex=True)
            )

    def set_dataframe(self, df: pd.DataFrame, clean_columns: bool = True):
        self.data_frame = df.copy()
        if clean_columns:
            self.data_frame.columns = (
                self.data_frame.columns.str.strip().str.replace("\n", " ", regex=True)
            )

    def get_data(self) -> pd.DataFrame:
        return self.data_frame.copy()

    def list_columns(self, column_type: Optional[str] = None) -> List[str]:
        if column_type == 'numeric':
            return self.data_frame.select_dtypes(include=[np.number]).columns.tolist()
        elif column_type == 'categorical':
            return self.data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
        return self.data_frame.columns.tolist()

    def quick_summary(self, numeric_only: bool = True) -> pd.DataFrame:
        return self.data_frame.describe() if numeric_only else self.data_frame.describe(include='all')


class BasePlotter:
    def __init__(self, data: DataManager, config: VisualizationConfig):
        self.data = data
        self.config = config

    def _slug(self, text: str) -> str:
        raw = str(text).strip()
        s = re.sub(r"\s+", "_", raw)
        s = re.sub(r"[^\w\-]", "", s)
        return re.sub(r"_+", "_", s).lower() or "chart"

    def _auto_adjust_ylim(self, ax, padding: Optional[float] = None):
        plt.draw()

        y_min, y_max = ax.get_ylim()
        expand_factor = 1.15 if padding is None else (1 + float(padding))
        new_y_max = y_max * expand_factor
        ax.set_ylim(y_min, new_y_max)

    def _save_figure(self, fig, label: str, chart_type: str, save_path=None, formats=None):
        formats = formats or ['png']
        saved_files = []

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            saved_files.append(save_path)

        else:
            base = f"{chart_type}_{self._slug(label)}_{self.data.dataset_base}"
            for fmt in formats:
                file_path = self.data.output_dir / f"{base}.{fmt}"
                file_path.parent.mkdir(parents=True, exist_ok=True)

                fig.savefig(file_path, dpi=300, bbox_inches="tight", facecolor="white")

                saved_files.append(file_path)

        return saved_files

    def _apply_font_config(self, font_config):
        cfg = self.config.get_font_config(font_config)
        plt.rcParams["font.family"] = cfg["family"]
        return cfg

    def _style_axes(self, ax, title, xlabel, ylabel, font):
        ax.set_title(title, fontsize=font["title_size"], fontweight="bold", pad=20, color=self.config.DARK_BLUE, loc="left")
        ax.set_xlabel(xlabel, fontsize=font["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.set_ylabel(ylabel, fontsize=font["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.tick_params(axis="both", labelsize=font["tick_size"])
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
        ax.grid(False, axis="x")
        ax.set_facecolor("white")

    def _annotate_bars(self, ax, fmt="{:,.0f}", font_size=None):
        font_size = font_size or self.config.VALUE_LABEL_SIZE
        for patch in ax.patches:
            height = patch.get_height()
            if np.isnan(height) or height <= 0:
                continue
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold",
                color=self.config.DARK_BLUE,
            )


class BarChartPlotter(BasePlotter):
    def plot(self, chart_name, column, xlabel=None, ylabel=None, ylim_padding=None, label=None, colors=None, size=None, font=None, order=None, sort_by="x", ascending=True, annotate=True, annotate_format="{:,.0f}", save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        counts = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
        if sort_by == "y":
            counts = counts.sort_values("count", ascending=ascending)
        else:
            counts = counts.sort_values(column, ascending=ascending)
        order = order or counts[column].tolist()
        palette = colors or self.config.get_palette(len(order))
        sns.barplot(data=counts, x=column, y="count", order=order, ci=None, ax=ax, palette=palette)
        self._auto_adjust_ylim(ax, padding=ylim_padding)
        resolved_xlabel = xlabel or self.data.default_xlabel or column.replace("_", " ").title()
        resolved_ylabel = ylabel or self.data.default_ylabel
        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        if annotate:
            self._annotate_bars(ax, annotate_format, font_config["value_label_size"])
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "bar_chart", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class HistogramPlotter(BasePlotter):
    def plot(self, chart_name, column, xlabel=None, ylabel=None, ylim_padding=None, bins=20, size=None, font=None, kde=True, color=None, alpha=0.85, annotate=True, annotate_format="{:,.0f}", show_mean_line=True, show_stats_box=True, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])
        mean_val = float(df[column].mean())
        median_val = float(df[column].median())
        color = color or self.config.LIGHT_BLUE
        sns.histplot(data=df, x=column, bins=bins, kde=kde, color=color, alpha=alpha, ax=ax, edgecolor="white", linewidth=1.5)
        if kde and hasattr(ax, "lines") and ax.lines:
            ax.lines[0].set_color(self.config.LIGHT_BLUE)
            ax.lines[0].set_linewidth(3)
        palette = self.config.get_palette(len(ax.patches))
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(palette[i])
        self._auto_adjust_ylim(ax, padding=ylim_padding)
        resolved_xlabel = xlabel or self.data.default_xlabel or column.replace("_", " ").title()
        resolved_ylabel = ylabel or self.data.default_ylabel or "Frequency"
        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        if annotate:
            self._annotate_bars(ax, annotate_format, font_config["value_label_size"])
        if show_mean_line:
            ax.axvline(mean_val, color=self.config.LIGHT_BLUE, linestyle="--", linewidth=2)
        if show_stats_box:
            ax.text(
                0.02,
                0.98,
                f"Mean: {mean_val:,.2f}\nMedian: {median_val:,.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_config["label_size"] * 0.9,
                color=self.config.DARK_BLUE,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=self.config.DARK_BLUE, alpha=0.9),
            )
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "histogram", save_path, save_formats)
        plt.close(fig)
        return fig, saved

class PieChartPlotter(BasePlotter):
    def plot(self, chart_name, column, size=(14, 10), font=None, annotate=False, annotate_format="{:.1f}%%", explode_top_n=0, explode_value=0.001, label=None, show_legend=True, legend_loc="center left", legend_bbox_to_anchor=(0.95, 0.5), colors=None, explode=None, startangle=90, pctdistance=0.8, max_slices=20, save_path=None, save_formats=None):
        df = self.data.get_data()
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        df = df.copy()
        s = df[column]
        counts = s.fillna("Missing").value_counts(dropna=False).rename_axis(column).reset_index(name="count")

        if len(counts) > max_slices:
            counts = counts.sort_values("count", ascending=False)
            top = counts.iloc[: max_slices - 1]
            other = pd.DataFrame({column: ["Other"], "count": [counts.iloc[max_slices - 1 :]["count"].sum()]})
            counts = pd.concat([top, other])

        labels = counts[column].astype(str).tolist()
        palette = colors or self.config.pie_palette(labels)

        if explode is None:
            explode = [(explode_value if i < explode_top_n else 0) for i in range(len(labels))]

        pie_kwargs = {
            "x": counts["count"].values,
            "colors": palette,
            "explode": explode,
            "startangle": startangle,
            "textprops": {"fontsize": font_config["tick_size"], "color": self.config.DARK_BLUE},
            "wedgeprops": {"linewidth": 1, "edgecolor": "white"},
            "autopct": None
        }

        result = ax.pie(**pie_kwargs)
        ax.axis("equal")
        ax.set_title(chart_name, fontsize=font_config["title_size"], fontweight="bold", color=self.config.DARK_BLUE)

        if show_legend:
            pct = (counts["count"] / counts["count"].sum() * 100).round(1)
            legend_labels = [f"{lbl} ({p:.1f}%)" for lbl, p in zip(labels, pct)]
            ax.legend(result[0], legend_labels, title="Categories", loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

        fig.subplots_adjust(left=0.05, right=0.85, top=0.88, bottom=0.05)

        saved = self._save_figure(fig, label or chart_name, "pie_chart", save_path, save_formats or ["png"])
        plt.close(fig)
        return fig, saved


class ScatterPlotPlotter(BasePlotter):
    def plot(self, chart_name, x_column, y_column, hue=None, size_column=None, xlabel=None, ylabel=None, size=None, point_size=40, alpha=1.0, minimal=True, show_regression=False, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        df = df.copy()
        df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
        df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
        df = df.dropna(subset=[x_column, y_column])
        size = size or (8, 6)
        fig, ax = plt.subplots(figsize=size)
        if minimal:
            ax.scatter(df[x_column], df[y_column], s=point_size, alpha=alpha)
            ax.set_title(chart_name)
            ax.set_xlabel(xlabel or x_column.replace("_", " ").title())
            ax.set_ylabel(ylabel or y_column.replace("_", " ").title())
        else:
            palette = self.config.get_palette(1)
            sns.scatterplot(data=df, x=x_column, y=y_column, color=palette[0], s=point_size, alpha=alpha, ax=ax)
            if show_regression:
                sns.regplot(data=df, x=x_column, y=y_column, scatter=False, ax=ax)
            self._style_axes(ax, chart_name, xlabel or x_column, ylabel or y_column, self.config.get_font_config())
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "scatter_plot", save_path, save_formats or ["png"])
        plt.close(fig)
        return fig, saved


class LineChartPlotter(BasePlotter):
    def plot(self, chart_name, x_column, y_columns, xlabel=None, ylabel=None, size=None, font=None, colors=None, markers=True, linewidth=2.5, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        palette = colors or self.config.get_palette(len(y_columns))
        for i, col in enumerate(y_columns):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            d = df[[x_column, col]].dropna()
            sns.lineplot(
                x=x_column,
                y=col,
                data=d,
                color=palette[i],
                linewidth=linewidth,
                marker="o" if markers else None,
                ax=ax,
                label=col.replace("_", " ").title(),
            )
        self._style_axes(ax, chart_name, xlabel or x_column.replace("_", " ").title(), ylabel or "Value", font_config)
        if len(y_columns) > 1:
            ax.legend()
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "line_chart", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class HeatmapPlotter(BasePlotter):
    def plot(self, chart_name="Correlation Heatmap", columns=None, size=(12, 10), font=None, cmap="coolwarm", annot=True, fmt=".2f", label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        df_numeric = df.select_dtypes(include=[np.number]) if columns is None else df[columns].select_dtypes(include=[np.number])
        corr = df_numeric.corr()
        fig, ax = plt.subplots(figsize=size)
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(chart_name, fontsize=self.config.TITLE_SIZE, fontweight="bold", color=self.config.DARK_BLUE)
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "heatmap", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class UnivariateScatterPlotter(BasePlotter):
    def plot(self, chart_name, column, size=None, point_size=40, alpha=0.9, jitter=0.06, orientation="v", label=None, save_path=None, save_formats=None, font=None):
        df = self.data.get_data()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])
        vals = df[column].values
        jitter_vals = np.random.normal(0, jitter, len(vals))
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        if orientation.lower() == "v":
            ax.scatter(vals, jitter_vals, s=point_size, alpha=alpha, color=self.config.LIGHT_BLUE)
            xlabel, ylabel = column.replace("_", " ").title(), ""
            ax.set_yticks([])
        else:
            ax.scatter(jitter_vals, vals, s=point_size, alpha=alpha, color=self.config.LIGHT_BLUE)
            xlabel, ylabel = "", column.replace("_", " ").title()
        self._style_axes(ax, chart_name, xlabel, ylabel, font_config)
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "univariate_scatter", save_path, save_formats or ["png"])
        plt.close(fig)
        return fig, saved


class OverlappingHistogramPlotter(BasePlotter):
    def plot(self, chart_name, value_column, group_column, xlabel=None, ylabel="Density",
             bins=20, size=None, font=None, colors=None, alpha=0.6, kde=True,
             kde_linewidth=2.5, show_legend=True, legend_loc="best",
             label=None, save_path=None, save_formats=None, max_groups=5):

        df = self.data.get_data().copy()
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna(subset=[value_column, group_column])

        groups = df[group_column].unique()
        if len(groups) > max_groups:
            top = df[group_column].value_counts().head(max_groups).index.tolist()
            df = df[df[group_column].isin(top)]
            groups = top

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        palette = colors or self.config.get_palette(len(groups))

        for i, g in enumerate(groups):
            d = df[df[group_column] == g][value_column]

            ax.hist(
                d,
                bins=bins,
                alpha=alpha,
                color=palette[i],
                label=str(g),
                edgecolor="white",
                linewidth=0.5,
                density=True
            )

            if kde and len(d) > 1 and d.nunique() > 1:
                try:
                    kde_obj = stats.gaussian_kde(d)
                    x_min, x_max = d.min(), d.max()
                    rng = x_max - x_min
                    xs = np.linspace(x_min - 0.1 * rng, x_max + 0.1 * rng, 200)
                    ax.plot(xs, kde_obj(xs), color=palette[i], linewidth=kde_linewidth)
                except Exception:
                    pass

        self._style_axes(ax, chart_name, xlabel or value_column, ylabel, font_config)
        ax.set_facecolor("#fafafa")

        if show_legend:
            ax.legend(title=group_column.replace("_", " ").title(), loc=legend_loc)

        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "overlapping_histogram", save_path, save_formats)
        plt.close(fig)

        return fig, saved


class RidgePlotter(BasePlotter):
    def plot(self, chart_name, value_column, group_column, xlabel=None, ylabel="Density",
             size=None, font=None, colors=None, fill_alpha=0.7, kde=True,
             kde_linewidth=2.5, show_grid=True, label=None, save_path=None, save_formats=None, max_groups=10):

        df = self.data.get_data().copy()
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna(subset=[value_column, group_column])

        groups = df[group_column].unique()
        if len(groups) > max_groups:
            top = df[group_column].value_counts().head(max_groups).index.tolist()
            df = df[df[group_column].isin(top)]
            groups = top

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        palette = colors or self.config.get_palette(len(groups))

        sorted_groups = df.groupby(group_column)[value_column].mean().sort_values(ascending=True).index

        y_offset = 0
        y_offsets = {}
        for i, g in enumerate(sorted_groups):
            d = df[df[group_column] == g][value_column]
            y_offsets[g] = y_offset

            if kde and len(d) > 1 and d.nunique() > 1:
                try:
                    kde_obj = stats.gaussian_kde(d)
                    x_min, x_max = d.min(), d.max()
                    rng = x_max - x_min
                    xs = np.linspace(x_min - 0.1 * rng, x_max + 0.1 * rng, 200)
                    kde_values = kde_obj(xs)
                    
                    max_density = kde_values.max() if len(kde_values) > 0 else 0
                    spacing = max(0.2, max_density * 1.5)
                    
                    ax.plot(xs, kde_values + y_offset, color=palette[i], linewidth=kde_linewidth)
                    if fill_alpha > 0:
                        ax.fill_between(xs, y_offset, kde_values + y_offset, alpha=fill_alpha, color=palette[i])
                except Exception:
                    pass
            
            y_offset += spacing

        ax.set_yticks(list(y_offsets.values()))
        ax.set_yticklabels([str(g) for g in sorted_groups])
        ax.set_ylabel(group_column.replace("_", " ").title(), fontsize=font_config["label_size"], fontweight="600", color=self.config.DARK_BLUE)

        self._style_axes(ax, chart_name, xlabel or value_column, "Groups", font_config)
        if not show_grid:
            ax.grid(False)

        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "ridge_plot", save_path, save_formats)
        plt.close(fig)

        return fig, saved


class LollipopChartPlotter(BasePlotter):
    def plot(self, chart_name, column, xlabel=None, ylabel=None, ylim_padding=None, label=None, color=None, size=None, font=None, sort_by="value", ascending=False, annotate=True, annotate_format="{:,.0f}", save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        counts = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
        
        if sort_by == "value":
            counts = counts.sort_values("count", ascending=ascending)
        else:
            counts = counts.sort_values(column, ascending=ascending)
            
        y_pos = np.arange(len(counts))
        color = color or self.config.LIGHT_BLUE
        
        ax.vlines(x=y_pos, ymin=0, ymax=counts["count"], colors=color, linestyles='-', lw=2)
        ax.scatter(y_pos, counts["count"], s=50, color=color, zorder=3)
        
        ax.set_xticks(y_pos)
        ax.set_xticklabels(counts[column].astype(str), rotation=45, ha="right")
        
        self._auto_adjust_ylim(ax, padding=ylim_padding)
        resolved_xlabel = xlabel or self.data.default_xlabel or column.replace("_", " ").title()
        resolved_ylabel = ylabel or self.data.default_ylabel
        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        
        if annotate:
            for i, v in enumerate(counts["count"]):
                if not np.isnan(v) and v > 0:
                    ax.text(i, v, annotate_format.format(v), ha="center", va="bottom", fontsize=font_config["value_label_size"], fontweight="bold", color=self.config.DARK_BLUE)
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "lollipop_chart", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class DivergingBarPlotter(BasePlotter):
    def plot(self, chart_name, positive_column, negative_column, categories_column, xlabel=None, ylabel=None, size=None, font=None, pos_color=None, neg_color=None, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        
        pos_color = pos_color or '#2E8B57'  
        neg_color = neg_color or '#DC143C'  

        x_pos = np.arange(len(df))
        width = 0.4

        bars1 = ax.barh(x_pos - width/2, df[positive_column], color=pos_color, label=positive_column.replace("_", " ").title(), height=width)
        bars2 = ax.barh(x_pos + width/2, -df[negative_column], color=neg_color, label=negative_column.replace("_", " ").title(), height=width)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(df[categories_column].astype(str))

        self._style_axes(ax, chart_name, xlabel or "Values", categories_column.replace("_", " ").title(), font_config)
        ax.legend(loc="lower right")

        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "diverging_bar_chart", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class StackedAreaPlotter(BasePlotter):
    def plot(self, chart_name, x_column, y_columns, xlabel=None, ylabel=None, size=None, font=None, colors=None, alpha=0.8, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        
        df_sorted = df.sort_values(by=x_column)
        x_vals = df_sorted[x_column]
        
        y_data = []
        for col in y_columns:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors="coerce")
            y_data.append(df_sorted[col].fillna(0).values)
        
        y_data = np.array(y_data)
        
        palette = colors or self.config.get_palette(len(y_columns))
        
        ax.stackplot(x_vals, *y_data, labels=y_columns, colors=palette, alpha=alpha)
        
        ax.set_title(chart_name, fontsize=font_config["title_size"], fontweight="bold", pad=20, color=self.config.DARK_BLUE, loc="left")
        ax.set_xlabel(xlabel or x_column.replace("_", " ").title(), fontsize=font_config["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.set_ylabel(ylabel or "Cumulative Value", fontsize=font_config["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.tick_params(axis="both", labelsize=font_config["tick_size"])
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
        ax.grid(False, axis="x")
        ax.set_facecolor("white")
        
        ax.legend(loc="upper left")
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "stacked_area_chart", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class KPICardPlotter(BasePlotter):
    def plot(self, chart_name, values, labels, size=(12, 6), font=None, label=None, save_path=None, save_formats=None):
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        n = len(values)
        rows = (n + 1) // 2
        height_per_row = 1.0 / rows
        
        for i, (val, lbl) in enumerate(zip(values, labels)):
            row_idx = i // 2
            col_idx = i % 2
            
            x_start = 0.05 + col_idx * 0.45
            y_start = 0.9 - row_idx * height_per_row
            
            ax.text(x_start, y_start, str(val), fontsize=font_config["title_size"]*1.2, fontweight="bold", color=self.config.DARK_BLUE, ha="left", va="top")
            ax.text(x_start, y_start - 0.05, lbl, fontsize=font_config["label_size"], color=self.config.DARK_BLUE, ha="left", va="top")
            
            rect = plt.Rectangle((x_start - 0.02, y_start - 0.15), 0.4, 0.05, linewidth=1, edgecolor=self.config.DARK_BLUE, facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title(chart_name, fontsize=font_config["title_size"], fontweight="bold", pad=20, color=self.config.DARK_BLUE, loc="left")
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "kpi_card", save_path, save_formats or ["png"])
        plt.close(fig)
        return fig, saved


class SmallMultiplesPlotter(BasePlotter):
    def plot(self, chart_name, x_column, y_column, group_column, chart_type="bar", xlabel=None, ylabel=None, size=None, font=None, colors=None, label=None, save_path=None, save_formats=None, max_groups=9):
        df = self.data.get_data()
        groups = df[group_column].unique()
        if len(groups) > max_groups:
            top = df[group_column].value_counts().head(max_groups).index.tolist()
            df = df[df[group_column].isin(top)]
            groups = top

        n_groups = len(groups)
        cols = 3
        rows = (n_groups + cols - 1) // cols
        figsize = size or (14, 5 * rows)
        
        font_config = self._apply_font_config(font)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_groups > 1 else [axes]
        else:
            axes = axes.flatten()

        palette = colors or self.config.get_palette(1)

        for i, g in enumerate(groups):
            sub_df = df[df[group_column] == g]
            ax = axes[i]
            
            if chart_type == "bar":
                counts = sub_df[x_column].value_counts(dropna=False).rename_axis(x_column).reset_index(name="count")
                sns.barplot(data=counts, x=x_column, y="count", ax=ax, color=palette[0])
                ax.set_title(f"{g}", fontsize=font_config["label_size"], fontweight="bold", color=self.config.DARK_BLUE)
                ax.set_xlabel(xlabel or x_column.replace("_", " ").title(), fontsize=font_config["tick_size"], color=self.config.DARK_BLUE)
                ax.set_ylabel(ylabel or "Count", fontsize=font_config["tick_size"], color=self.config.DARK_BLUE)
                
            elif chart_type == "hist":
                sub_df[y_column] = pd.to_numeric(sub_df[y_column], errors="coerce")
                sub_df_clean = sub_df.dropna(subset=[y_column])
                sns.histplot(data=sub_df_clean, x=y_column, ax=ax, color=palette[0])
                ax.set_title(f"{g}", fontsize=font_config["label_size"], fontweight="bold", color=self.config.DARK_BLUE)
                ax.set_xlabel(xlabel or y_column.replace("_", " ").title(), fontsize=font_config["tick_size"], color=self.config.DARK_BLUE)
                ax.set_ylabel(ylabel or "Frequency", fontsize=font_config["tick_size"], color=self.config.DARK_BLUE)

            ax.tick_params(axis="both", labelsize=font_config["tick_size"])
            sns.despine(ax=ax, top=True, right=True)
            ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
            ax.grid(False, axis="x")
            ax.set_facecolor("white")

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(chart_name, fontsize=font_config["title_size"], fontweight="bold", y=0.98, color=self.config.DARK_BLUE)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        saved = self._save_figure(fig, label or chart_name, "small_multiples", save_path, save_formats)
        plt.close(fig)
        return fig, saved


class HighlightTablePlotter(BasePlotter):
    def plot(self, chart_name, columns, group_column=None, metric_column=None, size=None, font=None, label=None, save_path=None, save_formats=None, max_rows=20):
        df = self.data.get_data()
        if group_column and metric_column:
            summary = df.groupby(group_column)[metric_column].agg(['mean', 'std']).reset_index()
            if len(summary) > max_rows:
                summary = summary.head(max_rows)
            data_to_plot = summary.set_index(group_column)
        else:
            data_to_plot = df[columns].head(max_rows).set_index(columns[0]) if len(columns) > 1 else df[columns].head(max_rows)

        size = size or (12, min(8, len(data_to_plot) * 0.5 + 1))
        font_config = self._apply_font_config(font)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('tight')
        ax.axis('off')

        table_data = data_to_plot.round(2).values
        col_labels = data_to_plot.columns.astype(str).tolist()
        row_labels = data_to_plot.index.astype(str).tolist()

        table = ax.table(cellText=table_data,
                         colLabels=col_labels,
                         rowLabels=row_labels,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(font_config["tick_size"])
        table.scale(1, 2)

        for (i, j), cell in table.get_celld().items():
            if i == 0:  
                cell.set_facecolor(self.config.DARK_BLUE)
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight='bold', color=self.config.DARK_BLUE)
            else:
                cell.set_facecolor('white')
                cell.set_text_props(color=self.config.DARK_BLUE)

        ax.set_title(chart_name, fontsize=font_config["title_size"], fontweight="bold", pad=20, color=self.config.DARK_BLUE, loc="left")

        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "highlight_table", save_path, save_formats or ["png"])
        plt.close(fig)
        return fig, saved

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from matplotlib import colors as mcolors
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class VisualizationConfig:
    DARK_BLUE = "#132249"
    LIGHT_BLUE = "#358cbf"
    PIE_PALETTE_NAME = "Blues"
    FONT_FAMILY = "Times New Roman"
    TITLE_SIZE = 24
    LABEL_SIZE = 18
    TICK_SIZE = 14
    VALUE_LABEL_SIZE = 12
    DEFAULT_SIZE = (14, 8)
    DPI_SCREEN = 300
    DPI_PRINT = 600

    @classmethod
    def get_palette(cls, n: int) -> List:
        cmap = LinearSegmentedColormap.from_list("alnayzak", [cls.LIGHT_BLUE, cls.DARK_BLUE], N=n)
        return [cmap(i / max(1, n - 1)) for i in range(n)]

    @classmethod
    def get_alnayzak_palette(cls, n: int) -> List:
        """Alias for get_palette for compatibility"""
        return cls.get_palette(n)

    @classmethod
    def pie_palette(cls, labels, emphasize=None):
        n = len(labels)
        raw = sns.color_palette(cls.PIE_PALETTE_NAME, n)
        pal = [mcolors.to_hex(c) for c in raw]
        if emphasize is not None and emphasize in labels:
            idx = labels.index(emphasize)
            pal[idx] = cls.DARK_BLUE
        return pal

    @staticmethod
    def _text_color_for(bg_hex: str) -> str:
        r, g, b = mcolors.to_rgb(bg_hex)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance > 0.6 else "#ffffff"

    @classmethod
    def get_font_config(cls, custom: Optional[Dict] = None) -> Dict:
        config = {
            "family": cls.FONT_FAMILY,
            "title_size": cls.TITLE_SIZE,
            "label_size": cls.LABEL_SIZE,
            "tick_size": cls.TICK_SIZE,
            "value_label_size": cls.VALUE_LABEL_SIZE,
        }
        if custom:
            config.update(custom)
        return config


class DataManager:
    def __init__(self, dataset_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                 default_xlabel: Optional[str] = None, default_ylabel: Optional[str] = "Count",
                 ylim_padding: float = 0.1, df: Optional[pd.DataFrame] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else Path("in_memory.csv")
        self.dataset_base = self._strip_suffixes(self.dataset_path.name)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            parent = self.dataset_path.parent
            self.output_dir = parent / f"{self.dataset_base}_run"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_frame = None
        self.default_xlabel = default_xlabel
        self.default_ylabel = default_ylabel
        self.ylim_padding = float(ylim_padding)

        if df is not None:
            self.set_dataframe(df)

    @staticmethod
    def _strip_suffixes(filename: str) -> str:
        base = filename.split(".", 1)[0].strip()
        return base or "dataset"

    def load_data(self, clean_columns: bool = True):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        self.data_frame = pd.read_csv(self.dataset_path)
        if clean_columns:
            self.data_frame.columns = (
                self.data_frame.columns.str.strip().str.replace("\n", " ", regex=True)
            )

    def set_dataframe(self, df: pd.DataFrame, clean_columns: bool = True):
        self.data_frame = df.copy()
        if clean_columns:
            self.data_frame.columns = (
                self.data_frame.columns.str.strip().str.replace("\n", " ", regex=True)
            )

    def get_data(self) -> pd.DataFrame:
        return self.data_frame.copy()

    def list_columns(self, column_type: Optional[str] = None) -> List[str]:
        if column_type == 'numeric':
            return self.data_frame.select_dtypes(include=[np.number]).columns.tolist()
        elif column_type == 'categorical':
            return self.data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
        return self.data_frame.columns.tolist()

    def quick_summary(self, numeric_only: bool = True) -> pd.DataFrame:
        return self.data_frame.describe() if numeric_only else self.data_frame.describe(include='all')


class BasePlotter:
    def __init__(self, data: DataManager, config: VisualizationConfig):
        self.data = data
        self.config = config

    def _slug(self, text: str) -> str:
        raw = str(text).strip()
        s = re.sub(r"\s+", "_", raw)
        s = re.sub(r"[^\w\-]", "", s)
        return re.sub(r"_+", "_", s).lower() or "chart"

    def _auto_adjust_ylim(self, ax, padding: Optional[float] = None):
        plt.draw()
        y_min, y_max = ax.get_ylim()
        expand_factor = 1.15 if padding is None else (1 + float(padding))
        new_y_max = y_max * expand_factor
        ax.set_ylim(y_min, new_y_max)

    def _save_figure(self, fig, label: str, chart_type: str, save_path=None, formats=None):
        formats = formats or ['png']
        saved_files = []

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            saved_files.append(save_path)
        else:
            base = f"{chart_type}_{self._slug(label)}_{self.data.dataset_base}"
            for fmt in formats:
                file_path = self.data.output_dir / f"{base}.{fmt}"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(file_path, dpi=300, bbox_inches="tight", facecolor="white")
                saved_files.append(file_path)

        return saved_files

    def _apply_font_config(self, font_config):
        cfg = self.config.get_font_config(font_config)
        plt.rcParams["font.family"] = cfg["family"]
        return cfg

    def _style_axes(self, ax, title, xlabel, ylabel, font):
        ax.set_title(title, fontsize=font["title_size"], fontweight="bold", pad=20, 
                    color=self.config.DARK_BLUE, loc="left")
        ax.set_xlabel(xlabel, fontsize=font["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.set_ylabel(ylabel, fontsize=font["label_size"], fontweight="600", color=self.config.DARK_BLUE)
        ax.tick_params(axis="both", labelsize=font["tick_size"])
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
        ax.grid(False, axis="x")
        ax.set_facecolor("white")

    def _annotate_bars(self, ax, fmt="{:,.0f}", font_size=None):
        font_size = font_size or self.config.VALUE_LABEL_SIZE
        for patch in ax.patches:
            height = patch.get_height()
            if np.isnan(height) or height <= 0:
                continue
            ax.text(patch.get_x() + patch.get_width() / 2, height, fmt.format(height),
                   ha="center", va="bottom", fontsize=font_size, fontweight="bold",
                   color=self.config.DARK_BLUE)

    def _clean_label(self, txt: str) -> str:
        """Format labels consistently."""
        if txt is None:
            return ""
        txt = str(txt).strip()
        txt = txt.replace("_", " ").title()
        return txt


# ============================================================================
# ADD THESE NEW PLOTTER CLASSES TO YOUR EXISTING EDA FILE
# Place these BEFORE the DataVisualizer class
# ============================================================================

class ComprehensiveTLSPlotter(BasePlotter):
    """TLS Distribution Horizontal Bar Chart"""
    
    def plot(self, tls_column="Current TLS", chart_name="TLS Locations", 
             size=(12, 8), font=None, label=None, save_path=None, save_formats=None):
        df = self.data.get_data()
        
        if tls_column not in df.columns:
            print(f"⚠ Column '{tls_column}' not found")
            return None, []
        
        font_config = self._apply_font_config(font)
        
        # Aggregate data
        counts = (df.groupby(tls_column).size()
                 .reset_index(name="Count")
                 .sort_values("Count", ascending=True))
        
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")
        
        # Colors
        n = len(counts)
        colors = self.config.get_palette(n)
        
        # Plot
        bars = ax.barh(counts[tls_column], counts["Count"], color=colors, 
                      edgecolor="white", linewidth=1.4)
        
        # Annotate
        for bar, val in zip(bars, counts["Count"]):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2, str(val),
                   va="center", fontsize=self.config.VALUE_LABEL_SIZE,
                   fontweight="bold", color=self.config.DARK_BLUE)
        
        self._style_axes(ax, chart_name, "Count", "TLS", font_config)
        ax.grid(True, axis="x", alpha=0.15)
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "tls_distribution", 
                                 save_path, save_formats)
        plt.close(fig)
        return fig, saved


# class ComprehensiveGeographicPlotter(BasePlotter):
#     """Two-panel: TLS counts + Average scores"""
    
#     def plot(self, tls_column="Current TLS", score_columns=None,
#              chart_name="Geographic Distribution", size=(20, 8), font=None,
#              label=None, save_path=None, save_formats=None):
        
#         df = self.data.get_data()
#         score_columns = score_columns or ["Arabic Score", "English Score", "Math Score"]
        
#         font_config = self._apply_font_config(font)
        
#         # Aggregate stats
#         agg_dict = {"Gender": "count"}
#         for col in score_columns:
#             if col in df.columns:
#                 agg_dict[col] = "mean"
        
#         stats = (df.groupby(tls_column)
#                 .agg(agg_dict)
#                 .rename(columns={"Gender": "Count"})
#                 .round(2)
#                 .reset_index())
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
#         fig.patch.set_facecolor("white")
        
#         # LEFT: Count per TLS
#         colors = self.config.get_palette(len(stats))
#         bars = ax1.bar(stats[tls_column], stats["Count"], color=colors, edgecolor="white")
        
#         for bar, val in zip(bars, stats["Count"]):
#             ax1.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
#                     ha="center", fontsize=self.config.VALUE_LABEL_SIZE,
#                     fontweight="bold", color=self.config.DARK_BLUE)
        
#         self._style_axes(ax1, "Student Count by TLS", "TLS", "Count", font_config)
#         ax1.tick_params(axis="x", rotation=45)
        
#         # RIGHT: Average scores
#         palette = self.config.get_palette(len(score_columns))
#         x = np.arange(len(stats))
#         width = 0.25
        
#         for i, col in enumerate(score_columns):
#             if col in stats.columns:
#                 ax2.bar(x + (i - 1)*width, stats[col], width,
#                        label=self._clean_label(col), color=palette[i],
#                        edgecolor="white", linewidth=1.2)
        
#         self._style_axes(ax2, "Average Scores by TLS", "TLS", "Average", font_config)
#         ax2.legend(frameon=True, fontsize=14)
#         ax2.set_xticks(x)
#         ax2.set_xticklabels(stats[tls_column], rotation=45, ha="right")
        
#         # Extend x-axis for legend
#         x_min, x_max = ax2.get_xlim()
#         ax2.set_xlim(x_min, x_max + (x_max * 0.30))
        
#         plt.tight_layout()
#         saved = self._save_figure(fig, label or chart_name, "geographic_dist", 
#                                  save_path, save_formats)
#         plt.close(fig)
#         return fig, saved

class ComprehensiveGeographicPlotter(BasePlotter):
    """Two-panel: TLS counts + Average scores (NO hardcoded 'Gender')"""

    def plot(
        self,
        tls_column="Current TLS",
        score_columns=None,
        chart_name="Geographic Distribution",
        size=(20, 8),
        font=None,
        label=None,
        save_path=None,
        save_formats=None,
    ):
        df = self.data.get_data()

        if tls_column not in df.columns:
            print(f"⚠ Column '{tls_column}' not found")
            return None, []

        score_columns = score_columns or []
        font_config = self._apply_font_config(font)

        # Count students per TLS (NO Gender needed)
        stats = df.groupby(tls_column).size().reset_index(name="Count")

        # Add mean scores if those columns exist
        for col in score_columns:
            if col in df.columns:
                stats[col] = df.groupby(tls_column)[col].mean().round(2)

        stats = stats.sort_values("Count", ascending=False).reset_index(drop=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
        fig.patch.set_facecolor("white")

        # LEFT: Count per TLS
        colors = self.config.get_palette(len(stats))
        bars = ax1.bar(stats[tls_column], stats["Count"], color=colors, edgecolor="white")

        for bar, val in zip(bars, stats["Count"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                str(int(val)),
                ha="center",
                fontsize=self.config.VALUE_LABEL_SIZE,
                fontweight="bold",
                color=self.config.DARK_BLUE,
            )

        self._style_axes(ax1, "Student Count by TLS", "TLS", "Count", font_config)
        ax1.tick_params(axis="x", rotation=45)

        # RIGHT: Average scores
        valid_scores = [c for c in score_columns if c in stats.columns]
        if valid_scores:
            palette = self.config.get_palette(len(valid_scores))
            x = np.arange(len(stats))
            width = 0.25 if len(valid_scores) <= 3 else 0.18

            for i, col in enumerate(valid_scores):
                ax2.bar(
                    x + (i - (len(valid_scores) - 1) / 2) * width,
                    stats[col].fillna(0),
                    width,
                    label=self._clean_label(col),
                    color=palette[i],
                    edgecolor="white",
                    linewidth=1.2,
                )

            self._style_axes(ax2, "Average Scores by TLS", "TLS", "Average", font_config)
            ax2.legend(frameon=True, fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels(stats[tls_column], rotation=45, ha="right")
        else:
            ax2.axis("off")
            ax2.text(
                0.5,
                0.5,
                "No numeric score columns detected",
                ha="center",
                va="center",
                fontsize=16,
                color=self.config.DARK_BLUE,
            )

        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "geographic_dist", save_path, save_formats)
        plt.close(fig)
        return fig, saved

class ComprehensiveDemographicsPlotter(BasePlotter):
    """Multi-panel demographics: Grade distribution, Gender pie, Grade×Gender"""
    
    def plot(self, grade_column="Grade", gender_column="Gender",
             chart_name="Student Demographics", size=(20, 12), font=None,
             label=None, save_path=None, save_formats=None):
        
        df = self.data.get_data()
        font_config = self._apply_font_config(font)
        
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.patch.set_facecolor("white")
        
        # TOP LEFT: Students by Grade
        ax1 = fig.add_subplot(gs[0, 0])
        dist = df.groupby(grade_column).size().reset_index(name="Count")
        colors = self.config.get_palette(len(dist))
        bars = ax1.bar(dist[grade_column], dist["Count"], color=colors, edgecolor="white")
        
        for bar, val in zip(bars, dist["Count"]):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
                    ha="center", fontsize=self.config.VALUE_LABEL_SIZE, fontweight="bold")
        
        self._style_axes(ax1, "Students by Grade", "Grade", "Count", font_config)
        
        # TOP RIGHT: Gender Pie
        ax2 = fig.add_subplot(gs[0, 1])
        if gender_column in df.columns:
            counts = df[gender_column].value_counts()
            colors = self.config.pie_palette(counts.index)
            ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors,
                   textprops={'fontsize': 14, 'fontweight': 'bold'})
            ax2.set_title("Gender Distribution", fontsize=self.config.TITLE_SIZE,
                         fontweight="bold", color=self.config.DARK_BLUE)
        
        # BOTTOM: Grade × Gender
        ax3 = fig.add_subplot(gs[1, :])
        if gender_column in df.columns:
            temp = df.groupby([grade_column, gender_column]).size().reset_index(name="Count")
            pivot = temp.pivot(index=grade_column, columns=gender_column, values="Count")
            
            x = np.arange(len(pivot.index))
            width = 0.35
            
            for i, gender in enumerate(pivot.columns):
                ax3.bar(x + (i - 0.5)*width, pivot[gender], width, label=gender,
                       color=self.config.get_palette(len(pivot.columns))[i],
                       edgecolor="white")
            
            self._style_axes(ax3, "Students by Gender & Grade", "Grade", "Count", font_config)
            ax3.set_xticks(x)
            ax3.set_xticklabels(pivot.index)
            ax3.legend()
            
            # Extend x-axis for legend
            x_min, x_max = ax3.get_xlim()
            ax3.set_xlim(x_min, x_max + (x_max * 0.30))
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "demographics", 
                                 save_path, save_formats)
        plt.close(fig)
        return fig, saved


class ComprehensiveSuccessRatesPlotter(BasePlotter):
    """Multi-panel success rates analysis"""
    
    def plot(self, score_columns=None, grade_column="Grade", gender_column="Gender",
             threshold=10, chart_name="Success Rates Analysis", size=(22, 12),
             font=None, label=None, save_path=None, save_formats=None):
        
        df = self.data.get_data()
        score_columns = score_columns or ["Arabic Score", "English Score", "Math Score"]
        font_config = self._apply_font_config(font)
        
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.patch.set_facecolor("white")
        
        palette = self.config.get_palette(len(score_columns))
        width = 0.25
        
        # TOP LEFT: Success by Grade
        ax1 = fig.add_subplot(gs[0, 0])
        rows = []
        for grade in df[grade_column].unique():
            entry = {grade_column: grade}
            grade_df = df[df[grade_column] == grade]
            for col in score_columns:
                if col in df.columns:
                    total = grade_df[col].notna().sum()
                    passed = (grade_df[col] >= threshold).sum()
                    entry[col] = 100 * passed / total if total else 0
            rows.append(entry)
        
        s_df = pd.DataFrame(rows)
        x = np.arange(len(s_df))
        
        for i, col in enumerate(score_columns):
            if col in s_df.columns:
                ax1.bar(x + (i - 1)*width, s_df[col], width,
                       label=self._clean_label(col), color=palette[i], edgecolor="white")
        
        self._style_axes(ax1, f"Success Rates ≥{threshold}/20", "Grade", 
                        "Success Rate (%)", font_config)
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(s_df[grade_column])
        
        x_min, x_max = ax1.get_xlim()
        ax1.set_xlim(x_min, x_max + (x_max * 0.30))
        
        # TOP RIGHT: Success by Gender
        ax2 = fig.add_subplot(gs[0, 1])
        if gender_column in df.columns:
            rows = []
            for gender in df[gender_column].unique():
                entry = {gender_column: gender}
                gender_df = df[df[gender_column] == gender]
                for col in score_columns:
                    if col in df.columns:
                        total = gender_df[col].notna().sum()
                        passed = (gender_df[col] >= threshold).sum()
                        entry[col] = 100 * passed / total if total else 0
                rows.append(entry)
            
            gdf = pd.DataFrame(rows)
            x = np.arange(len(gdf))
            
            for i, col in enumerate(score_columns):
                if col in gdf.columns:
                    ax2.bar(x + (i - 1)*width, gdf[col], width,
                           label=self._clean_label(col), color=palette[i], edgecolor="white")
            
            self._style_axes(ax2, "Success Rates by Gender", "Gender", 
                            "Success Rate (%)", font_config)
            ax2.set_xticks(x)
            ax2.set_xticklabels(gdf[gender_column])
            ax2.legend()
            
            x_min, x_max = ax2.get_xlim()
            ax2.set_xlim(x_min, x_max + (x_max * 0.30))
        
        # BOTTOM LEFT: Overall subject success
        ax3 = fig.add_subplot(gs[1, 0])
        overall = []
        for col in score_columns:
            if col in df.columns:
                total = df[col].notna().sum()
                passed = (df[col] >= threshold).sum()
                rate = 100 * passed / total if total else 0
                overall.append((self._clean_label(col), rate))
        
        if overall:
            labels, values = zip(*overall)
            bars = ax3.barh(labels, values, color=palette[:len(labels)], edgecolor="white")
            
            for bar, val in zip(bars, values):
                ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                        va="center", fontsize=self.config.VALUE_LABEL_SIZE, fontweight="bold")
            
            self._style_axes(ax3, "Overall Success Rates", "Success Rate (%)", 
                            "Subject", font_config)
        
        # BOTTOM RIGHT: Gender × Grade (first score column only)
        ax4 = fig.add_subplot(gs[1, 1])
        if gender_column in df.columns and score_columns:
            first_score = score_columns[0]
            rows = []
            for grade in df[grade_column].unique():
                for gender in df[gender_column].unique():
                    subset = df[(df[grade_column] == grade) & (df[gender_column] == gender)]
                    if first_score in df.columns:
                        total = subset[first_score].notna().sum()
                        passed = (subset[first_score] >= threshold).sum()
                        rate = 100 * passed / total if total else 0
                        rows.append({grade_column: grade, gender_column: gender, "Rate": rate})
            
            if rows:
                gdf = pd.DataFrame(rows)
                pivot = gdf.pivot(index=grade_column, columns=gender_column, values="Rate")
                
                x = np.arange(len(pivot.index))
                genders = list(pivot.columns)
                colors = self.config.get_palette(len(genders))
                
                for i, gender in enumerate(genders):
                    ax4.bar(x + (i - 0.5)*width, pivot[gender], width,
                           label=gender, color=colors[i], edgecolor="white")
                
                self._style_axes(ax4, f"{self._clean_label(first_score)} Success by Gender & Grade",
                                "Grade", "Success Rate (%)", font_config)
                ax4.set_xticks(x)
                ax4.set_xticklabels(pivot.index)
                ax4.legend()
                
                x_min, x_max = ax4.get_xlim()
                ax4.set_xlim(x_min, x_max + (x_max * 0.30))
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "success_rates", 
                                 save_path, save_formats)
        plt.close(fig)
        return fig, saved


class ComprehensiveScoreDistributionPlotter(BasePlotter):
    """KDE distributions across groups"""
    
    def plot(self, score_columns=None, group_column="Grade", 
             chart_name="Score Distributions by Grade", size=(22, 7),
             font=None, label=None, save_path=None, save_formats=None):
        
        df = self.data.get_data()
        score_columns = score_columns or ["Arabic Score", "English Score", "Math Score"]
        font_config = self._apply_font_config(font)
        
        fig, axes = plt.subplots(1, len(score_columns), figsize=size)
        fig.patch.set_facecolor("white")
        
        if len(score_columns) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, score_columns):
            if col not in df.columns:
                continue
            
            for grade in df[group_column].unique():
                data = df[df[group_column] == grade][col].dropna()
                if not data.empty:
                    sns.kdeplot(data=data, ax=ax, label=str(grade), linewidth=2.5)
            
            ax.set_title(self._clean_label(col), fontsize=self.config.TITLE_SIZE,
                        color=self.config.DARK_BLUE, fontweight="bold")
            ax.set_xlabel("Score", fontsize=self.config.LABEL_SIZE, color=self.config.DARK_BLUE)
            ax.set_ylabel("Density", fontsize=self.config.LABEL_SIZE, color=self.config.DARK_BLUE)
            ax.legend(frameon=True)
            sns.despine(ax=ax)
            ax.grid(True, linestyle="-", alpha=0.2)
        
        plt.tight_layout()
        saved = self._save_figure(fig, label or chart_name, "score_distributions",
                                 save_path, save_formats)
        plt.close(fig)
        return fig, saved


# ============================================================================
# UPDATE THE DataVisualizer CLASS - ADD THESE LINES IN __init__ method
# ============================================================================

# Add these lines after self.ht = HighlightTablePlotter(self.data, self.config):

"""
        self.comp_tls = ComprehensiveTLSPlotter(self.data, self.config)
        self.comp_geo = ComprehensiveGeographicPlotter(self.data, self.config)
        self.comp_demo = ComprehensiveDemographicsPlotter(self.data, self.config)
        self.comp_success = ComprehensiveSuccessRatesPlotter(self.data, self.config)
        self.comp_score_dist = ComprehensiveScoreDistributionPlotter(self.data, self.config)
"""

# ============================================================================
# ADD THESE METHODS TO DataVisualizer CLASS (after highlight_table method)
# ============================================================================

"""
    def comprehensive_tls_distribution(self, *a, **k):
        return self.comp_tls.plot(*a, **k)

    def comprehensive_geographic_dist(self, *a, **k):
        return self.comp_geo.plot(*a, **k)

    def comprehensive_demographics(self, *a, **k):
        return self.comp_demo.plot(*a, **k)

    def comprehensive_success_rates(self, *a, **k):
        return self.comp_success.plot(*a, **k)

    def comprehensive_score_distributions(self, *a, **k):
        return self.comp_score_dist.plot(*a, **k)
"""

class DataVisualizer:
    def __init__(self, dataset_path, output_dir=None, default_xlabel=None, default_ylabel="Count", ylim_padding=0.1, df=None):
        sns.set_theme(context="talk", style="whitegrid")
        self.config = VisualizationConfig()
        self.data = DataManager(dataset_path, output_dir, default_xlabel, default_ylabel, ylim_padding, df)
        self.bar = BarChartPlotter(self.data, self.config)
        self.hist = HistogramPlotter(self.data, self.config)
        self.pie = PieChartPlotter(self.data, self.config)
        self.scatter = ScatterPlotPlotter(self.data, self.config)
        self.line = LineChartPlotter(self.data, self.config)
        self.heatmap = HeatmapPlotter(self.data, self.config)
        self.uni_scatter = UnivariateScatterPlotter(self.data, self.config)
        self.overlapping_hist = OverlappingHistogramPlotter(self.data, self.config)
        self.ridge = RidgePlotter(self.data, self.config)
        self.lollipop = LollipopChartPlotter(self.data, self.config)
        self.diverging_bar = DivergingBarPlotter(self.data, self.config)
        self.stacked_area = StackedAreaPlotter(self.data, self.config)
        self.kpi = KPICardPlotter(self.data, self.config)
        self.sm = SmallMultiplesPlotter(self.data, self.config)
        self.ht = HighlightTablePlotter(self.data, self.config)

        self.comp_tls = ComprehensiveTLSPlotter(self.data, self.config)
        self.comp_geo = ComprehensiveGeographicPlotter(self.data, self.config)
        self.comp_demo = ComprehensiveDemographicsPlotter(self.data, self.config)
        self.comp_success = ComprehensiveSuccessRatesPlotter(self.data, self.config)
        self.comp_score_dist = ComprehensiveScoreDistributionPlotter(self.data, self.config)

    def load_data(self, clean_columns=True):
        self.data.load_data(clean_columns)

    def set_dataframe(self, df, clean_columns=True):
        self.data.set_dataframe(df, clean_columns)

    def get_data(self):
        return self.data.get_data()

    def bar_chart(self, *a, **k):
        return self.bar.plot(*a, **k)

    def histogram(self, *a, **k):
        return self.hist.plot(*a, **k)

    def pie_chart(self, *a, **k):
        return self.pie.plot(*a, **k)

    def scatter_plot(self, *a, **k):
        return self.scatter.plot(*a, **k)

    def line_chart(self, *a, **k):
        return self.line.plot(*a, **k)

    def correlation_heatmap(self, *a, **k):
        return self.heatmap.plot(*a, **k)

    def univariate_scatter(self, *a, **k):
        return self.uni_scatter.plot(*a, **k)

    def overlapping_histogram(self, *a, **k):
        return self.overlapping_hist.plot(*a, **k)

    def ridge_plot(self, *a, **k):
        return self.ridge.plot(*a, **k)

    def lollipop_chart(self, *a, **k):
        return self.lollipop.plot(*a, **k)

    def diverging_bar_chart(self, *a, **k):
        return self.diverging_bar.plot(*a, **k)

    def stacked_area_chart(self, *a, **k):
        return self.stacked_area.plot(*a, **k)

    def kpi_card(self, *a, **k):
        return self.kpi.plot(*a, **k)

    def small_multiples(self, *a, **k):
        return self.sm.plot(*a, **k)

    def highlight_table(self, *a, **k):
        return self.ht.plot(*a, **k)

    def plot_all_distributions(self, columns=None, **kwargs):
        df = self.get_data()
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        results = []
        for col in columns:
            try:
                fig, saved = self.hist.plot(f"Distribution of {col}", col, **kwargs)
                results.append((fig, saved))
            except:
                pass
        return results

    def plot_all_value_counts(self, columns=None, max_categories=20, **kwargs):
        df = self.get_data()
        columns = columns or df.select_dtypes(exclude=[np.number]).columns.tolist()
        results = []
        for col in columns:
            if df[col].nunique() <= max_categories:
                try:
                    fig, saved = self.bar.plot(f"Distribution of {col}", col, **kwargs)
                    results.append((fig, saved))
                except:
                    pass
        return results
    
    def comprehensive_tls_distribution(self, *a, **k):
        return self.comp_tls.plot(*a, **k)

    def comprehensive_geographic_dist(self, *a, **k):
        return self.comp_geo.plot(*a, **k)

    def comprehensive_demographics(self, *a, **k):
        return self.comp_demo.plot(*a, **k)

    def comprehensive_success_rates(self, *a, **k):
        return self.comp_success.plot(*a, **k)

    def comprehensive_score_distributions(self, *a, **k):
        return self.comp_score_dist.plot(*a, **k)