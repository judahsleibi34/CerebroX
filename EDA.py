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
    @classmethod
    def pie_palette(cls, labels, emphasize=None):
        n = len(labels)

        # Always generate EXACTLY n colors
        raw = sns.color_palette(cls.PIE_PALETTE_NAME, n)
        pal = [mcolors.to_hex(c) for c in raw]

        # Optional emphasize
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

            fig.savefig(save_path, dpi=300)   # ← ALWAYS 300 DPI, NO bbox_inches
            saved_files.append(save_path)

        else:
            base = f"{chart_type}_{self._slug(label)}_{self.data.dataset_base}"
            for fmt in formats:
                file_path = self.data.output_dir / f"{base}.{fmt}"
                file_path.parent.mkdir(parents=True, exist_ok=True)

                fig.savefig(file_path, dpi=300)   

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

class DataVisualizer:
    def __init__(self, dataset_path, output_dir=None, default_xlabel=None, default_ylabel="Count", ylim_padding=0.1, df=None):
        sns.set_theme(context="talk", style="whitegrid")
        self.config = VisualizationConfig()
        self.data = DataManager(dataset_path, output_dir, default_xlabel, default_ylabel, ylim_padding, df)
        self.bar = BarChartPlotter(self.data, self.config)
        self.hist = HistogramPlotter(self.data, self.config)
        self.pie = PieChartPlotter(self.data, self.config)
        self.grouped_bar = BarChartPlotter(self.data, self.config)
        self.box = HistogramPlotter(self.data, self.config)
        self.scatter = ScatterPlotPlotter(self.data, self.config)
        self.line = LineChartPlotter(self.data, self.config)
        self.heatmap = HeatmapPlotter(self.data, self.config)
        self.uni_scatter = UnivariateScatterPlotter(self.data, self.config)
        self.overlapping_hist = OverlappingHistogramPlotter(self.data, self.config)

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
