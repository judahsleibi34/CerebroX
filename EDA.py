# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
# import re
# from pathlib import Path
# from typing import Optional, Union, List, Dict, Tuple, Any
# import warnings
# from matplotlib import colors as mcolors

# warnings.filterwarnings('ignore')


# class VisualizationConfig:
#     DARK_BLUE = "#132249"
#     LIGHT_BLUE = "#358cbf"
#     PIE_PALETTE_NAME = "Blues"

#     FONT_FAMILY = "Times New Roman"
#     TITLE_SIZE = 24
#     LABEL_SIZE = 18
#     TICK_SIZE = 14
#     VALUE_LABEL_SIZE = 12
    
#     DEFAULT_SIZE = (14, 8)
    
#     DPI_SCREEN = 300
#     DPI_PRINT = 600
    
#     @classmethod
#     def get_palette(cls, n: int) -> List:
#         cmap = LinearSegmentedColormap.from_list(
#             "alnayzak", [cls.LIGHT_BLUE, cls.DARK_BLUE], N=n
#         )
#         return [cmap(i / max(1, n - 1)) for i in range(n)]
    
#     @classmethod
#     def pie_palette(cls, labels, emphasize=None):
#         n = len(labels)
#         raw = sns.color_palette(cls.PIE_PALETTE_NAME, n + 3)[1:-0 or None]
#         pal = [mcolors.to_hex(c) for c in raw]

#         if emphasize is not None and emphasize in labels:
#             i = list(labels).index(emphasize)
#             pal[i] = cls.DARK_BLUE
#         return pal

#     @staticmethod
#     def _text_color_for(bg_hex: str) -> str:
#         r, g, b = mcolors.to_rgb(bg_hex)
#         luminance = 0.299 * r + 0.587 * g + 0.114 * b
#         return "#000000" if luminance > 0.6 else "#ffffff"

#     @classmethod
#     def get_alnayzak_palette(cls, n: int) -> List[str]:

#         base_colors = [
#             cls.DARK_BLUE,
#             cls.LIGHT_BLUE,
#             "#1A3A6E",  
#             "#4CA6D8",  
#             "#86CAE8",  
#             "#FFFFFF"   
#         ]
#         k = (n + len(base_colors) - 1) // len(base_colors)
#         return (base_colors * k)[:n]
    
#     @classmethod
#     def get_font_config(cls, custom: Optional[Dict] = None) -> Dict:
#         config = {
#             "family": cls.FONT_FAMILY,
#             "title_size": cls.TITLE_SIZE,
#             "label_size": cls.LABEL_SIZE,
#             "tick_size": cls.TICK_SIZE,
#             "value_label_size": cls.VALUE_LABEL_SIZE,
#         }
#         if custom:
#             config.update(custom)
#         return config


# class DataVisualizer:
#     def __init__(
#         self,
#         dataset_path: Union[str, Path],
#         output_dir: Optional[Union[str, Path]] = None,
#         default_xlabel: Optional[str] = None,
#         default_ylabel: Optional[str] = "Count",
#         ylim_padding: float = 0.1,
#         df: Optional[pd.DataFrame] = None,  
#     ):

#         self.dataset_path = Path(dataset_path) if dataset_path else Path("in_memory.csv")
#         self.dataset_base = self._strip_suffixes(self.dataset_path.name)
        
#         if output_dir:
#             self.output_dir = Path(output_dir)
#         else:
#             self.output_dir = self.dataset_path.parent / self.dataset_base
        
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.data_frame: Optional[pd.DataFrame] = None
#         self.default_xlabel = default_xlabel
#         self.default_ylabel = default_ylabel
#         self.ylim_padding = float(ylim_padding)
        
#         sns.set_theme(context="talk", style="whitegrid")
        
#         self.config = VisualizationConfig()
#         if df is not None:
#             self.set_dataframe(df)
    
#     def _strip_suffixes(self, filename: str) -> str:
#         base = filename.split(".", 1)[0].strip()
#         return base or "dataset"
    
#     def _slug(self, text: str) -> str:
#         raw = str(text).strip()
#         s = re.sub(r"\s+", "_", raw)
#         s = re.sub(r"[^\w\-]", "", s)
#         s = re.sub(r"_+", "_", s).lower() or "chart"
#         return s
    
#     def _auto_adjust_ylim(self, ax, padding: Optional[float] = None):
#         plt.draw()
#         y_min, y_max = ax.get_ylim()
#         pad_ratio = self.ylim_padding if padding is None else float(padding)
#         pad = (y_max - y_min) * max(0.0, pad_ratio)
#         ax.set_ylim(y_min, y_max + pad)
    
#     def _save_figure(
#         self, 
#         fig, 
#         label: str, 
#         chart_type: str = "chart",
#         save_path: Optional[Path] = None,
#         formats: List[str] = None
#     ) -> List[Path]:
        
#         if formats is None:
#             formats = ['png']
        
#         saved_files = []
        
#         if save_path:
#             save_path = Path(save_path)
#             save_path.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(save_path, dpi=self.config.DPI_PRINT, bbox_inches="tight")
#             saved_files.append(save_path)
#         else:
#             base_label = f"{chart_type}_{self._slug(label)}"
#             base = f"{base_label}_{self.dataset_base}"
            
#             for fmt in formats:
#                 dpi = self.config.DPI_SCREEN if fmt == 'jpg' else self.config.DPI_PRINT
#                 file_path = self.output_dir / f"{base}.{fmt}"
#                 fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
#                 saved_files.append(file_path)
        
#         return saved_files
    
#     def _apply_font_config(self, font_config: Optional[Dict] = None):
#         config = self.config.get_font_config(font_config)
#         plt.rcParams["font.family"] = config["family"]
#         return config
    
#     def _style_axes(
#         self, 
#         ax, 
#         title: str, 
#         xlabel: str, 
#         ylabel: str,
#         font_config: Dict,
#         grid: bool = True
#     ):
        
#         ax.set_title(
#             title, 
#             fontsize=font_config["title_size"], 
#             fontweight="bold", 
#             pad=20, 
#             color=self.config.DARK_BLUE, 
#             loc="left"
#         )
#         ax.set_xlabel(
#             xlabel, 
#             fontsize=font_config["label_size"], 
#             fontweight="600", 
#             color=self.config.DARK_BLUE
#         )
#         ax.set_ylabel(
#             ylabel, 
#             fontsize=font_config["label_size"], 
#             fontweight="600", 
#             color=self.config.DARK_BLUE
#         )
#         ax.tick_params(axis="both", labelsize=font_config["tick_size"])
        
#         sns.despine(ax=ax, top=True, right=True)
        
#         if grid:
#             ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
#             ax.set_axisbelow(True)
        
#         ax.set_facecolor("white")
    
#     def _annotate_bars(
#         self, 
#         ax, 
#         annotate_format: str = "{:,.0f}",
#         font_size: Optional[int] = None
#     ):

#         if font_size is None:
#             font_size = self.config.VALUE_LABEL_SIZE
        
#         for patch in ax.patches:
#             height = patch.get_height()
#             if np.isnan(height) or height <= 0:
#                 continue
            
#             ax.text(
#                 patch.get_x() + patch.get_width() / 2,
#                 height,
#                 annotate_format.format(height),
#                 ha="center",
#                 va="bottom",
#                 fontsize=font_size,
#                 fontweight="bold",
#                 color=self.config.DARK_BLUE,
#             )
    
#     def load_data(self, clean_columns: bool = True):
#         if not self.dataset_path.exists():
#             raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
#         self.data_frame = pd.read_csv(self.dataset_path)
        
#         if clean_columns:
#             self.data_frame.columns = (
#                 self.data_frame.columns
#                 .str.strip()
#                 .str.replace("\n", " ", regex=True)
#             )
        
#         print(f"✓ Loaded {len(self.data_frame)} rows from {self.dataset_path.name}")
    
#     def set_dataframe(self, df: pd.DataFrame, clean_columns: bool = True):
#         self.data_frame = df.copy()
        
#         if clean_columns:
#             self.data_frame.columns = (
#                 self.data_frame.columns
#                 .str.strip()
#                 .str.replace("\n", " ", regex=True)
#             )
    
#     def get_data(self) -> pd.DataFrame:
#         if self.data_frame is None:
#             raise ValueError("No data loaded. Call load_data() first.")
#         return self.data_frame.copy()

#     def bar_chart(
#         self,
#         chart_name: str,
#         column: str,
#         *,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         ylim_padding: Optional[float] = None,
#         label: Optional[str] = None,
#         colors: Optional[List] = None,
#         size: Tuple[int, int] = None,
#         font: Optional[Dict] = None,
#         order: Optional[List] = None,
#         sort_by: str = "x",
#         ascending: bool = True,
#         annotate: bool = True,
#         annotate_format: str = "{:,.0f}",
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
        
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         df = self.data_frame.copy()
#         counts = (
#             df[column]
#             .value_counts(dropna=False)
#             .rename_axis(column)
#             .reset_index(name="count")
#         )
        
#         if sort_by == "y":
#             counts = counts.sort_values("count", ascending=ascending)
#         else:
#             counts = counts.sort_values(column, ascending=ascending)
        
#         order = order or counts[column].tolist()
#         palette = colors or self.config.get_palette(len(order))
        
#         sns.barplot(
#             data=counts,
#             x=column,
#             y="count",
#             order=order,
#             ci=None,
#             ax=ax,
#             palette=palette,
#         )
        
#         self._auto_adjust_ylim(ax, padding=ylim_padding)
        
#         resolved_xlabel = xlabel or self.default_xlabel or column.replace("_", " ").title()
#         resolved_ylabel = ylabel or self.default_ylabel
        
#         self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        
#         if annotate:
#             self._annotate_bars(ax, annotate_format, font_config["value_label_size"])
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig, 
#             label or chart_name, 
#             "bar_chart",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(5)
#         plt.close(fig)
        
#         return fig, saved
    
#     def histogram(
#         self,
#         chart_name: str,
#         column: str,
#         *,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         ylim_padding: Optional[float] = None,
#         bins: int = 20,
#         size: Tuple[int, int] = None,
#         font: Optional[Dict] = None,
#         kde: bool = True,
#         color: Optional[str] = None,
#         alpha: float = 0.85,
#         annotate: bool = True,
#         annotate_format: str = "{:,.0f}",
#         show_mean_line: bool = True,
#         show_stats_box: bool = True,
#         label: Optional[str] = None,
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
        
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         df = self.data_frame.copy()
#         df[column] = pd.to_numeric(df[column], errors="coerce")
#         df = df.dropna(subset=[column])
        
#         if len(df) == 0:
#             raise ValueError(f"No valid numeric data in column '{column}'")
        
#         mean_val = float(df[column].mean())
#         median_val = float(df[column].median())
        
#         color = color or self.config.LIGHT_BLUE
#         sns.histplot(
#             data=df,
#             x=column,
#             bins=bins,
#             kde=kde,
#             color=color,
#             alpha=alpha,
#             ax=ax,
#             edgecolor="white",
#             linewidth=1.5,
#         )
        
#         if kde and hasattr(ax, "lines") and len(ax.lines) > 0:
#             ax.lines[0].set_color(self.config.LIGHT_BLUE)
#             ax.lines[0].set_linewidth(3)
        
#         n_patches = len(ax.patches)
#         if n_patches > 0:
#             palette = self.config.get_palette(n_patches)
#             for i, patch in enumerate(ax.patches):
#                 patch.set_facecolor(palette[i])
        
#         self._auto_adjust_ylim(ax, padding=ylim_padding)
        
#         resolved_xlabel = xlabel or self.default_xlabel or column.replace("_", " ").title()
#         resolved_ylabel = ylabel or self.default_ylabel or "Frequency"
        
#         self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
#         ax.set_facecolor("#fafafa")
        
#         if annotate:
#             self._annotate_bars(ax, annotate_format, font_config["value_label_size"])
        
#         if show_mean_line:
#             ax.axvline(
#                 mean_val, 
#                 color=self.config.LIGHT_BLUE, 
#                 linestyle="--", 
#                 linewidth=2, 
#                 label=f"Mean = {mean_val:,.2f}"
#             )
        
#         if show_stats_box:
#             stats_text = f"Mean: {mean_val:,.2f}\nMedian: {median_val:,.2f}"
#             ax.text(
#                 0.02, 0.98, stats_text,
#                 transform=ax.transAxes,
#                 ha="left", va="top",
#                 fontsize=font_config["label_size"] * 0.9,
#                 color=self.config.DARK_BLUE,
#                 bbox=dict(
#                     boxstyle="round,pad=0.35", 
#                     facecolor="white", 
#                     edgecolor=self.config.DARK_BLUE, 
#                     alpha=0.9
#                 ),
#             )
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "histogram",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(5)
#         plt.close(fig)
        
#         return fig, saved
    
#     def pie_chart(
#         self,
#         chart_name: str,
#         column: str,
#         *,
#         size: Tuple[int, int] = (8, 6),
#         font: Optional[Dict] = None,
#         annotate: bool = False,
#         annotate_format: str = "{:.1f}%",
#         explode_top_n: int = 0,
#         explode_value: float = 0.001,
#         label: Optional[str] = None,
#         show_legend: bool = True,
#         legend_loc: str = "center left",
#         legend_bbox_to_anchor: Tuple = (0.95, 0.5),
#         colors: Optional[List] = None,
#         explode: Optional[List] = None,
#         startangle: float = 90,
#         pctdistance: float = 0.8,
#         max_slices: int = 20,
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#         ) -> Tuple[Any, List[Path]]:

#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")

#         font_config = self._apply_font_config(font)

#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
#         ax.set_facecolor("white")

#         df = self.data_frame.copy()

#         s = df[column]

#         if pd.api.types.is_categorical_dtype(s):
#             cats = list(s.cat.categories)
#             extras = [v for v in s.dropna().unique() if v not in cats]
#             need = (["Missing"] if "Missing" not in cats else []) + [x for x in extras if x is not None]
#             if need:
#                 s = s.cat.add_categories(need)


#         counts = (
#             s.fillna("Missing")
#             .value_counts(dropna=False)
#             .rename_axis(column)
#             .reset_index(name="count")
#         )


#         if len(counts) > max_slices:
#             counts = counts.sort_values("count", ascending=False)
#             top = counts.iloc[:max_slices - 1]
#             other = pd.DataFrame({column: ["Other"],
#                                 "count": [counts.iloc[max_slices - 1:]["count"].sum()]})
#             counts = pd.concat([top, other], ignore_index=True)

#         label_map = {1: "Passed", 0: "Failed"}
#         counts[column] = counts[column].replace(label_map).astype(str)

#         order = None
#         if {"Passed", "Failed"}.issubset(set(counts[column])):
#             order = ["Passed", "Failed"]
#         if order:
#             counts = counts.set_index(column).reindex(order).dropna().reset_index()

#         total = int(counts["count"].sum())
#         if total == 0:
#             raise ValueError(f"No data to plot for column '{column}'")

#         labels = counts[column].tolist()
#         emphasize_label = "Passed" if "Passed" in labels else None
#         palette = colors or self.config.pie_palette(labels, emphasize=emphasize_label)

#         if explode is None:
#             explode = [explode_value if i < explode_top_n else 0.0 for i in range(len(labels))]

#         pie_kwargs = dict(
#             x=counts["count"].values,
#             colors=palette,
#             explode=explode,
#             startangle=startangle,
#             textprops={"fontsize": font_config["tick_size"], "color": self.config.DARK_BLUE},
#             wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
#         )
#         if annotate:
#             pie_kwargs["autopct"] = annotate_format
#             pie_kwargs["pctdistance"] = pctdistance

#         res = ax.pie(**pie_kwargs)
#         if annotate:
#             wedges, texts, autotexts = res
#             for w, t in zip(wedges, autotexts):
#                 face = mcolors.to_hex(w.get_facecolor())
#                 t.set_color(self.config._text_color_for(face))
#                 t.set_fontweight("bold")
#         else:
#             wedges, texts = res  

#         ax.axis("equal")

#         ax.set_title(
#             chart_name,
#             fontsize=font_config["title_size"],
#             fontweight="bold",
#             color=self.config.DARK_BLUE,
#             loc="center",
#             pad=10,
#         )

#         if show_legend:
#             pct = (counts["count"] / total * 100).round(1)
#             legend_labels = [f"{lbl} ({p:.1f}%)" for lbl, p in zip(labels, pct)]
#             n_items = len(legend_labels)
#             ncol = 1 if n_items <= 10 else (2 if n_items <= 18 else 3)
#             leg = ax.legend(
#                 handles=wedges,
#                 labels=legend_labels,
#                 title="Categories",
#                 loc=legend_loc,
#                 bbox_to_anchor=legend_bbox_to_anchor,
#                 fontsize=font_config["tick_size"],
#                 frameon=True,
#                 ncol=ncol,
#                 columnspacing=1.2,
#                 handlelength=1.2,
#                 labelspacing=0.6,
#                 borderpad=0.8,
#             )
#             plt.setp(leg.get_title(), fontsize=font_config["tick_size"], fontweight="bold")

#         fig.subplots_adjust(left=0.02, right=0.75, top=0.88, bottom=0.08)

#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "pie_chart",
#             save_path,
#             save_formats or ["png"],
#         )
#         plt.close(fig)
#         return fig, saved


    
#     def grouped_bar_chart(
#         self,
#         chart_name: str,
#         group_col: str,
#         score_cols: List[str],
#         *,
#         agg_func: str = "mean",
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         ylim: Optional[Tuple] = None,
#         ylim_padding: Optional[float] = None,
#         label: Optional[str] = None,
#         colors: Optional[List] = None,
#         size: Tuple[int, int] = None,
#         font: Optional[Dict] = None,
#         ci: Optional[int] = None,
#         dodge: bool = True,
#         annotate: bool = True,
#         annotate_format: str = "{:,.1f}",
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
    
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         df = self.data_frame.copy()
#         if group_col not in df.columns:
#             raise ValueError(f"Group column '{group_col}' not found")
        
#         missing = [c for c in score_cols if c not in df.columns]
#         if missing:
#             raise ValueError(f"Score columns not found: {missing}")
        
#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         df_long = df.melt(
#             id_vars=[group_col],
#             value_vars=score_cols,
#             var_name="Subject",
#             value_name="Score"
#         ).dropna(subset=["Score"])
        
#         grouped = df_long.groupby([group_col, "Subject"], as_index=False).agg(
#             value=("Score", agg_func)
#         )
        
#         x_order = grouped[group_col].dropna().unique().tolist()
#         hue_order = grouped["Subject"].dropna().unique().tolist()
        
#         palette = colors or self.config.get_palette(len(hue_order))
        
#         sns.barplot(
#             data=grouped,
#             x=group_col,
#             y="value",
#             hue="Subject",
#             order=x_order,
#             hue_order=hue_order,
#             ci=ci,
#             dodge=dodge,
#             ax=ax,
#             palette=palette
#         )
        
#         if ylim is not None:
#             ax.set_ylim(*ylim)
#         else:
#             max_val = grouped["value"].max()
#             pad_ratio = self.ylim_padding if ylim_padding is None else float(ylim_padding)
#             padding = max_val * pad_ratio
#             ax.set_ylim(0, max_val + padding)
        
#         display_name = {
#             "mean": "Average",
#             "median": "Median",
#             "sum": "Total",
#             "count": "Count"
#         }.get(agg_func, agg_func.title())
        
#         resolved_xlabel = xlabel or group_col.replace("_", " ").title()
#         resolved_ylabel = ylabel or display_name
        
#         self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        
#         ax.legend(title=None, frameon=True, loc='upper right')
        
#         if annotate:
#             self._annotate_bars(ax, annotate_format, font_config["value_label_size"])
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "grouped_bar_chart",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(3)
#         plt.close(fig)
        
#         return fig, saved
    
#     def multiple_bar_charts(self, *args, **kwargs):
#         """Alias for grouped_bar_chart (backward compatibility)."""
#         return self.grouped_bar_chart(*args, **kwargs)
    
    
#     def quick_summary(self, numeric_only: bool = True) -> pd.DataFrame:
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         if numeric_only:
#             return self.data_frame.describe()
#         else:
#             return self.data_frame.describe(include='all')
    
#     def plot_all_distributions(
#         self,
#         columns: Optional[List[str]] = None,
#         **kwargs
#     ) -> List[Tuple[Any, List[Path]]]:
        
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         df = self.data_frame.copy()
        
#         if columns is None:
#             columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
#         results = []
#         for col in columns:
#             try:
#                 fig, saved = self.histogram(
#                     chart_name=f"Distribution of {col}",
#                     column=col,
#                     **kwargs
#                 )
#                 results.append((fig, saved))
#                 print(f"✓ Created histogram for {col}")
#             except Exception as e:
#                 print(f"⚠ Warning: Could not plot {col}: {e}")
        
#         return results
    
#     def plot_all_value_counts(
#         self,
#         columns: Optional[List[str]] = None,
#         max_categories: int = 20,
#         **kwargs
#     ) -> List[Tuple[Any, List[Path]]]:
    
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         df = self.data_frame.copy()
        
#         if columns is None:
#             columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
#         results = []
#         for col in columns:
#             try:
#                 n_unique = df[col].nunique()
#                 if n_unique > max_categories:
#                     print(f"⚠ Skipping {col}: too many categories ({n_unique})")
#                     continue
                
#                 fig, saved = self.bar_chart(
#                     chart_name=f"Distribution of {col}",
#                     column=col,
#                     **kwargs
#                 )
#                 results.append((fig, saved))
#                 print(f"✓ Created bar chart for {col}")
#             except Exception as e:
#                 print(f"⚠ Warning: Could not plot {col}: {e}")
        
#         return results
    
#     def correlation_heatmap(
#         self,
#         columns: Optional[List[str]] = None,
#         chart_name: str = "Correlation Heatmap",
#         size: Tuple[int, int] = (12, 10),
#         font: Optional[Dict] = None,
#         cmap: str = "coolwarm",
#         annot: bool = True,
#         fmt: str = ".2f",
#         label: Optional[str] = None,
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
        
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         font_config = self._apply_font_config(font)
        
#         df = self.data_frame.copy()
#         if columns is None:
#             df_numeric = df.select_dtypes(include=[np.number])
#         else:
#             df_numeric = df[columns].select_dtypes(include=[np.number])
        
#         if df_numeric.empty:
#             raise ValueError("No numeric columns found for correlation")
        
#         corr = df_numeric.corr()
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         sns.heatmap(
#             corr,
#             annot=annot,
#             fmt=fmt,
#             cmap=cmap,
#             center=0,
#             square=True,
#             linewidths=1,
#             cbar_kws={"shrink": 0.8},
#             ax=ax
#         )
        
#         ax.set_title(
#             chart_name,
#             fontsize=font_config["title_size"],
#             fontweight="bold",
#             color=self.config.DARK_BLUE,
#             pad=20
#         )
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "heatmap",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(3)
#         plt.close(fig)
        
#         return fig, saved
    
#     def box_plot(
#         self,
#         chart_name: str,
#         column: str,
#         group_by: Optional[str] = None,
#         *,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         size: Tuple[int, int] = None,
#         font: Optional[Dict] = None,
#         colors: Optional[List] = None,
#         label: Optional[str] = None,
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
    
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         # Setup
#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         # Prepare data
#         df = self.data_frame.copy()
#         df[column] = pd.to_numeric(df[column], errors="coerce")
#         df = df.dropna(subset=[column])
        
#         # Create plot
#         if group_by:
#             palette = colors or self.config.get_palette(df[group_by].nunique())
#             sns.boxplot(
#                 data=df,
#                 x=group_by,
#                 y=column,
#                 palette=palette,
#                 ax=ax
#             )
#             resolved_xlabel = xlabel or group_by.replace("_", " ").title()
#         else:
#             color = colors[0] if colors else self.config.LIGHT_BLUE
#             sns.boxplot(
#                 data=df,
#                 y=column,
#                 color=color,
#                 ax=ax
#             )
#             resolved_xlabel = xlabel or ""
        
#         resolved_ylabel = ylabel or column.replace("_", " ").title()
        
#         self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "box_plot",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(3)
#         plt.close(fig)
        
#         return fig, saved
    
#     def scatter_plot(
#         self,
#         chart_name: str,
#         x_column: str,
#         y_column: str,
#         *,
#         hue: str = None,
#         size_column: str = None,
#         xlabel: str = None,
#         ylabel: str = None,
#         size: tuple = None,
#         point_size: int = 40,
#         alpha: float = 1.0,
#         minimal: bool = True,
#         show_regression: bool = False,
#         label: str = None,
#         save_path=None,
#         save_formats=None,
#         **kwargs,  
#     ):
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")

#         df = self.data_frame.copy()
#         df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
#         df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
#         df = df.dropna(subset=[x_column, y_column])

#         size = size or (8, 6)
#         fig, ax = plt.subplots(figsize=size)

#         if minimal:
#             ax.grid(False)
#             fig.patch.set_facecolor("white")
#             ax.set_facecolor("white")

#             if size_column and size_column in df.columns:
#                 s_vals = pd.to_numeric(df[size_column], errors="coerce").fillna(0)
#                 ax.scatter(df[x_column], df[y_column], s=s_vals, alpha=alpha)
#             else:
#                 ax.scatter(df[x_column], df[y_column], s=point_size, alpha=alpha)
#             ax.set_title(chart_name, pad=10)
#             ax.set_xlabel(xlabel or x_column.replace("_", " ").title())
#             ax.set_ylabel(ylabel or y_column.replace("_", " ").title())

#             show_regression = False
#             hue = None

#         else:
#             palette = self.config.get_palette(1)
#             sns.scatterplot(
#                 data=df, x=x_column, y=y_column, color=palette[0], s=point_size, alpha=alpha, ax=ax
#             )
#             if show_regression:
#                 sns.regplot(data=df, x=x_column, y=y_column, scatter=False, ax=ax)

#             self._style_axes(
#                 ax,
#                 chart_name,
#                 xlabel or x_column.replace("_", " ").title(),
#                 ylabel or y_column.replace("_", " ").title(),
#                 self._apply_font_config(None),
#             )

#         plt.tight_layout()
#         saved = self._save_figure(
#             fig, label or chart_name, "scatter_plot", save_path, save_formats or ["png"]
#         )
#         plt.close(fig)
#         return fig, saved


    
#     def line_chart(
#         self,
#         chart_name: str,
#         x_column: str,
#         y_columns: Union[str, List[str]],
#         *,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         size: Tuple[int, int] = None,
#         font: Optional[Dict] = None,
#         colors: Optional[List] = None,
#         markers: bool = True,
#         linewidth: float = 2.5,
#         label: Optional[str] = None,
#         save_path: Optional[Path] = None,
#         save_formats: List[str] = None,
#     ) -> Tuple[Any, List[Path]]:
        
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)
        
#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
        
#         df = self.data_frame.copy()
        
#         if isinstance(y_columns, str):
#             y_columns = [y_columns]
        
#         palette = colors or self.config.get_palette(len(y_columns))
        
#         for i, y_col in enumerate(y_columns):
#             df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
#             data = df[[x_column, y_col]].dropna()
            
#             plot_args = {
#                 "x": x_column,
#                 "y": y_col,
#                 "data": data,
#                 "color": palette[i],
#                 "linewidth": linewidth,
#                 "label": y_col.replace("_", " ").title(),
#                 "ax": ax,
#             }
            
#             if markers:
#                 plot_args["marker"] = "o"
#                 plot_args["markersize"] = 6
            
#             sns.lineplot(**plot_args)
        
#         resolved_xlabel = xlabel or x_column.replace("_", " ").title()
#         resolved_ylabel = ylabel or "Value"
        
#         self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        
#         if len(y_columns) > 1:
#             ax.legend(frameon=True, loc='best')
        
#         plt.tight_layout()
        
#         saved = self._save_figure(
#             fig,
#             label or chart_name,
#             "line_chart",
#             save_path,
#             save_formats
#         )
        
#         plt.show()
#         plt.pause(3)
#         plt.close(fig)
        
#         return fig, saved
    
#     def get_info(self) -> None:
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         print(f"\n{'='*80}")
#         print(f"Dataset: {self.dataset_path.name}")
#         print(f"{'='*80}")
#         print(f"Shape: {self.data_frame.shape[0]:,} rows × {self.data_frame.shape[1]} columns")
#         print(f"Output directory: {self.output_dir}")
#         print(f"\nColumn types:")
#         print(self.data_frame.dtypes.value_counts())
#         print(f"\nMemory usage: {self.data_frame.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
#         print(f"{'='*80}\n")
    
#     def list_columns(self, column_type: Optional[str] = None) -> List[str]:
#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         if column_type == 'numeric':
#             return self.data_frame.select_dtypes(include=[np.number]).columns.tolist()
#         elif column_type == 'categorical':
#             return self.data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
#         else:
#             return self.data_frame.columns.tolist()
    
#     def export_summary_report(
#         self, 
#         output_file: Optional[Path] = None,
#         include_plots: bool = True
#     ) -> Path:

#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         output_file = output_file or (self.output_dir / "summary_report.txt")
#         output_file = Path(output_file)
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"{'='*80}\n")
#             f.write(f"DATA SUMMARY REPORT\n")
#             f.write(f"{'='*80}\n\n")
            
#             f.write(f"Dataset: {self.dataset_path.name}\n")
#             f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
#             f.write(f"{'='*80}\n")
#             f.write(f"OVERVIEW\n")
#             f.write(f"{'='*80}\n")
#             f.write(f"Rows: {self.data_frame.shape[0]:,}\n")
#             f.write(f"Columns: {self.data_frame.shape[1]}\n\n")
            
#             f.write(f"{'='*80}\n")
#             f.write(f"NUMERIC SUMMARY\n")
#             f.write(f"{'='*80}\n")
#             f.write(self.data_frame.describe().to_string())
#             f.write("\n\n")
            
#             f.write(f"{'='*80}\n")
#             f.write(f"MISSING VALUES\n")
#             f.write(f"{'='*80}\n")
#             missing = self.data_frame.isnull().sum()
#             missing = missing[missing > 0].sort_values(ascending=False)
#             if len(missing) > 0:
#                 f.write(missing.to_string())
#             else:
#                 f.write("No missing values found.\n")
#             f.write("\n\n")
            
#             f.write(f"{'='*80}\n")
#             f.write(f"COLUMN TYPES\n")
#             f.write(f"{'='*80}\n")
#             f.write(self.data_frame.dtypes.to_string())
#             f.write("\n")
        
#         print(f"✓ Summary report saved: {output_file}")
        
#         if include_plots:
#             print("\nGenerating visualization plots...")
#             self.plot_all_distributions()
#             self.plot_all_value_counts()
        
#         return output_file
    
#     def data_loading(self, df: Optional[pd.DataFrame] = None, clean_columns: bool = True):

#         if df is not None:
#             self.set_dataframe(df, clean_columns=clean_columns)
#             return
#         if self.data_frame is None:
#             self.load_data(clean_columns=clean_columns)

#     def univariate_scatter(
#     self,
#     chart_name: str,
#     column: str,
#     *,
#     size: tuple = None,
#     point_size: int = 40,
#     alpha: float = 0.9,
#     jitter: float = 0.06,
#     orientation: str = "v",
#     label: str = None,
#     save_path=None,
#     save_formats=None,
#     font: dict = None,
#     **kwargs,
# ):

#         if self.data_frame is None:
#             raise ValueError("Data not loaded. Call load_data() first.")

#         size = size or self.config.DEFAULT_SIZE
#         font_config = self._apply_font_config(font)

#         df = self.data_frame.copy()
#         df[column] = pd.to_numeric(df[column], errors="coerce")
#         df = df.dropna(subset=[column])

#         vals = df[column].values
#         jitter_vals = np.random.normal(0, jitter, len(vals))

#         fig, ax = plt.subplots(figsize=size)
#         fig.patch.set_facecolor("white")
#         ax.set_facecolor("white")

#         if orientation.lower() == "v":
#             ax.scatter(vals, jitter_vals, s=point_size, alpha=alpha, color=self.config.LIGHT_BLUE, edgecolor="none")
#             xlabel, ylabel = column.replace("_", " ").title(), ""

#             ax.set_yticks([])                       
#             ax.set_ylabel("")                        
#             ax.spines["left"].set_visible(False)      
#             ax.yaxis.set_visible(False)               

#         else:
#             ax.scatter(jitter_vals, vals, s=point_size, alpha=alpha, color=self.config.LIGHT_BLUE, edgecolor="none")
#             xlabel, ylabel = "", column.replace("_", " ").title()

#         self._style_axes(ax, chart_name, xlabel, ylabel, font_config)
#         ax.grid(False)  
#         sns.despine(ax=ax, top=True, right=True)

#         plt.tight_layout()
#         self._save_figure(fig, label or chart_name, "univariate_scatter", save_path, save_formats or ["png"])
#         plt.close(fig)

#         return fig

# data_visualisation = DataVisualizer


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
import warnings
from matplotlib import colors as mcolors

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
        cmap = LinearSegmentedColormap.from_list(
            "alnayzak", [cls.LIGHT_BLUE, cls.DARK_BLUE], N=n
        )
        return [cmap(i / max(1, n - 1)) for i in range(n)]

    @classmethod
    def pie_palette(cls, labels, emphasize=None):
        n = len(labels)
        raw = sns.color_palette(cls.PIE_PALETTE_NAME, n + 3)[1:-0 or None]
        pal = [mcolors.to_hex(c) for c in raw]

        if emphasize is not None and emphasize in labels:
            i = list(labels).index(emphasize)
            pal[i] = cls.DARK_BLUE
        return pal

    @staticmethod
    def _text_color_for(bg_hex: str) -> str:
        r, g, b = mcolors.to_rgb(bg_hex)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance > 0.6 else "#ffffff"

    @classmethod
    def get_alnayzak_palette(cls, n: int) -> List[str]:

        base_colors = [
            cls.DARK_BLUE,
            cls.LIGHT_BLUE,
            "#1A3A6E",
            "#4CA6D8",
            "#86CAE8",
            "#FFFFFF"
        ]
        k = (n + len(base_colors) - 1) // len(base_colors)
        return (base_colors * k)[:n]

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
            self.output_dir = self.dataset_path.parent / self.dataset_base

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_frame: Optional[pd.DataFrame] = None
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
                self.data_frame.columns
                .str.strip()
                .str.replace("\n", " ", regex=True)
            )

        print(f"✓ Loaded {len(self.data_frame)} rows from {self.dataset_path.name}")

    def set_dataframe(self, df: pd.DataFrame, clean_columns: bool = True):
        self.data_frame = df.copy()

        if clean_columns:
            self.data_frame.columns = (
                self.data_frame.columns
                .str.strip()
                .str.replace("\n", " ", regex=True)
            )

    def get_data(self) -> pd.DataFrame:
        if self.data_frame is None:
            raise ValueError("No data loaded. Call load_data() or set_dataframe() first.")
        return self.data_frame.copy()

    def get_info(self) -> None:
        if self.data_frame is None:
            raise ValueError("No data loaded. Call load_data() or set_dataframe() first.")

        print(f"\n{'='*80}")
        print(f"Dataset: {self.dataset_path.name}")
        print(f"{'='*80}")
        print(f"Shape: {self.data_frame.shape[0]:,} rows × {self.data_frame.shape[1]} columns")
        print(f"Output directory: {self.output_dir}")
        print(f"\nColumn types:")
        print(self.data_frame.dtypes.value_counts())
        print(f"\nMemory usage: {self.data_frame.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*80}\n")

    def list_columns(self, column_type: Optional[str] = None) -> List[str]:
        if self.data_frame is None:
            raise ValueError("No data loaded. Call load_data() or set_dataframe() first.")

        if column_type == 'numeric':
            return self.data_frame.select_dtypes(include=[np.number]).columns.tolist()
        elif column_type == 'categorical':
            return self.data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            return self.data_frame.columns.tolist()

    def quick_summary(self, numeric_only: bool = True) -> pd.DataFrame:
        if self.data_frame is None:
            raise ValueError("No data loaded. Call load_data() or set_dataframe() first.")

        if numeric_only:
            return self.data_frame.describe()
        else:
            return self.data_frame.describe(include='all')


class BasePlotter:
    def __init__(self, data: DataManager, config: VisualizationConfig):
        self.data = data
        self.config = config

    # ---- helpers shared by all plotters ----
    def _slug(self, text: str) -> str:
        raw = str(text).strip()
        s = re.sub(r"\s+", "_", raw)
        s = re.sub(r"[^\w\-]", "", s)
        s = re.sub(r"_+", "_", s).lower() or "chart"
        return s

    def _auto_adjust_ylim(self, ax, padding: Optional[float] = None):
        plt.draw()
        y_min, y_max = ax.get_ylim()
        pad_ratio = self.data.ylim_padding if padding is None else float(padding)
        pad = (y_max - y_min) * max(0.0, pad_ratio)
        ax.set_ylim(y_min, y_max + pad)

    def _save_figure(
        self,
        fig,
        label: str,
        chart_type: str = "chart",
        save_path: Optional[Path] = None,
        formats: Optional[List[str]] = None,
    ) -> List[Path]:

        if formats is None:
            formats = ['png']

        saved_files: List[Path] = []

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.DPI_PRINT, bbox_inches="tight")
            saved_files.append(save_path)
        else:
            base_label = f"{chart_type}_{self._slug(label)}"
            base = f"{base_label}_{self.data.dataset_base}"

            for fmt in formats:
                dpi = self.config.DPI_SCREEN if fmt == 'jpg' else self.config.DPI_PRINT
                file_path = self.data.output_dir / f"{base}.{fmt}"
                fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
                saved_files.append(file_path)

        return saved_files

    def _apply_font_config(self, font_config: Optional[Dict] = None) -> Dict:
        config = self.config.get_font_config(font_config)
        plt.rcParams["font.family"] = config["family"]
        return config

    def _style_axes(
        self,
        ax,
        title: str,
        xlabel: str,
        ylabel: str,
        font_config: Dict,
        grid: bool = True,
    ):
        ax.set_title(
            title,
            fontsize=font_config["title_size"],
            fontweight="bold",
            pad=20,
            color=self.config.DARK_BLUE,
            loc="left",
        )
        ax.set_xlabel(
            xlabel,
            fontsize=font_config["label_size"],
            fontweight="600",
            color=self.config.DARK_BLUE,
        )
        ax.set_ylabel(
            ylabel,
            fontsize=font_config["label_size"],
            fontweight="600",
            color=self.config.DARK_BLUE,
        )
        ax.tick_params(axis="both", labelsize=font_config["tick_size"])

        sns.despine(ax=ax, top=True, right=True)

        if grid:
            ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=1)
            ax.set_axisbelow(True)

        ax.set_facecolor("white")

    def _annotate_bars(
        self,
        ax,
        annotate_format: str = "{:,.0f}",
        font_size: Optional[int] = None,
    ):
        if font_size is None:
            font_size = self.config.VALUE_LABEL_SIZE

        for patch in ax.patches:
            height = patch.get_height()
            if np.isnan(height) or height <= 0:
                continue

            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                annotate_format.format(height),
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold",
                color=self.config.DARK_BLUE,
            )


class BarChartPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        column: str,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ylim_padding: Optional[float] = None,
        label: Optional[str] = None,
        colors: Optional[List] = None,
        size: Tuple[int, int] = None,
        font: Optional[Dict] = None,
        order: Optional[List] = None,
        sort_by: str = "x",
        ascending: bool = True,
        annotate: bool = True,
        annotate_format: str = "{:,.0f}",
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        counts = (
            df[column]
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )

        if sort_by == "y":
            counts = counts.sort_values("count", ascending=ascending)
        else:
            counts = counts.sort_values(column, ascending=ascending)

        order = order or counts[column].tolist()
        palette = colors or self.config.get_palette(len(order))

        sns.barplot(
            data=counts,
            x=column,
            y="count",
            order=order,
            ci=None,
            ax=ax,
            palette=palette,
        )

        self._auto_adjust_ylim(ax, padding=ylim_padding)

        resolved_xlabel = (
            xlabel or self.data.default_xlabel or column.replace("_", " ").title()
        )
        resolved_ylabel = ylabel or self.data.default_ylabel

        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)

        if annotate:
            self._annotate_bars(ax, annotate_format, font_config["value_label_size"])

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "bar_chart",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(5)
        plt.close(fig)

        return fig, saved


class HistogramPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        column: str,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ylim_padding: Optional[float] = None,
        bins: int = 20,
        size: Tuple[int, int] = None,
        font: Optional[Dict] = None,
        kde: bool = True,
        color: Optional[str] = None,
        alpha: float = 0.85,
        annotate: bool = True,
        annotate_format: str = "{:,.0f}",
        show_mean_line: bool = True,
        show_stats_box: bool = True,
        label: Optional[str] = None,
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])

        if len(df) == 0:
            raise ValueError(f"No valid numeric data in column '{column}'")

        mean_val = float(df[column].mean())
        median_val = float(df[column].median())

        color = color or self.config.LIGHT_BLUE
        sns.histplot(
            data=df,
            x=column,
            bins=bins,
            kde=kde,
            color=color,
            alpha=alpha,
            ax=ax,
            edgecolor="white",
            linewidth=1.5,
        )

        if kde and hasattr(ax, "lines") and len(ax.lines) > 0:
            ax.lines[0].set_color(self.config.LIGHT_BLUE)
            ax.lines[0].set_linewidth(3)

        n_patches = len(ax.patches)
        if n_patches > 0:
            palette = self.config.get_palette(n_patches)
            for i, patch in enumerate(ax.patches):
                patch.set_facecolor(palette[i])

        self._auto_adjust_ylim(ax, padding=ylim_padding)

        resolved_xlabel = (
            xlabel or self.data.default_xlabel or column.replace("_", " ").title()
        )
        resolved_ylabel = ylabel or self.data.default_ylabel or "Frequency"

        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)
        ax.set_facecolor("#fafafa")

        if annotate:
            self._annotate_bars(ax, annotate_format, font_config["value_label_size"])

        if show_mean_line:
            ax.axvline(
                mean_val,
                color=self.config.LIGHT_BLUE,
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean_val:,.2f}",
            )

        if show_stats_box:
            stats_text = f"Mean: {mean_val:,.2f}\nMedian: {median_val:,.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_config["label_size"] * 0.9,
                color=self.config.DARK_BLUE,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor=self.config.DARK_BLUE,
                    alpha=0.9,
                ),
            )

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "histogram",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(5)
        plt.close(fig)

        return fig, saved


class PieChartPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        column: str,
        *,
        size: Tuple[int, int] = (8, 6),
        font: Optional[Dict] = None,
        annotate: bool = False,
        annotate_format: str = "{:.1f}%",
        explode_top_n: int = 0,
        explode_value: float = 0.001,
        label: Optional[str] = None,
        show_legend: bool = True,
        legend_loc: str = "center left",
        legend_bbox_to_anchor: Tuple = (0.95, 0.5),
        colors: Optional[List] = None,
        explode: Optional[List] = None,
        startangle: float = 90,
        pctdistance: float = 0.8,
        max_slices: int = 20,
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        df = df.copy()
        s = df[column]

        if pd.api.types.is_categorical_dtype(s):
            cats = list(s.cat.categories)
            extras = [v for v in s.dropna().unique() if v not in cats]
            need = (["Missing"] if "Missing" not in cats else []) + [
                x for x in extras if x is not None
            ]
            if need:
                s = s.cat.add_categories(need)

        counts = (
            s.fillna("Missing")
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )

        if len(counts) > max_slices:
            counts = counts.sort_values("count", ascending=False)
            top = counts.iloc[: max_slices - 1]
            other = pd.DataFrame(
                {column: ["Other"], "count": [counts.iloc[max_slices - 1 :]["count"].sum()]}
            )
            counts = pd.concat([top, other], ignore_index=True)

        label_map = {1: "Passed", 0: "Failed"}
        counts[column] = counts[column].replace(label_map).astype(str)

        order = None
        if {"Passed", "Failed"}.issubset(set(counts[column])):
            order = ["Passed", "Failed"]
        if order:
            counts = counts.set_index(column).reindex(order).dropna().reset_index()

        total = int(counts["count"].sum())
        if total == 0:
            raise ValueError(f"No data to plot for column '{column}'")

        labels = counts[column].tolist()
        emphasize_label = "Passed" if "Passed" in labels else None
        palette = colors or self.config.pie_palette(labels, emphasize=emphasize_label)

        if explode is None:
            explode = [
                explode_value if i < explode_top_n else 0.0
                for i in range(len(labels))
            ]

        pie_kwargs = dict(
            x=counts["count"].values,
            colors=palette,
            explode=explode,
            startangle=startangle,
            textprops={
                "fontsize": font_config["tick_size"],
                "color": self.config.DARK_BLUE,
            },
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        if annotate:
            pie_kwargs["autopct"] = annotate_format
            pie_kwargs["pctdistance"] = pctdistance

        res = ax.pie(**pie_kwargs)
        if annotate:
            wedges, texts, autotexts = res
            for w, t in zip(wedges, autotexts):
                face = mcolors.to_hex(w.get_facecolor())
                t.set_color(self.config._text_color_for(face))
                t.set_fontweight("bold")
        else:
            wedges, texts = res

        ax.axis("equal")

        ax.set_title(
            chart_name,
            fontsize=font_config["title_size"],
            fontweight="bold",
            color=self.config.DARK_BLUE,
            loc="center",
            pad=10,
        )

        if show_legend:
            pct = (counts["count"] / total * 100).round(1)
            legend_labels = [f"{lbl} ({p:.1f}%)" for lbl, p in zip(labels, pct)]
            n_items = len(legend_labels)
            ncol = 1 if n_items <= 10 else (2 if n_items <= 18 else 3)
            leg = ax.legend(
                handles=wedges,
                labels=legend_labels,
                title="Categories",
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                fontsize=font_config["tick_size"],
                frameon=True,
                ncol=ncol,
                columnspacing=1.2,
                handlelength=1.2,
                labelspacing=0.6,
                borderpad=0.8,
            )
            plt.setp(leg.get_title(), fontsize=font_config["tick_size"], fontweight="bold")

        fig.subplots_adjust(left=0.02, right=0.75, top=0.88, bottom=0.08)

        saved = self._save_figure(
            fig,
            label or chart_name,
            "pie_chart",
            save_path,
            save_formats or ["png"],
        )
        plt.close(fig)
        return fig, saved


class GroupedBarChartPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        group_col: str,
        score_cols: List[str],
        *,
        agg_func: str = "mean",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple] = None,
        ylim_padding: Optional[float] = None,
        label: Optional[str] = None,
        colors: Optional[List] = None,
        size: Tuple[int, int] = None,
        font: Optional[Dict] = None,
        ci: Optional[int] = None,
        dodge: bool = True,
        annotate: bool = True,
        annotate_format: str = "{:,.1f}",
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()

        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")

        missing = [c for c in score_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Score columns not found: {missing}")

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        df_long = df.melt(
            id_vars=[group_col],
            value_vars=score_cols,
            var_name="Subject",
            value_name="Score",
        ).dropna(subset=["Score"])

        grouped = df_long.groupby([group_col, "Subject"], as_index=False).agg(
            value=("Score", agg_func)
        )

        x_order = grouped[group_col].dropna().unique().tolist()
        hue_order = grouped["Subject"].dropna().unique().tolist()

        palette = colors or self.config.get_palette(len(hue_order))

        sns.barplot(
            data=grouped,
            x=group_col,
            y="value",
            hue="Subject",
            order=x_order,
            hue_order=hue_order,
            ci=ci,
            dodge=dodge,
            ax=ax,
            palette=palette,
        )

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            max_val = grouped["value"].max()
            pad_ratio = self.data.ylim_padding if ylim_padding is None else float(
                ylim_padding
            )
            padding = max_val * pad_ratio
            ax.set_ylim(0, max_val + padding)

        display_name = {
            "mean": "Average",
            "median": "Median",
            "sum": "Total",
            "count": "Count",
        }.get(agg_func, agg_func.title())

        resolved_xlabel = xlabel or group_col.replace("_", " ").title()
        resolved_ylabel = ylabel or display_name

        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)

        ax.legend(title=None, frameon=True, loc="upper right")

        if annotate:
            self._annotate_bars(ax, annotate_format, font_config["value_label_size"])

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "grouped_bar_chart",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(3)
        plt.close(fig)

        return fig, saved


class BoxPlotPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        column: str,
        group_by: Optional[str] = None,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        size: Tuple[int, int] = None,
        font: Optional[Dict] = None,
        colors: Optional[List] = None,
        label: Optional[str] = None,
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])

        if group_by:
            palette = colors or self.config.get_palette(df[group_by].nunique())
            sns.boxplot(
                data=df,
                x=group_by,
                y=column,
                palette=palette,
                ax=ax,
            )
            resolved_xlabel = xlabel or group_by.replace("_", " ").title()
        else:
            color = colors[0] if colors else self.config.LIGHT_BLUE
            sns.boxplot(
                data=df,
                y=column,
                color=color,
                ax=ax,
            )
            resolved_xlabel = xlabel or ""

        resolved_ylabel = ylabel or column.replace("_", " ").title()

        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "box_plot",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(3)
        plt.close(fig)

        return fig, saved


class ScatterPlotPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        x_column: str,
        y_column: str,
        *,
        hue: str = None,
        size_column: str = None,
        xlabel: str = None,
        ylabel: str = None,
        size: tuple = None,
        point_size: int = 40,
        alpha: float = 1.0,
        minimal: bool = True,
        show_regression: bool = False,
        label: str = None,
        save_path=None,
        save_formats=None,
        **kwargs,
    ):
        df = self.data.get_data()

        df = df.copy()
        df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
        df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
        df = df.dropna(subset=[x_column, y_column])

        size = size or (8, 6)
        fig, ax = plt.subplots(figsize=size)

        if minimal:
            ax.grid(False)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            if size_column and size_column in df.columns:
                s_vals = pd.to_numeric(df[size_column], errors="coerce").fillna(0)
                ax.scatter(df[x_column], df[y_column], s=s_vals, alpha=alpha)
            else:
                ax.scatter(df[x_column], df[y_column], s=point_size, alpha=alpha)
            ax.set_title(chart_name, pad=10)
            ax.set_xlabel(xlabel or x_column.replace("_", " ").title())
            ax.set_ylabel(ylabel or y_column.replace("_", " ").title())

            show_regression = False
            hue = None

        else:
            palette = self.config.get_palette(1)
            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                color=palette[0],
                s=point_size,
                alpha=alpha,
                ax=ax,
            )
            if show_regression:
                sns.regplot(data=df, x=x_column, y=y_column, scatter=False, ax=ax)

            self._style_axes(
                ax,
                chart_name,
                xlabel or x_column.replace("_", " ").title(),
                ylabel or y_column.replace("_", " ").title(),
                self._apply_font_config(None),
            )

        plt.tight_layout()
        saved = self._save_figure(
            fig,
            label or chart_name,
            "scatter_plot",
            save_path,
            save_formats or ["png"],
        )
        plt.close(fig)
        return fig, saved


class LineChartPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        x_column: str,
        y_columns: Union[str, List[str]],
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        size: Tuple[int, int] = None,
        font: Optional[Dict] = None,
        colors: Optional[List] = None,
        markers: bool = True,
        linewidth: float = 2.5,
        label: Optional[str] = None,
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        df = df.copy()

        if isinstance(y_columns, str):
            y_columns = [y_columns]

        palette = colors or self.config.get_palette(len(y_columns))

        for i, y_col in enumerate(y_columns):
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            data = df[[x_column, y_col]].dropna()

            plot_args = {
                "x": x_column,
                "y": y_col,
                "data": data,
                "color": palette[i],
                "linewidth": linewidth,
                "label": y_col.replace("_", " ").title(),
                "ax": ax,
            }

            if markers:
                plot_args["marker"] = "o"
                plot_args["markersize"] = 6

            sns.lineplot(**plot_args)

        resolved_xlabel = xlabel or x_column.replace("_", " ").title()
        resolved_ylabel = ylabel or "Value"

        self._style_axes(ax, chart_name, resolved_xlabel, resolved_ylabel, font_config)

        if len(y_columns) > 1:
            ax.legend(frameon=True, loc="best")

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "line_chart",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(3)
        plt.close(fig)

        return fig, saved


class HeatmapPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str = "Correlation Heatmap",
        columns: Optional[List[str]] = None,
        size: Tuple[int, int] = (12, 10),
        font: Optional[Dict] = None,
        cmap: str = "coolwarm",
        annot: bool = True,
        fmt: str = ".2f",
        label: Optional[str] = None,
        save_path: Optional[Path] = None,
        save_formats: List[str] = None,
    ) -> Tuple[Any, List[Path]]:

        df = self.data.get_data()
        font_config = self._apply_font_config(font)

        df = df.copy()
        if columns is None:
            df_numeric = df.select_dtypes(include=[np.number])
        else:
            df_numeric = df[columns].select_dtypes(include=[np.number])

        if df_numeric.empty:
            raise ValueError("No numeric columns found for correlation")

        corr = df_numeric.corr()

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")

        sns.heatmap(
            corr,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(
            chart_name,
            fontsize=font_config["title_size"],
            fontweight="bold",
            color=self.config.DARK_BLUE,
            pad=20,
        )

        plt.tight_layout()

        saved = self._save_figure(
            fig,
            label or chart_name,
            "heatmap",
            save_path,
            save_formats,
        )

        plt.show()
        plt.pause(3)
        plt.close(fig)

        return fig, saved


class UnivariateScatterPlotter(BasePlotter):
    def plot(
        self,
        chart_name: str,
        column: str,
        *,
        size: tuple = None,
        point_size: int = 40,
        alpha: float = 0.9,
        jitter: float = 0.06,
        orientation: str = "v",
        label: str = None,
        save_path=None,
        save_formats=None,
        font: dict = None,
        **kwargs,
    ):

        df = self.data.get_data()

        size = size or self.config.DEFAULT_SIZE
        font_config = self._apply_font_config(font)

        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])

        vals = df[column].values
        jitter_vals = np.random.normal(0, jitter, len(vals))

        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if orientation.lower() == "v":
            ax.scatter(
                vals,
                jitter_vals,
                s=point_size,
                alpha=alpha,
                color=self.config.LIGHT_BLUE,
                edgecolor="none",
            )
            xlabel, ylabel = column.replace("_", " ").title(), ""

            ax.set_yticks([])
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)

        else:
            ax.scatter(
                jitter_vals,
                vals,
                s=point_size,
                alpha=alpha,
                color=self.config.LIGHT_BLUE,
                edgecolor="none",
            )
            xlabel, ylabel = "", column.replace("_", " ").title()

        self._style_axes(ax, chart_name, xlabel, ylabel, font_config)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)

        plt.tight_layout()
        self._save_figure(
            fig,
            label or chart_name,
            "univariate_scatter",
            save_path,
            save_formats or ["png"],
        )
        plt.close(fig)

        return fig, []


class DataVisualizer:
    """
    Facade/manager class that holds the data and exposes plotter instances
    while preserving the original method names for backward compatibility.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        default_xlabel: Optional[str] = None,
        default_ylabel: Optional[str] = "Count",
        ylim_padding: float = 0.1,
        df: Optional[pd.DataFrame] = None,
    ):

        # visual style
        sns.set_theme(context="talk", style="whitegrid")
        self.config = VisualizationConfig()

        # data manager
        self.data = DataManager(
            dataset_path=dataset_path,
            output_dir=output_dir,
            default_xlabel=default_xlabel,
            default_ylabel=default_ylabel,
            ylim_padding=ylim_padding,
            df=df,
        )

        # plotter instances
        self.bar = BarChartPlotter(self.data, self.config)
        self.hist = HistogramPlotter(self.data, self.config)
        self.pie = PieChartPlotter(self.data, self.config)
        self.grouped_bar = GroupedBarChartPlotter(self.data, self.config)
        self.box = BoxPlotPlotter(self.data, self.config)
        self.scatter = ScatterPlotPlotter(self.data, self.config)
        self.line = LineChartPlotter(self.data, self.config)
        self.heatmap = HeatmapPlotter(self.data, self.config)
        self.uni_scatter = UnivariateScatterPlotter(self.data, self.config)

    # ----------------- data helpers / compatibility -----------------
    def load_data(self, clean_columns: bool = True):
        self.data.load_data(clean_columns=clean_columns)

    def set_dataframe(self, df: pd.DataFrame, clean_columns: bool = True):
        self.data.set_dataframe(df, clean_columns=clean_columns)

    def get_data(self) -> pd.DataFrame:
        return self.data.get_data()

    def get_info(self) -> None:
        self.data.get_info()

    def list_columns(self, column_type: Optional[str] = None) -> List[str]:
        return self.data.list_columns(column_type=column_type)

    def quick_summary(self, numeric_only: bool = True) -> pd.DataFrame:
        return self.data.quick_summary(numeric_only=numeric_only)

    # ----------------- per-plot convenience wrappers -----------------
    def bar_chart(self, *args, **kwargs):
        return self.bar.plot(*args, **kwargs)

    def histogram(self, *args, **kwargs):
        return self.hist.plot(*args, **kwargs)

    def pie_chart(self, *args, **kwargs):
        return self.pie.plot(*args, **kwargs)

    def grouped_bar_chart(self, *args, **kwargs):
        return self.grouped_bar.plot(*args, **kwargs)

    def multiple_bar_charts(self, *args, **kwargs):
        """Alias for grouped_bar_chart (backward compatibility)."""
        return self.grouped_bar_chart(*args, **kwargs)

    def box_plot(self, *args, **kwargs):
        return self.box.plot(*args, **kwargs)

    def scatter_plot(self, *args, **kwargs):
        return self.scatter.plot(*args, **kwargs)

    def line_chart(self, *args, **kwargs):
        return self.line.plot(*args, **kwargs)

    def correlation_heatmap(self, *args, **kwargs):
        return self.heatmap.plot(*args, **kwargs)

    def univariate_scatter(self, *args, **kwargs):
        return self.uni_scatter.plot(*args, **kwargs)

    # ----------------- higher-level helpers -----------------
    def plot_all_distributions(
        self,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[Any, List[Path]]]:

        df = self.get_data()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []
        for col in columns:
            try:
                fig, saved = self.hist.plot(
                    chart_name=f"Distribution of {col}",
                    column=col,
                    **kwargs,
                )
                results.append((fig, saved))
                print(f"✓ Created histogram for {col}")
            except Exception as e:
                print(f"⚠ Warning: Could not plot {col}: {e}")

        return results

    def plot_all_value_counts(
        self,
        columns: Optional[List[str]] = None,
        max_categories: int = 20,
        **kwargs,
    ) -> List[Tuple[Any, List[Path]]]:

        df = self.get_data()

        if columns is None:
            columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

        results = []
        for col in columns:
            try:
                n_unique = df[col].nunique()
                if n_unique > max_categories:
                    print(f"⚠ Skipping {col}: too many categories ({n_unique})")
                    continue

                fig, saved = self.bar.plot(
                    chart_name=f"Distribution of {col}",
                    column=col,
                    **kwargs,
                )
                results.append((fig, saved))
                print(f"✓ Created bar chart for {col}")
            except Exception as e:
                print(f"⚠ Warning: Could not plot {col}: {e}")

        return results

    def export_summary_report(
        self,
        output_file: Optional[Path] = None,
        include_plots: bool = True,
    ) -> Path:

        df = self.get_data()

        output_file = output_file or (self.data.output_dir / "summary_report.txt")
        output_file = Path(output_file)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"DATA SUMMARY REPORT\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Dataset: {self.data.dataset_path.name}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"OVERVIEW\n")
            f.write(f"{'='*80}\n")
            f.write(f"Rows: {df.shape[0]:,}\n")
            f.write(f"Columns: {df.shape[1]}\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"NUMERIC SUMMARY\n")
            f.write(f"{'='*80}\n")
            f.write(df.describe().to_string())
            f.write("\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"MISSING VALUES\n")
            f.write(f"{'='*80}\n")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                f.write(missing.to_string())
            else:
                f.write("No missing values found.\n")
            f.write("\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"COLUMN TYPES\n")
            f.write(f"{'='*80}\n")
            f.write(df.dtypes.to_string())
            f.write("\n")

        print(f"✓ Summary report saved: {output_file}")

        if include_plots:
            print("\nGenerating visualization plots...")
            self.plot_all_distributions()
            self.plot_all_value_counts()

        return output_file

    def data_loading(self, df: Optional[pd.DataFrame] = None, clean_columns: bool = True):
        """
        Backward-compatible helper: if df is passed, use it; otherwise read from disk.
        """
        if df is not None:
            self.set_dataframe(df, clean_columns=clean_columns)
        else:
            self.load_data(clean_columns=clean_columns)


# convenient alias to keep your last line intact
data_visualisation = DataVisualizer
