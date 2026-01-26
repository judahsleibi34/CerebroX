import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import numpy as np
import pandas as pd


from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QFrame, QComboBox, QScrollArea, QStyledItemDelegate, QPushButton
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from config import PALETTE

from EDA import (
    DataVisualizer,
    ColumnResolver,
    get_col_groups,
    choose_xy,
    choose_group,
)

from EDA import best_numeric_cols, best_categorical_cols

class PlottingThread(QThread):
    finished = Signal(str, bool, str)  # plot_type, success, message

    def __init__(self, plot_type, df, output_dir):
        super().__init__()
        self.plot_type = plot_type
        self.df = df.copy(deep=True)
        self.output_dir = output_dir

    # -------------------------
    # TYPE-BASED HELPERS
    # -------------------------
    def _clean_columns(self):
        self.df.columns = (
            self.df.columns.astype(str)
            .str.strip()
            .str.replace("\n", " ", regex=False)
        )

    def _is_id_like(self, s: pd.Series) -> bool:
        s = s.dropna()
        if s.empty:
            return False
        return (s.nunique() / len(s)) > 0.95  # high uniqueness => likely id

    def _best_numeric_cols(self, k=3):
        nums = self.df.select_dtypes(include=["number"])
        if nums.shape[1] == 0:
            return []

        stats = []
        for c in nums.columns:
            s = nums[c].dropna()
            if len(s) < 10:
                continue
            if self._is_id_like(s):
                continue
            stats.append((c, float(s.notna().mean()), float(s.var())))
        stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [c for c, _, _ in stats[:k]]

    def _best_score_cols(self, k=3):
        numeric = self.df.select_dtypes(include=["number"]).columns.tolist()
        scored = []
        for c in numeric:
            name = c.lower()
            if any(key in name for key in ["score", "mark", "result", "total", "math", "arabic", "english"]):
                if not self._is_id_like(self.df[c]):
                    scored.append(c)
        if scored:
            return scored[:k]
        return self._best_numeric_cols(k=k)

    def _categorical_like_cols(self, max_unique=25):
        good = []

        # object/category columns
        for c in self.df.select_dtypes(exclude=["number"]).columns:
            nunique = self.df[c].nunique(dropna=True)
            if 2 <= nunique <= max_unique:
                good.append((c, nunique))

        # numeric but low-cardinality (grade/class sometimes int)
        for c in self.df.select_dtypes(include=["number"]).columns:
            nunique = self.df[c].nunique(dropna=True)
            if 2 <= nunique <= max_unique and not self._is_id_like(self.df[c]):
                good.append((c, nunique))

        good.sort(key=lambda x: x[1])
        return [c for c, _ in good]

    def _pick_group_col(self, prefer_keywords=None, max_unique=12):
        prefer_keywords = [k.lower() for k in (prefer_keywords or [])]
        cols = self._categorical_like_cols(max_unique=max_unique)
        if not cols:
            return None

        if prefer_keywords:
            for c in cols:
                name = c.lower()
                if any(k in name for k in prefer_keywords):
                    return c

        return cols[0]

    def _auto_success_threshold(self, s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return 0
        mx = float(s.max())
        if mx <= 1.5:
            return 0.5
        if mx <= 20.5:
            return 10.0
        return float(s.median())

    def _plot_success_rates(self, score_cols, group_col=None, chart_name="Success Rates Analysis"):
        import matplotlib.pyplot as plt

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Overall success per score
        labels = []
        rates = []
        for col in score_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            thr = self._auto_success_threshold(s)
            valid = s.dropna()
            if valid.empty:
                continue
            rate = 100.0 * float((valid >= thr).mean())
            labels.append(col)
            rates.append(rate)

        if labels:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.barh(labels, rates)
            ax.set_title(f"{chart_name} (Overall)")
            ax.set_xlabel("Success Rate (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate(rates):
                ax.text(min(v + 1, 99), i, f"{v:.1f}%", va="center")
            fig.tight_layout()
            fig.savefig(out / "success_rates_overall.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # By group for first score
        if group_col and score_cols:
            col = score_cols[0]
            s = pd.to_numeric(self.df[col], errors="coerce")
            thr = self._auto_success_threshold(s)

            tmp = self.df[[group_col, col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna(subset=[group_col, col])

            if not tmp.empty:
                grp = tmp.groupby(group_col)[col].apply(lambda x: 100.0 * float((x >= thr).mean())).sort_values()
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.barh(grp.index.astype(str), grp.values)
                ax.set_title(f"{chart_name} by {group_col} ({col})")
                ax.set_xlabel("Success Rate (%)")
                ax.set_xlim(0, 100)
                for i, v in enumerate(grp.values):
                    ax.text(min(v + 1, 99), i, f"{v:.1f}%", va="center")
                fig.tight_layout()
                fig.savefig(out / "success_rates_by_group.png", dpi=300, bbox_inches="tight", facecolor="white")
                plt.close(fig)

    # -------------------------
    # MAIN THREAD RUN
    # -------------------------
    def run(self):
        try:
            self._clean_columns()

            # new visualizer per thread
            visualizer = DataVisualizer(dataset_path="", output_dir=self.output_dir, df=self.df)

            if self.plot_type == "Histogram":
                visualizer.plot_all_distributions()
                self.finished.emit(self.plot_type, True, "Histograms generated successfully")

            elif self.plot_type == "Scatter Plot":
                numeric_cols = self._best_numeric_cols(k=2)
                if len(numeric_cols) >= 2:
                    visualizer.scatter_plot("Scatter Plot", numeric_cols[0], numeric_cols[1])
                    self.finished.emit(self.plot_type, True, "Scatter plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns")

            elif self.plot_type == "Correlation Heatmap":
                numeric_cols = self._best_numeric_cols(k=30)
                if len(numeric_cols) >= 2:
                    visualizer.correlation_heatmap(columns=numeric_cols)
                    self.finished.emit(self.plot_type, True, "Correlation heatmap generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns for correlation")

            elif self.plot_type == "Bar Chart":
                visualizer.plot_all_value_counts(max_categories=15)
                self.finished.emit(self.plot_type, True, "Bar charts generated successfully")

            elif self.plot_type == "Line Plot":
                numeric_cols = self._best_numeric_cols(k=4)
                if len(numeric_cols) >= 2:
                    x = numeric_cols[0]
                    ys = numeric_cols[1:][:3]
                    visualizer.line_chart("Line Chart", x, ys)
                    self.finished.emit(self.plot_type, True, "Line plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns")

            elif self.plot_type == "Overlapping Histogram":
                group_col = self._pick_group_col(max_unique=8)
                num_cols = self._best_numeric_cols(k=1)
                if group_col and num_cols:
                    value_col = num_cols[0]
                    visualizer.overlapping_histogram(f"Distribution of {value_col} by {group_col}", value_col, group_col)
                    self.finished.emit(self.plot_type, True, "Overlapping histogram generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need (1 categorical-like) + (1 numeric)")

            elif self.plot_type == "Pie Chart":
                group_col = self._pick_group_col(max_unique=12)
                if group_col:
                    visualizer.pie_chart(f"Pie Chart of {group_col}", group_col, annotate=True, annotate_format="{:.1f}%%")
                    self.finished.emit(self.plot_type, True, "Pie chart generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No suitable categorical-like column found for pie chart")

            elif self.plot_type == "Ridgeline Plot":
                group_col = self._pick_group_col(max_unique=10)
                num_cols = self._best_numeric_cols(k=1)
                if group_col and num_cols:
                    visualizer.ridge_plot(f"Ridgeline Plot of {num_cols[0]} by {group_col}", num_cols[0], group_col)
                    self.finished.emit(self.plot_type, True, "Ridgeline plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need (1 categorical-like) + (1 numeric)")

            elif self.plot_type == "Lollipop Chart":
                group_col = self._pick_group_col(max_unique=20)
                if group_col:
                    visualizer.lollipop_chart(f"Lollipop Chart of {group_col}", group_col)
                    self.finished.emit(self.plot_type, True, "Lollipop chart generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No suitable categorical-like column found")

            elif self.plot_type == "Diverging Bar Chart":
                numeric_cols = self._best_numeric_cols(k=2)
                if len(numeric_cols) >= 2:
                    cat_col = self._pick_group_col(max_unique=30)
                    if not cat_col:
                        cat_col = "_auto_category"
                        self.df[cat_col] = np.arange(len(self.df)).astype(str)
                        visualizer.set_dataframe(self.df)
                    visualizer.diverging_bar_chart("Diverging Bar Chart", numeric_cols[0], numeric_cols[1], cat_col)
                    self.finished.emit(self.plot_type, True, "Diverging bar chart generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns")

            elif self.plot_type == "Stacked Area Chart":
                numeric_cols = self._best_numeric_cols(k=4)
                if len(numeric_cols) >= 3:
                    x_col = "_auto_x"
                    self.df = self.df.copy()
                    self.df[x_col] = np.arange(len(self.df))
                    visualizer.set_dataframe(self.df)
                    y_cols = numeric_cols[:3]
                    visualizer.stacked_area_chart("Stacked Area Chart", x_col, y_cols)
                    self.finished.emit(self.plot_type, True, "Stacked area chart generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 3 numeric columns")

            elif self.plot_type == "KPI Card":
                num_cols = self._best_numeric_cols(k=1)
                if num_cols:
                    col = num_cols[0]
                    vals = self.df[col].dropna().head(4).tolist()
                    if vals:
                        labels = [f"{col} - {i+1}" for i in range(len(vals))]
                        visualizer.kpi_card(f"KPI Cards for {col}", values=vals, labels=labels)
                        self.finished.emit(self.plot_type, True, "KPI cards generated successfully")
                    else:
                        self.finished.emit(self.plot_type, False, "No numeric values available for KPI cards")
                else:
                    self.finished.emit(self.plot_type, False, "No numeric columns found")

            elif self.plot_type == "Small Multiples":
                group_col = self._pick_group_col(max_unique=9)
                num_cols = self._best_numeric_cols(k=1)
                if group_col and num_cols:
                    visualizer.small_multiples(
                        f"Small Multiples of {num_cols[0]} by {group_col}",
                        num_cols[0], num_cols[0], group_col,
                        chart_type="hist"
                    )
                    self.finished.emit(self.plot_type, True, "Small multiples generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need (1 categorical-like) + (1 numeric)")

            elif self.plot_type == "Highlight Table":
                cols = list(self.df.columns)[:3]
                if len(cols) >= 2:
                    visualizer.highlight_table("Highlight Table", columns=cols)
                    self.finished.emit(self.plot_type, True, "Highlight table generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Not enough columns for highlight table")

            # ---------- TYPE-BASED “TLS/Geo” ----------
            elif self.plot_type == "TLS Distribution":
                tls_col = self._pick_group_col(prefer_keywords=["tls", "location", "site"], max_unique=50)
                if tls_col:
                    # If you have comprehensive_tls_distribution in EDA, use it:
                    if hasattr(visualizer, "comprehensive_tls_distribution"):
                        visualizer.comprehensive_tls_distribution(tls_column=tls_col)
                    else:
                        visualizer.bar_chart(f"TLS Distribution ({tls_col})", tls_col, sort_by="y", ascending=False)
                    self.finished.emit(self.plot_type, True, f"TLS distribution generated using '{tls_col}'")
                else:
                    self.finished.emit(self.plot_type, False, "No categorical-like TLS/location column found")

            elif self.plot_type == "Geographic Distribution":
                tls_col = self._pick_group_col(prefer_keywords=["tls", "location", "site"], max_unique=50)
                score_cols = self._best_score_cols(k=3)
                if tls_col and score_cols and hasattr(visualizer, "comprehensive_geographic_dist"):
                    visualizer.comprehensive_geographic_dist(tls_column=tls_col, score_columns=score_cols)
                    self.finished.emit(self.plot_type, True, f"Geographic distribution generated using '{tls_col}'")
                elif tls_col:
                    # fallback: just counts by location
                    visualizer.bar_chart(f"Geographic Distribution ({tls_col})", tls_col, sort_by="y", ascending=False)
                    self.finished.emit(self.plot_type, True, f"Geographic distribution (fallback counts) using '{tls_col}'")
                else:
                    self.finished.emit(self.plot_type, False, "No categorical-like location/TLS column found")

            # ---------- TYPE-BASED “Grade/Success/Score Dist” ----------
            elif self.plot_type == "Student Demographics":
                group_col = self._pick_group_col(max_unique=25)
                if group_col:
                    visualizer.bar_chart(f"Distribution of {group_col}", group_col, sort_by="y", ascending=False)
                    self.finished.emit(self.plot_type, True, f"Demographics generated using '{group_col}'")
                else:
                    self.finished.emit(self.plot_type, False, "No categorical-like column detected")

            elif self.plot_type == "Success Rates Analysis":
                score_cols = self._best_score_cols(k=3)
                group_col = self._pick_group_col(max_unique=12)
                if score_cols:
                    self._plot_success_rates(score_cols, group_col=group_col, chart_name="Success Rates Analysis")
                    self.finished.emit(self.plot_type, True, "Success rates generated (type-based)")
                else:
                    self.finished.emit(self.plot_type, False, "No suitable numeric score columns found")

            elif self.plot_type == "Score Distributions by Grade":
                group_col = self._pick_group_col(max_unique=12)
                score_cols = self._best_score_cols(k=1)
                if group_col and score_cols:
                    value_col = score_cols[0]
                    visualizer.overlapping_histogram(f"{value_col} distribution by {group_col}", value_col, group_col)
                    self.finished.emit(self.plot_type, True, f"Score distributions generated by '{group_col}'")
                else:
                    self.finished.emit(self.plot_type, False, "Need (1 categorical-like grouping) + (1 numeric score column)")

            else:
                self.finished.emit(self.plot_type, False, f"{self.plot_type} not implemented")

        except Exception as e:
            self.finished.emit(self.plot_type, False, f"Error: {str(e)}")

class CenterDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Plotting_Page(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Exploratory Data Analysis")
        self.setStyleSheet(f"background: {PALETTE['bg']};")

        # Data-related attributes
        self.merged_df = None
        self.sheets_dfs = {}
        self.current_sheet = None
        self.current_df = None
        self.processed_path: Optional[str] = None  # path to processed CSV

        # Plotting / EDA
        self.visualizer = None
        self.plot_checkboxes = {}
        self.active_threads = {}
        self.output_dir: Optional[Path] = None

        self.completed_plots = []
        self.failed_plots = []

        # ---------- MAIN LAYOUT ----------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        # Title
        main_title = QLabel("Exploratory Data Analysis")
        main_title.setAlignment(Qt.AlignLeft)
        main_title.setStyleSheet(f"color: {PALETTE['text']};")
        main_title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        layout.addWidget(main_title)

        # ---------- SHEET SELECTION ----------
        sheet_layout = QHBoxLayout()
        sheet_label = QLabel("Select Sheet:")
        sheet_label.setStyleSheet(f"color: {PALETTE['text']}; font-size: 14px;")
        sheet_layout.addWidget(sheet_label)

        self.sheet_combo = QComboBox()
        self.sheet_combo.setFixedWidth(250)
        self.sheet_combo.addItem("All sheets (merged)")
        self.sheet_combo.currentTextChanged.connect(self.change_sheet)
        self.sheet_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                selection-background-color: {PALETTE['primary']};
            }}
            """
        )
        self.sheet_combo.setItemDelegate(CenterDelegate(self.sheet_combo))

        sheet_layout.addWidget(self.sheet_combo)
        sheet_layout.addStretch()
        layout.addLayout(sheet_layout)

        # ---------- PLOTTING OPTIONS ----------
        subtitle_load = QLabel("Plotting Options")
        subtitle_load.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle_load.setStyleSheet(f"color: {PALETTE['text']};")
        subtitle_load.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(subtitle_load)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {PALETTE['bg']};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {PALETTE['border']};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {PALETTE['primary']};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

        scroll_container = QWidget()
        scroll_container.setStyleSheet("background: transparent;")
        plot_option = QVBoxLayout(scroll_container)
        plot_option.setContentsMargins(0, 8, 0, 8)
        plot_option.setSpacing(8)

        # Select / Deselect All
        self.select_all_cb = QCheckBox("Select / Deselect All")
        self.select_all_cb.setStyleSheet(
            f"""
            QCheckBox {{
                color: {PALETTE['accent']};
                spacing: 8px;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 0px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {PALETTE['accent']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {PALETTE['primary']};
            }}
            QCheckBox::indicator:checked {{
                background: {PALETTE['accent']};
                border: 2px solid {PALETTE['accent']};
            }}
            """
        )
        self.select_all_cb.stateChanged.connect(self._toggle_all_plots)
        plot_option.addWidget(self.select_all_cb)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(
            f"background: {PALETTE['border']}; max-height: 1px; margin: 4px 0px;"
        )
        plot_option.addWidget(divider)

        # Plot types and checkboxes
        plot_types = [
            "Histogram",
            "Scatter Plot",
            "Correlation Heatmap",
            "Bar Chart",
            "Line Plot",
            "Overlapping Histogram",
            "Pie Chart",  
            "Ridgeline Plot",
            "Lollipop Chart",
            "Diverging Bar Chart",
            "Stacked Area Chart",
            "KPI Card",
            "Small Multiples",
            "Highlight Table",
            "TLS Distribution",
            "Geographic Distribution",
            "Student Demographics",
            "Success Rates Analysis",
            "Score Distributions by Grade",
        ]

        for plot_type in plot_types:
            cb = QCheckBox(plot_type)
            cb.setStyleSheet(
                f"""
                QCheckBox {{
                    color: {PALETTE['text']};
                    spacing: 8px;
                    font-size: 13px;
                    padding: 4px 0px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 1px solid {PALETTE['border']};
                    border-radius: 3px;
                }}
                QCheckBox::indicator:hover {{
                    border: 1px solid {PALETTE['primary']};
                }}
                QCheckBox::indicator:checked {{
                    background: {PALETTE['primary']};
                    border: 1px solid {PALETTE['primary']};
                }}
                """
            )
            cb.stateChanged.connect(
                lambda state, pt=plot_type: self._on_plot_toggled(pt, state)
            )
            plot_option.addWidget(cb)
            self.plot_checkboxes[plot_type] = cb

        plot_option.addStretch(1)
        scroll_area.setWidget(scroll_container)
        layout.addWidget(scroll_area, stretch=1)

        # ---------- STATUS / SUMMARY ----------
        summary_title = QLabel("Generation Summary")
        summary_title.setStyleSheet(
            f"color: {PALETTE['text']}; font-size: 14px; font-weight: 600; margin-top: 8px;"
        )
        layout.addWidget(summary_title)

        self.summary_label = QLabel(
            "No plots generated yet. Select plot types above to begin."
        )
        self.summary_label.setStyleSheet(
            f"""
            QLabel {{
                color: {PALETTE['text']};
                background: {PALETTE['panel']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
            }}
            """
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setMinimumHeight(80)
        layout.addWidget(self.summary_label)

        # ---------- NAV BUTTONS AT THE END ----------
        nav_row = QHBoxLayout()
        nav_row.setSpacing(12)

        # --- RIGHT SIDE: BACK + EXIT ---
        right_buttons = QHBoxLayout()
        right_buttons.setSpacing(10)

        self.back_btn = QPushButton("Back")
        self.back_btn.setFixedSize(140, 38)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['panel']};
                color:{PALETTE['text']};
                border:1px solid {PALETTE['border']};
                border-radius:6px;
                padding:8px 16px;
                font-size:14px;
            }}
            QPushButton:hover {{
                background:{PALETTE['surface']};
                border:1px solid {PALETTE['primary']};
            }}
            QPushButton:pressed {{
                background:{PALETTE['border']};
            }}
            """
        )
        self.back_btn.clicked.connect(self.go_back)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setFixedSize(140, 38)
        self.exit_btn.setCursor(Qt.PointingHandCursor)
        self.exit_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['primary']};
                color:{PALETTE['text']};
                border:none;
                border-radius:6px;
                padding:8px 16px;
                font-size:14px;
            }}
            QPushButton:hover {{
                background:{PALETTE['accent_hover']};
            }}
            QPushButton:pressed {{
                background:{PALETTE['accent_pressed']};
            }}
            """
        )
        self.exit_btn.clicked.connect(self.exit_app)

        right_buttons.addWidget(self.back_btn)
        right_buttons.addWidget(self.exit_btn)

        nav_row.addStretch()          # push right buttons to the right
        nav_row.addLayout(right_buttons)

        layout.addLayout(nav_row)

    # ---------- NAVIGATION METHODS ----------
    def go_back(self):
        self.cleanup_threads() 
        """Navigate back to previous window (e.g. cleaning page)."""
        main_window = self.window()
        if hasattr(main_window, "go_cleaning"):
            main_window.go_cleaning()
        elif hasattr(main_window, "stack"):
            # Fallback if using a QStackedWidget; adjust index as needed
            try:
                main_window.stack.setCurrentIndex(1)
            except Exception:
                pass

    def exit_app(self):
        self.cleanup_threads() 
        QApplication.quit()

    def _initialize_visualizer(self):
        if self.current_df is None or self.current_df.empty:
            return False

        try:
            if self.processed_path:
                self.output_dir = Path(self.processed_path).parent / "plots"
            else:
                self.output_dir = Path.cwd() / "plots"

            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.visualizer = DataVisualizer(
                dataset_path="",  
                output_dir=self.output_dir,
                df=self.current_df
            )

            print(f"✓ Visualizer initialized with {len(self.current_df)} rows")
            print(f"✓ Plots will be saved to: {self.output_dir}")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize visualizer: {str(e)}")
            self._update_error_summary(f"Failed to initialize: {str(e)}")
            return False


    def set_dataframes(self, merged_df, sheets_dfs: dict, processed_path: Optional[str] = None):
        """Set dataframes + optional processed file path from cleaning window."""
        self.merged_df = merged_df
        self.sheets_dfs = sheets_dfs or {}
        self.current_sheet = None
        self.current_df = self.merged_df
        if processed_path is not None:
            self.processed_path = processed_path

        # Populate sheet combo
        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")
        for sheet_name in self.sheets_dfs.keys():
            self.sheet_combo.addItem(sheet_name)

        # Initialize visualizer with new data
        self._initialize_visualizer()

    def set_dataframe(self, df, processed_path: Optional[str] = None):
        """Set a single dataframe (no sheets)."""
        self.merged_df = df
        self.sheets_dfs = {}
        self.current_sheet = None
        self.current_df = self.merged_df
        if processed_path is not None:
            self.processed_path = processed_path

        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")

        self._initialize_visualizer()

    # ---------- SHEET CHANGES ----------
    def change_sheet(self, name: str):
        """Handle sheet selection change."""
        if name == "All sheets (merged)" or not name:
            self.current_sheet = None
            self.current_df = self.merged_df
        else:
            self.current_sheet = name
            self.current_df = self.sheets_dfs.get(name, self.merged_df)

        print(f"Sheet changed to: {name}")
        print(
            f"Current dataframe shape: "
            f"{self.current_df.shape if self.current_df is not None else 'None'}"
        )

        # Reinitialize visualizer with new dataframe
        self._initialize_visualizer()

        # Uncheck all plots when changing sheets
        for cb in self.plot_checkboxes.values():
            cb.setChecked(False)

        # Clear summary
        self.completed_plots = []
        self.failed_plots = []
        self._update_summary()

    # ---------- PLOT CHECKBOX HANDLERS ----------
    def _toggle_all_plots(self, state):
        """Toggle all plot checkboxes on/off."""
        is_checked = state == Qt.CheckState.Checked.value
        for cb in self.plot_checkboxes.values():
            cb.setChecked(is_checked)

    def _on_plot_toggled(self, plot_type, state):
        """Handle individual plot checkbox toggle."""
        is_checked = state == Qt.CheckState.Checked.value

        if is_checked:
            self._activate_plot(plot_type)
        else:
            self._deactivate_plot(plot_type)
            # Remove from tracking if unchecked
            if plot_type in self.completed_plots:
                self.completed_plots.remove(plot_type)
            if plot_type in self.failed_plots:
                self.failed_plots.remove(plot_type)
            self._update_summary()

    def _activate_plot(self, plot_type):
        """Start generation for the given plot type."""
        if self.current_df is None or self.current_df.empty:
            print("⚠ No data available. Please load data first.")
            self._update_error_summary("⚠ No data available. Please load data from the previous steps first.")
            self.plot_checkboxes[plot_type].setChecked(False)
            return

        # Ensure output directory exists
        if not self.output_dir:
            if self.processed_path:
                self.output_dir = Path(self.processed_path).parent / "plots"
            else:
                self.output_dir = Path.cwd() / "plots"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        sheet_name = self.current_sheet if self.current_sheet else "All sheets (merged)"
        print(f"⚙ Generating {plot_type} for sheet: {sheet_name}")

        self._update_progress_summary(plot_type)

        # IMPORTANT: thread uses df copy + output_dir, no shared visualizer
        thread = PlottingThread(plot_type, self.current_df, self.output_dir)
        thread.finished.connect(lambda pt, success, msg: self._on_plot_finished(pt, success, msg))
        thread.start()

        self.active_threads[plot_type] = thread


    def _deactivate_plot(self, plot_type):
        sheet_name = self.current_sheet if self.current_sheet else "All sheets (merged)"
        print(f"Deactivating plot: {plot_type} for sheet: {sheet_name}")

        if plot_type in self.active_threads:
            thread = self.active_threads[plot_type]
            if thread.isRunning():
                thread.quit()
                thread.wait()
            del self.active_threads[plot_type]


    def _on_plot_finished(self, plot_type, success, message):
        thread = self.active_threads.get(plot_type)

        if thread and thread.isRunning():
            thread.quit()
            thread.wait()

        if success:
            print(f"✓ {message}")
            if plot_type not in self.completed_plots:
                self.completed_plots.append(plot_type)
            if plot_type in self.failed_plots:
                self.failed_plots.remove(plot_type)
        else:
            print(f"✗ {message}")
            if plot_type not in self.failed_plots:
                self.failed_plots.append(plot_type)
            if plot_type in self.completed_plots:
                self.completed_plots.remove(plot_type)
            self.plot_checkboxes[plot_type].setChecked(False)

        if plot_type in self.active_threads:
            del self.active_threads[plot_type]

        self._update_summary()

    def _update_progress_summary(self, plot_type):
        """Show that a plot is being generated."""
        summary = f'<span style="color: #88ccff;">⚙ Generating {plot_type}...</span>'
        if self.completed_plots or self.failed_plots:
            summary = self._build_summary() + "<br><br>" + summary
        self.summary_label.setText(summary)

    def _update_error_summary(self, error_msg):
        """Display error message in summary."""
        self.summary_label.setText(
            f'<span style="color: #ff4444;">{error_msg}</span>'
        )

    def _update_summary(self):
        """Update summary label with overall plot status."""
        if not self.completed_plots and not self.failed_plots:
            self.summary_label.setText(
                "No plots generated yet. Select plot types above to begin."
            )
            return

        self.summary_label.setText(self._build_summary())

    def _build_summary(self):
        """Build rich-text summary HTML."""
        summary_parts = []

        if self.completed_plots:
            summary_parts.append(
                "<span style='color: #00ff88;'><b>✓ Successfully Generated "
                f"({len(self.completed_plots)}):</b></span><br>"
            )
            summary_parts.append(", ".join(self.completed_plots))

        if self.failed_plots:
            if summary_parts:
                summary_parts.append("<br><br>")
            summary_parts.append(
                "<span style='color: #ff4444;'><b>✗ Failed "
                f"({len(self.failed_plots)}):</b></span><br>"
            )
            summary_parts.append(", ".join(self.failed_plots))

        if self.output_dir and self.completed_plots:
            summary_parts.append(
                "<br><br>📁 <b>Output Folder:</b><br>"
                f"<span style='font-size: 11px;'>{self.output_dir}</span>"
            )

        return "".join(summary_parts)

    # ---------- PUBLIC HELPERS ----------
    def get_selected_plots(self):
        """Return list of currently selected plot types."""
        return [
            plot_type
            for plot_type, cb in self.plot_checkboxes.items()
            if cb.isChecked()
        ]

    def get_output_directory(self):
        """Return directory where plots are being saved."""
        return self.output_dir

    # ---------- LIFECYCLE ----------
    def showEvent(self, event):
        """Auto-load data from cleaning window when shown."""
        super().showEvent(event)
        main_window = self.window()

        if hasattr(main_window, "cleaning_window"):
            cleaning = main_window.cleaning_window

            if hasattr(cleaning, "merged_df") and cleaning.merged_df is not None:
                processed_path = getattr(cleaning, "processed_path", None)
                self.set_dataframes(cleaning.merged_df, cleaning.sheets_dfs, processed_path)

    def cleanup_threads(self):
        for plot_type, thread in list(self.active_threads.items()):
            if thread.isRunning():
                thread.quit()
                thread.wait()
            del self.active_threads[plot_type]

