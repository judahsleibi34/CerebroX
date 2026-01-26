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

# Import the EnhancedEduVisualizer from your EDA module
from EDA import EnhancedEduVisualizer


class PlottingThread(QThread):
    finished = Signal(str, bool, str)  # plot_type, success, message

    def __init__(self, plot_type, df, output_dir):
        super().__init__()
        self.plot_type = plot_type
        self.df = df.copy(deep=True)
        self.output_dir = output_dir

    def _clean_columns(self):
        self.df.columns = self.df.columns.astype(str).str.strip().str.replace("\n", " ", regex=False)

    def _is_id_like(self, s: pd.Series) -> bool:
        s = s.dropna()
        return (s.nunique() / len(s)) > 0.95 if not s.empty else False

    def _best_numeric_cols(self, k=3):
        nums = self.df.select_dtypes(include=["number"])
        stats = []
        for c in nums.columns:
            s = nums[c].dropna()
            if len(s) < 5 or self._is_id_like(s): continue
            stats.append((c, s.notna().mean(), s.var()))
        stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [c for c, _, _ in stats[:k]]

    def _pick_group_col(self, prefer_keywords=None, max_unique=12):
        # Look for categorical-like columns
        candidates = []
        for c in self.df.columns:
            nu = self.df[c].nunique()
            if 2 <= nu <= max_unique:
                candidates.append(c)
        
        if prefer_keywords and candidates:
            for c in candidates:
                if any(k.lower() in c.lower() for k in prefer_keywords):
                    return c
        return candidates[0] if candidates else None

    def run(self):
        try:
            self._clean_columns()
            
            # Create output directory if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize EnhancedEduVisualizer with the dataframe
            viz = EnhancedEduVisualizer(self.df)
            
            # Map plot types to EnhancedEduVisualizer methods
            if self.plot_type == "Histogram":
                # Generate histograms for all numeric columns
                for col in viz.numeric_cols:
                    save_path = f"{self.output_dir}/{col}_histogram.png"
                    viz.plot_numeric_histogram(col, save_path=save_path)
                self.finished.emit(self.plot_type, True, f"Histograms generated for {len(viz.numeric_cols)} columns")
            
            elif self.plot_type == "Bar Chart":
                # Generate bar charts for all categorical columns
                for col in viz.categorical_cols:
                    save_path = f"{self.output_dir}/{col}_bar.png"
                    viz.plot_categorical_bar(col, save_path=save_path)
                self.finished.emit(self.plot_type, True, f"Bar charts generated for {len(viz.categorical_cols)} columns")

            elif self.plot_type == "Pie Chart":
                group = self._pick_group_col(max_unique=10)
                if group and group in viz.categorical_cols:
                    save_path = f"{self.output_dir}/{group}_pie.png"
                    viz.plot_categorical_bar(group, save_path=save_path)  # Using bar chart as pie chart alternative
                    self.finished.emit(self.plot_type, True, f"Pie chart for {group} generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No suitable categorical column found for pie chart")

            elif self.plot_type == "Scatter Plot":
                cols = self._best_numeric_cols(2)
                if len(cols) >= 2:
                    # EnhancedEduVisualizer doesn't have scatter plot, so we'll create a basic one
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(self.df[cols[0]], self.df[cols[1]], alpha=0.6)
                    ax.set_xlabel(cols[0])
                    ax.set_ylabel(cols[1])
                    ax.set_title(f"Scatter Plot: {cols[0]} vs {cols[1]}")
                    ax.grid(True, alpha=0.3)
                    
                    save_path = f"{self.output_dir}/scatter_{cols[0]}_vs_{cols[1]}.png"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    self.finished.emit(self.plot_type, True, f"Scatter plot: {cols[0]} vs {cols[1]}")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns for scatter plot")

            elif self.plot_type == "Correlation Heatmap":
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                numeric_cols = self._best_numeric_cols(15)
                if len(numeric_cols) >= 2:
                    corr_matrix = self.df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                    ax.set_title('Correlation Heatmap', fontsize=16, pad=20)
                    
                    save_path = f"{self.output_dir}/correlation_heatmap.png"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    self.finished.emit(self.plot_type, True, "Correlation heatmap generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns for correlation heatmap")

            elif self.plot_type == "Line Plot":
                cols = self._best_numeric_cols(1)
                if cols:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(self.df.index, self.df[cols[0]], marker='o', linewidth=1.5)
                    ax.set_xlabel('Index')
                    ax.set_ylabel(cols[0])
                    ax.set_title(f'Line Plot: {cols[0]}')
                    ax.grid(True, alpha=0.3)
                    
                    save_path = f"{self.output_dir}/line_plot_{cols[0]}.png"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    self.finished.emit(self.plot_type, True, f"Line plot for {cols[0]} generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No numeric columns found for line plot")

            elif self.plot_type == "Box Plot":
                cols = self._best_numeric_cols(3)
                if cols:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 6))
                    data_to_plot = [self.df[col].dropna() for col in cols if col in self.df.columns]
                    ax.boxplot(data_to_plot, labels=cols)
                    ax.set_title('Box Plot')
                    ax.set_ylabel('Value')
                    plt.xticks(rotation=45)
                    
                    save_path = f"{self.output_dir}/box_plot.png"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    self.finished.emit(self.plot_type, True, f"Box plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No numeric columns found for box plot")

            elif self.plot_type == "Comparison Plot":
                # Find a categorical column and numeric columns for comparison
                cat_col = self._pick_group_col(max_unique=10)
                numeric_cols = self._best_numeric_cols(3)
                
                if cat_col and numeric_cols:
                    save_path = f"{self.output_dir}/comparison_{cat_col}.png"
                    viz.plot_comparison_by_category(numeric_cols, cat_col, save_path=save_path)
                    self.finished.emit(self.plot_type, True, f"Comparison plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need categorical and numeric columns for comparison plot")

            elif self.plot_type == "All Plots":
                save_path = f"{self.output_dir}/all_plots"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                viz.generate_all_plots(save_dir=save_path)
                self.finished.emit(self.plot_type, True, "All plots generated successfully")

            else:
                self.finished.emit(self.plot_type, False, f"Plot type '{self.plot_type}' not implemented")

        except Exception as e:
            self.finished.emit(self.plot_type, False, f"Error: {str(e)}")


class CenterDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Plotting_Page(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.source_path: Optional[str] = None 
        self.setWindowTitle("Educational Data Visualization")
        self.setStyleSheet(f"background: {PALETTE['bg']};")
        self.source_path: Optional[str] = None   # <--- ADD THIS

        # Data-related attributes
        self.merged_df = None
        self.sheets_dfs = {}
        self.current_sheet = None
        self.current_df = None
        self.processed_path: Optional[str] = None

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
        main_title = QLabel("Educational Data Visualization")
        main_title.setAlignment(Qt.AlignLeft)
        main_title.setStyleSheet(f"color: {PALETTE['text']};")
        main_title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        layout.addWidget(main_title)

        # Subtitle
        subtitle = QLabel("Generate professional visualizations for educational datasets")
        subtitle.setStyleSheet(f"color: {PALETTE['text']}; font-size: 14px; opacity: 0.8;")
        layout.addWidget(subtitle)

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
        subtitle_load = QLabel("Visualization Options")
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

        # Dynamic plot types based on available methods
        plot_types = [
            "Histogram",
            "Bar Chart", 
            "Scatter Plot",
            "Correlation Heatmap",
            "Line Plot",
            "Box Plot",
            "Pie Chart",
            "Comparison Plot",
            "All Plots",
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
            "No visualizations generated yet. Select visualization types above to begin."
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

        # ---------- NAV BUTTONS ----------
        nav_row = QHBoxLayout()
        nav_row.setSpacing(12)

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

        nav_row.addStretch()
        nav_row.addLayout(right_buttons)

        layout.addLayout(nav_row)

    # ---------- NAVIGATION METHODS ----------
    def go_back(self):
        self.cleanup_threads()
        main_window = self.window()
        if hasattr(main_window, "go_cleaning"):
            main_window.go_cleaning()
        elif hasattr(main_window, "stack"):
            try:
                main_window.stack.setCurrentIndex(1)
            except Exception:
                pass

    def exit_app(self):
        self.cleanup_threads()
        QApplication.quit()

    def _initialize_visualizer(self):
        print("DEBUG self.processed_path =", repr(self.processed_path))
        print("DEBUG self.source_path   =", repr(getattr(self, "source_path", None)))
        print("DEBUG cwd =", Path.cwd())

        if self.current_df is None or self.current_df.empty:
            return False

        try:
            processed = (self.processed_path or "").strip() if isinstance(self.processed_path, str) else ""
            source = (getattr(self, "source_path", None) or "").strip()

            if processed:
                p = Path(processed).expanduser().resolve()

                # processed_path is a folder -> save plots inside it
                if p.exists() and p.is_dir():
                    self.output_dir = p / "plots"
                else:
                    # processed_path is a file -> save plots next to it
                    self.output_dir = p.parent / "plots"

            elif source:
                s = Path(source).expanduser().resolve()
                # if user only loads a file and never "saves cleaned"
                self.output_dir = s.parent / f"{s.stem}_plots"

            else:
                # last fallback (only if you truly have no file path)
                self.output_dir = Path.cwd().resolve() / "plots"

            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Plots will be saved to: {self.output_dir}")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize visualizer: {str(e)}")
            self._update_error_summary(f"Failed to initialize: {str(e)}")
            return False

    def set_dataframes(self, merged_df, sheets_dfs: dict,
                   processed_path: Optional[str] = None,
                   source_path: Optional[str] = None):
        self.merged_df = merged_df
        self.sheets_dfs = sheets_dfs or {}
        self.current_sheet = None
        self.current_df = self.merged_df

        self.processed_path = processed_path
        self.source_path = source_path  # <--- IMPORTANT

        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")
        for sheet_name in self.sheets_dfs.keys():
            self.sheet_combo.addItem(sheet_name)

        self._initialize_visualizer()

    def set_dataframe(self, df,
                  processed_path: Optional[str] = None,
                  source_path: Optional[str] = None):
        self.merged_df = df
        self.sheets_dfs = {}
        self.current_sheet = None
        self.current_df = self.merged_df

        self.processed_path = processed_path
        self.source_path = source_path  # <--- IMPORTANT

        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")

        self._initialize_visualizer()

    # ---------- SHEET CHANGES ----------
    def change_sheet(self, name: str):
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

        self._initialize_visualizer()

        for cb in self.plot_checkboxes.values():
            cb.setChecked(False)

        self.completed_plots = []
        self.failed_plots = []
        self._update_summary()

    # ---------- PLOT CHECKBOX HANDLERS ----------
    def _toggle_all_plots(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        for cb in self.plot_checkboxes.values():
            cb.setChecked(is_checked)

    def _on_plot_toggled(self, plot_type, state):
        is_checked = state == Qt.CheckState.Checked.value

        if is_checked:
            self._activate_plot(plot_type)
        else:
            self._deactivate_plot(plot_type)
            if plot_type in self.completed_plots:
                self.completed_plots.remove(plot_type)
            if plot_type in self.failed_plots:
                self.failed_plots.remove(plot_type)
            self._update_summary()

    def _activate_plot(self, plot_type):
        if self.current_df is None or self.current_df.empty:
            print("⚠ No data available. Please load data first.")
            self._update_error_summary("⚠ No data available. Please load data from the previous steps first.")
            self.plot_checkboxes[plot_type].setChecked(False)
            return

        if not self.output_dir:
            ok = self._initialize_visualizer()
            if not ok:
                return

        sheet_name = self.current_sheet if self.current_sheet else "All sheets (merged)"
        print(f"⚙ Generating {plot_type} for sheet: {sheet_name}")

        self._update_progress_summary(plot_type)

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
        summary = f'<span style="color: #88ccff;">⚙ Generating {plot_type}...</span>'
        if self.completed_plots or self.failed_plots:
            summary = self._build_summary() + "<br><br>" + summary
        self.summary_label.setText(summary)

    def _update_error_summary(self, error_msg):
        self.summary_label.setText(
            f'<span style="color: #ff4444;">{error_msg}</span>'
        )

    def _update_summary(self):
        if not self.completed_plots and not self.failed_plots:
            self.summary_label.setText(
                "No visualizations generated yet. Select visualization types above to begin."
            )
            return

        self.summary_label.setText(self._build_summary())

    def _build_summary(self):
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
        return [
            plot_type
            for plot_type, cb in self.plot_checkboxes.items()
            if cb.isChecked()
        ]

    def get_output_directory(self):
        return self.output_dir

    def showEvent(self, event):
        super().showEvent(event)
        main_window = self.window()

        if hasattr(main_window, "cleaning_window"):
            cleaning = main_window.cleaning_window

            if hasattr(cleaning, "merged_df") and cleaning.merged_df is not None:
                processed_path = getattr(cleaning, "processed_path", None)
                source_path = getattr(cleaning, "loaded_filename", None)  # <--- ADD THIS

                print("DEBUG cleaning.processed_path =", repr(processed_path))
                print("DEBUG cleaning.loaded_filename =", repr(source_path))

                self.set_dataframes(cleaning.merged_df, cleaning.sheets_dfs,
                                    processed_path=processed_path,
                                    source_path=source_path)

    def cleanup_threads(self):
        for plot_type, thread in list(self.active_threads.items()):
            if thread.isRunning():
                thread.quit()
                thread.wait()
            del self.active_threads[plot_type]