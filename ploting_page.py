import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots

from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QFrame, QComboBox, QScrollArea, QStyledItemDelegate, QPushButton
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from config import PALETTE
from EDA import DataVisualizer


class PlottingThread(QThread):
    finished = Signal(str, bool, str)  # plot_type, success, message

    def __init__(self, visualizer, plot_type, df):
        super().__init__()
        self.visualizer = visualizer
        self.plot_type = plot_type
        self.df = df

    def run(self):
        try:
            if self.plot_type == "Histogram":
                self.visualizer.plot_all_distributions()
                self.finished.emit(self.plot_type, True, "Histograms generated successfully")

            elif self.plot_type == "Scatter Plot":
                numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
                if len(numeric_cols) >= 2:
                    self.visualizer.scatter_plot(
                        "Scatter Plot",
                        numeric_cols[0],
                        numeric_cols[1]
                    )
                    self.finished.emit(self.plot_type, True, "Scatter plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns")

            elif self.plot_type == "Correlation Heatmap":
                self.visualizer.correlation_heatmap()
                self.finished.emit(self.plot_type, True, "Correlation heatmap generated successfully")

            elif self.plot_type == "Bar Chart":
                self.visualizer.plot_all_value_counts(max_categories=15)
                self.finished.emit(self.plot_type, True, "Bar charts generated successfully")

            elif self.plot_type == "Line Plot":
                numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
                if len(numeric_cols) >= 2:
                    self.visualizer.line_chart(
                        "Line Chart",
                        numeric_cols[0],
                        numeric_cols[1:][:3]  # First col as x, next up to 3 as y
                    )
                    self.finished.emit(self.plot_type, True, "Line plot generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "Need at least 2 numeric columns")

            elif self.plot_type == "Overlapping Histogram":
                # Find suitable categorical and numeric columns
                categorical_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()
                numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
                
                if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    # Use first categorical for grouping, first numeric for values
                    group_col = categorical_cols[0]
                    value_col = numeric_cols[0]
                    
                    self.visualizer.overlapping_histogram(
                        f"Distribution of {value_col} by {group_col}",
                        value_col,
                        group_col
                    )
                    self.finished.emit(self.plot_type, True, "Overlapping histogram generated successfully")

                else:
                    self.finished.emit(self.plot_type, False, "Need at least 1 categorical and 1 numeric column")

            elif self.plot_type == "Pie Chart":
                categorical_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()

                if len(categorical_cols) >= 1:
                    col = categorical_cols[0]

                    self.visualizer.pie_chart(
                        f"Pie Chart of {col}",
                        col,
                        annotate=True,
                        annotate_format="{:.1f}%%"
                    )
                    self.finished.emit(self.plot_type, True, "Pie chart generated successfully")
                else:
                    self.finished.emit(self.plot_type, False, "No categorical column found for pie chart")

            else:
                self.finished.emit(self.plot_type, False, f"{self.plot_type} not yet implemented")

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
            self._update_error_summary(
                "⚠ No data available. Please load data from the previous steps first."
            )
            self.plot_checkboxes[plot_type].setChecked(False)
            return

        if self.visualizer is None:
            if not self._initialize_visualizer():
                self.plot_checkboxes[plot_type].setChecked(False)
                return

        sheet_name = self.current_sheet if self.current_sheet else "All sheets (merged)"
        print(f"⚙ Generating {plot_type} for sheet: {sheet_name}")

        # Show progress in summary
        self._update_progress_summary(plot_type)

        # Create and start plotting thread
        thread = PlottingThread(self.visualizer, plot_type, self.current_df)
        thread.finished.connect(
            lambda pt, success, msg: self._on_plot_finished(pt, success, msg)
        )
        thread.start()

        self.active_threads[plot_type] = thread

    def _deactivate_plot(self, plot_type):
        """Stop generation thread for a plot (if running)."""
        sheet_name = self.current_sheet if self.current_sheet else "All sheets (merged)"
        print(f"Deactivating plot: {plot_type} for sheet: {sheet_name}")

        if plot_type in self.active_threads:
            thread = self.active_threads[plot_type]
            if thread.isRunning():
                thread.terminate()
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

