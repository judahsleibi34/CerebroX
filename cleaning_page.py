from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QComboBox, QCheckBox, QHBoxLayout, QScrollArea, QFrame,
    QTabWidget, QLineEdit, QMessageBox, QStyledItemDelegate
)

from pathlib import Path
import pandas as pd

from database import CerebroXDB
from config import PALETTE
from data_preperation import Data_analysis

class CenterDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class DatasetCleaning_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Cleaning")
        self.setStyleSheet(f"background: {PALETTE['bg']};")

        self.current_df = None          
        self.merged_df = None           
        self.sheets_dfs = {}             
        self.current_sheet = None      
        self.data_analyzer = None

        self.rename_edits = {}           
        self.value_edits = {}            
        self.drop_checkboxes = {}        
        self.processed_path = None      
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(16)

        main_title = QLabel("Dataset Cleaning")
        main_title.setAlignment(Qt.AlignLeft)
        main_title.setStyleSheet(f"color: {PALETTE['text']};")
        main_title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        main_layout.addWidget(main_title)

        self.sheet_combo = QComboBox()
        self.sheet_combo.setFixedWidth(200)
        self.sheet_combo.addItem("All sheets (merged)")
        self.sheet_combo.currentTextChanged.connect(self.change_sheet)
        self.sheet_combo.setStyleSheet(
            f"""
            QComboBox {{
                background:{PALETTE['panel']};
                color:{PALETTE['text']};
                border:1px solid {PALETTE['border']};
                border-radius:6px;
                padding:4px 8px;
            }}
            QComboBox QAbstractItemView {{
                background:{PALETTE['panel']};
                color:{PALETTE['text']};
                selection-background-color:{PALETTE['primary']};
            }}
            """
        )
        self.sheet_combo.setItemDelegate(CenterDelegate(self.sheet_combo))
        main_layout.addWidget(self.sheet_combo, alignment=Qt.AlignLeft)

        self.sheet_combo.setEditable(True)
        le = self.sheet_combo.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid {PALETTE['border']};
                background: {PALETTE['panel']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {PALETTE['surface']};
                color: {PALETTE['muted']};
                padding: 8px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {PALETTE['primary']};
                color: {PALETTE['text']};
            }}
            QTabBar::tab:hover {{
                background: {PALETTE['border']};
                color: {PALETTE['text']};
            }}
            """
        )
        main_layout.addWidget(self.tabs, stretch=1)

        self.renaming_tab = QWidget()
        renaming_layout = QVBoxLayout(self.renaming_tab)
        renaming_layout.setContentsMargins(16, 16, 16, 16)
        renaming_layout.setSpacing(12)

        rename_title = QLabel("Column Renaming")
        rename_title.setStyleSheet(f"color: {PALETTE['text']};")
        rename_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        renaming_layout.addWidget(rename_title)

        self.columns_scroll = QScrollArea()
        self.columns_scroll.setWidgetResizable(True)
        self.columns_scroll.setFrameShape(QFrame.NoFrame)
        self.columns_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.columns_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.columns_scroll.setStyleSheet(
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

        self.columns_outer = QWidget()
        self.columns_outer.setStyleSheet("background: transparent;")
        self.columns_outer_layout = QVBoxLayout(self.columns_outer)
        self.columns_outer_layout.setContentsMargins(0, 0, 0, 0)
        self.columns_outer_layout.setSpacing(0)
        self.columns_outer_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.columns_widget = QWidget()
        self.columns_widget.setStyleSheet("background: transparent;")
        self.columns_layout = QVBoxLayout(self.columns_widget)
        self.columns_layout.setContentsMargins(0, 8, 0, 8)
        self.columns_layout.setSpacing(8)

        self.columns_outer_layout.addWidget(self.columns_widget)
        self.columns_scroll.setWidget(self.columns_outer)

        split_layout = QHBoxLayout()
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(20)

        split_layout.addWidget(self.columns_scroll, stretch=2)

        self.values_panel = QWidget()
        self.values_panel.setStyleSheet("background: transparent;")
        values_layout = QVBoxLayout(self.values_panel)
        values_layout.setContentsMargins(0, 0, 0, 0)
        values_layout.setSpacing(8)

        value_title = QLabel("Value Mapping")
        value_title.setStyleSheet(f"color: {PALETTE['text']};")
        value_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        values_layout.addWidget(value_title)

        self.value_col_combo = QComboBox()
        self.value_col_combo.setStyleSheet(f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                selection-background-color: {PALETTE['primary']};
                selection-color: {PALETTE['text']};
                outline: 0;
            }}
            QComboBox QScrollBar:vertical {{
                background: {PALETTE['bg']};
                width: 10px;
                border-radius: 5px;
            }}
            QComboBox QScrollBar::handle:vertical {{
                background: {PALETTE['border']};
                border-radius: 5px;
                min-height: 30px;
            }}
        """)
        self.value_col_combo.currentIndexChanged.connect(self._on_value_col_changed)
        values_layout.addWidget(self.value_col_combo)

        self.values_scroll = QScrollArea()
        self.values_scroll.setWidgetResizable(True)
        self.values_scroll.setFrameShape(QFrame.NoFrame)
        self.values_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.values_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.values_scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            """
        )

        self.values_inner = QWidget()
        self.values_inner.setStyleSheet("background: transparent;")
        self.values_inner_layout = QVBoxLayout(self.values_inner)
        self.values_inner_layout.setContentsMargins(0, 8, 0, 8)
        self.values_inner_layout.setSpacing(6)
        self.values_scroll.setWidget(self.values_inner)

        values_layout.addWidget(self.values_scroll, stretch=1)
        split_layout.addWidget(self.values_panel, stretch=1)
        renaming_layout.addLayout(split_layout, stretch=1)

        self.tabs.addTab(self.renaming_tab, "Renaming")

        self.cleaning_tab = QWidget()
        cleaning_main_layout = QVBoxLayout(self.cleaning_tab)
        cleaning_main_layout.setContentsMargins(16, 16, 16, 16)
        cleaning_main_layout.setSpacing(16)

        wrangling_title = QLabel("Data Wrangling")
        wrangling_title.setStyleSheet(f"color: {PALETTE['text']};")
        wrangling_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        cleaning_main_layout.addWidget(wrangling_title)

        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setFrameShape(QFrame.NoFrame)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setStyleSheet(f"""
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
        """)

        scroll_container = QWidget()
        scroll_container.setStyleSheet("background: transparent;")
        cleaning_layout = QVBoxLayout(scroll_container)
        cleaning_layout.setContentsMargins(0, 0, 12, 0)
        cleaning_layout.setSpacing(20)

        quality_label = QLabel("Data Quality Actions")
        quality_label.setStyleSheet(f"color: {PALETTE['text']}; font-size: 14px; font-weight: 600;")
        cleaning_layout.addWidget(quality_label)

        null_container = QWidget()
        null_container.setStyleSheet("background: transparent;")
        null_layout = QHBoxLayout(null_container)
        null_layout.setContentsMargins(0, 0, 0, 0)
        null_layout.setSpacing(12)

        null_label = QLabel("Drop Null Values:")
        null_label.setStyleSheet(f"color: {PALETTE['text']}; font-size: 13px;")
        null_label.setFixedWidth(140)
        null_layout.addWidget(null_label)

        self.null_combo = QComboBox()
        self.null_combo.addItems(["Don't drop", "Drop rows with ANY null", "Drop rows with ALL nulls"])
        self.null_combo.setFixedWidth(240)
        self.null_combo.setStyleSheet(f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                selection-background-color: {PALETTE['primary']};
            }}
        """)
        null_layout.addWidget(self.null_combo)
        null_layout.addStretch()
        cleaning_layout.addWidget(null_container)

        dup_container = QWidget()
        dup_container.setStyleSheet("background: transparent;")
        dup_layout = QHBoxLayout(dup_container)
        dup_layout.setContentsMargins(0, 0, 0, 0)
        dup_layout.setSpacing(12)

        dup_label = QLabel("Drop Duplicates:")
        dup_label.setStyleSheet(f"color: {PALETTE['text']}; font-size: 13px;")
        dup_label.setFixedWidth(140)
        dup_layout.addWidget(usage_dup_label := dup_label)  # keep style

        self.dup_combo = QComboBox()
        self.dup_combo.addItems(["Don't drop", "Keep first occurrence", "Keep last occurrence"])
        self.dup_combo.setFixedWidth(240)
        self.dup_combo.setStyleSheet(f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                selection-background-color: {PALETTE['primary']};
            }}
        """)
        dup_layout.addWidget(self.dup_combo)
        dup_layout.addStretch()
        cleaning_layout.addWidget(dup_container)

        self.apply_quality_btn = QPushButton("Apply Data Quality Actions")
        self.apply_quality_btn.setFixedSize(240, 38)
        self.apply_quality_btn.setCursor(Qt.PointingHandCursor)
        self.apply_quality_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PALETTE['primary']};
                color: {PALETTE['text']};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {PALETTE['accent_hover']}; }}
            QPushButton:pressed {{ background: {PALETTE['glow']}; }}
        """)
        self.apply_quality_btn.clicked.connect(self.apply_quality_actions)
        cleaning_layout.addWidget(self.apply_quality_btn, alignment=Qt.AlignLeft)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        cleaning_layout.addWidget(divider)

        drop_label = QLabel("Select Columns to Drop")
        drop_label.setStyleSheet(f"color: {PALETTE['text']}; font-size: 14px; font-weight: 600;")
        cleaning_layout.addWidget(drop_label)

        self.drop_inner = QWidget()
        self.drop_inner.setStyleSheet("background: transparent;")
        self.drop_inner_layout = QVBoxLayout(self.drop_inner)
        self.drop_inner_layout.setContentsMargins(0, 8, 0, 8)
        self.drop_inner_layout.setSpacing(8)
        cleaning_layout.addWidget(self.drop_inner)

        cleaning_layout.addStretch(1)
        main_scroll.setWidget(scroll_container)
        cleaning_main_layout.addWidget(main_scroll)

        self.tabs.addTab(self.cleaning_tab, "Wrangling")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.apply_clean_btn = QPushButton("Apply & Save")
        self.apply_clean_btn.setFixedSize(160, 38)
        self.apply_clean_btn.setCursor(Qt.PointingHandCursor)
        self.apply_clean_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['primary']};
                color:{PALETTE['text']};
                border:none;
                border-radius:6px;
                padding:8px 16px;
                font-size:14px;
                font-weight:600;
            }}
            QPushButton:hover {{
                background:{PALETTE['accent_hover']};
            }}
            QPushButton:pressed {{
                background:{PALETTE['glow']};
            }}
            """
        )
        self.apply_clean_btn.clicked.connect(self.run_cleaning_process)
        button_row.addWidget(self.apply_clean_btn)

        self.plot_btn = QPushButton("Go to Plotting")
        self.plot_btn.setFixedSize(160, 38)
        self.plot_btn.setCursor(Qt.PointingHandCursor)
        self.plot_btn.setStyleSheet(
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
        self.plot_btn.clicked.connect(self.go_to_plot_frame)
        button_row.addWidget(self.plot_btn)

        button_row.addStretch()    
        button_row.addStretch()

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

        button_row.addWidget(self.back_btn)
        button_row.addWidget(self.exit_btn)
        main_layout.addLayout(button_row)

    def go_to_plot_frame(self):
            main_window = self.window()
            if hasattr(main_window, "stack") and hasattr(main_window, "plotting"):
                main_window.stack.setCurrentWidget(main_window.plotting)
            else:
                self.show_msg(
                    QMessageBox.Icon.Warning,
                    "Plotting Screen Not Found",
                    "Could not find the plotting screen in the main window.\n"
                    "Make sure you created main_window.plotting and added it to main_window.stack."
                )


    def apply_quality_actions(self):
        selected_sheet = self.sheet_combo.currentText()

        if not selected_sheet:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "No Sheet Selected",
                "Please select a sheet first."
            )
            return

        if selected_sheet == "All sheets (merged)":
            df = self.merged_df
        elif selected_sheet in self.sheets_dfs:
            df = self.sheets_dfs[selected_sheet]
        else:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "Invalid Sheet",
                "The selected sheet is not available."
            )
            return

        if df is None or df.empty:
            self.show_msg(QMessageBox.Icon.Warning, "Empty Sheet", "The selected sheet is empty.")
            return

        actions_performed = []
        initial_rows = len(df)

        try:
            null_option = self.null_combo.currentText()
            if null_option == "Drop rows with ANY null":
                before = len(df)
                df.dropna(how='any', inplace=True)
                rows_removed = before - len(df)
                actions_performed.append(f"✓ Dropped {rows_removed} rows with ANY null values")

            elif null_option == "Drop rows with ALL nulls":
                before = len(df)
                df.dropna(how='all', inplace=True)
                rows_removed = before - len(df)
                actions_performed.append(f"✓ Dropped {rows_removed} rows with ALL null values")

            dup_option = self.dup_combo.currentText()
            if dup_option == "Keep first occurrence":
                before = len(df)
                df.drop_duplicates(keep='first', inplace=True)
                rows_removed = before - len(df)
                actions_performed.append(f"✓ Removed {rows_removed} duplicate rows (kept first)")

            elif dup_option == "Keep last occurrence":
                before = len(df)
                df.drop_duplicates(keep='last', inplace=True)
                rows_removed = before - len(df)
                actions_performed.append(f"✓ Removed {rows_removed} duplicate rows (kept last)")

            if selected_sheet == "All sheets (merged)":
                self.merged_df = df
            else:
                self.sheets_dfs[selected_sheet] = df

            final_rows = len(df)
            total_removed = initial_rows - final_rows

            if actions_performed:
                summary = f"Sheet: {selected_sheet}\n\n"
                summary += "\n".join(actions_performed)
                summary += f"\n\nTotal rows removed: {total_removed}"
                summary += f"\nRows remaining: {final_rows}"
                self.show_msg(QMessageBox.Icon.Information, "Actions Applied Successfully", summary)
            else:
                self.show_msg(QMessageBox.Icon.Information, "No Actions Applied",
                              "No null or duplicate handling options were selected.")

            self.populate_drop_columns()

        except Exception as e:
            self.show_msg(QMessageBox.Icon.Critical, "Error", f"Failed to apply quality actions:\n{e}")

    def _clear_drop_layout(self):
        while self.drop_inner_layout.count():
            item = self.drop_inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.drop_checkboxes.clear()

    def populate_drop_columns(self):
        self._clear_drop_layout()

        selected_sheet = self.sheet_combo.currentText()

        if not selected_sheet:
            return

        if selected_sheet == "All sheets (merged)":
            df = self.merged_df
        elif selected_sheet in self.sheets_dfs:
            df = self.sheets_dfs[selected_sheet]
        else:
            return

        if df is None or df.empty:
            return

        self.select_all_cb = QCheckBox("Select / Deselect All")
        self.select_all_cb.setStyleSheet(f"""
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
        """)
        self.select_all_cb.stateChanged.connect(self._toggle_all_columns)
        self.drop_inner_layout.addWidget(self.select_all_cb)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px; margin: 4px 0px;")
        self.drop_inner_layout.addWidget(divider)

        for col in df.columns:
            cb = QCheckBox(str(col))
            cb.setStyleSheet(f"""
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
            """)

            self.drop_inner_layout.addWidget(cb)
            self.drop_checkboxes[col] = cb

        self.drop_inner_layout.addStretch(1)

    def _toggle_all_columns(self, state):
        checked = (state == Qt.Checked)
        for cb in self.drop_checkboxes.values():
            cb.setChecked(checked)

    def drop_selected_columns(self):
        selected_sheet = self.sheet_combo.currentText()

        if not selected_sheet:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "No Sheet Selected",
                "Please select a sheet first."
            )
            return

        if selected_sheet == "All sheets (merged)":
            df = self.merged_df
        elif selected_sheet in self.sheets_dfs:
            df = self.sheets_dfs[selected_sheet]
        else:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "Invalid Sheet",
                "The selected sheet is not available."
            )
            return

        to_drop = [col for col, cb in self.drop_checkboxes.items() if cb.isChecked()]
        if not to_drop:
            self.show_msg(QMessageBox.Icon.Information, "Nothing selected",
                          "Please check at least one column to drop.")
            return

        df.drop(columns=to_drop, inplace=True, errors="ignore")

        if selected_sheet == "All sheets (merged)":
            self.merged_df = df
        else:
            self.sheets_dfs[selected_sheet] = df

        self.show_msg(
            QMessageBox.Icon.Information,
            "Columns Dropped",
            f"Dropped {len(to_drop)} column(s) from sheet '{selected_sheet}':\n\n" + "\n".join(to_drop)
        )

        self.populate_drop_columns()

    def on_tab_changed(self, index: int):
        is_wrangling_tab = (self.tabs.widget(index) is self.cleaning_tab)
        if is_wrangling_tab:
            self.populate_drop_columns()

    def _on_value_col_changed(self, index: int):
        if index < 0:
            return
        real_col = self.value_col_combo.itemData(index)
        if real_col:
            self.show_values_for_column(real_col)

    def _populate_value_col_combo(self):
        self.value_col_combo.blockSignals(True)
        self.value_col_combo.clear()

        if self.merged_df is not None:
            obj_cols = self.merged_df.select_dtypes(include="object").columns
            for col in obj_cols:
                raw = str(col)
                parts = [p.strip() for p in raw.split('/') if p.strip()]
                if len(parts) >= 2 and parts[-1] in ("العلامات", "العـلامات"):
                    display_label = " / ".join(parts[:-1])
                else:
                    display_label = raw.strip()
                self.value_col_combo.addItem(display_label, col)

        self.value_col_combo.blockSignals(False)

        if self.value_col_combo.count() > 0:
            self.value_col_combo.setCurrentIndex(0)
            real_col = self.value_col_combo.itemData(0)
            if real_col:
                self.show_values_for_column(real_col)

    def _normalize_headers_inplace(self, df: pd.DataFrame):
        """Use Data_analysis._normalize_header on all column names."""
        if df is None:
            return
        df.columns = [Data_analysis._normalize_header(c) for c in df.columns]

    def _normalize_values_inplace(self, df: pd.DataFrame):
        """
        Use Data_analysis._normalize_value on all object columns.
        """
        if df is None:
            return

        obj_cols = df.select_dtypes(include="object").columns
        if not len(obj_cols):
            return

        for col in obj_cols:
            df[col] = df[col].map(Data_analysis._normalize_value)

    def _safe_filename(self, name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()


    def run_cleaning_process(self):
        if self.current_df is None and self.merged_df is None:
            self.show_msg(QMessageBox.Icon.Warning, "No Data", "Please load a dataset first.")
            return
        
        selected_sheet = self.sheet_combo.currentText()
        if not selected_sheet or selected_sheet == "All sheets (merged)":
            df = self.merged_df
            sheet_key = None
        else:
            df = self.sheets_dfs.get(selected_sheet, self.merged_df)
            sheet_key = selected_sheet

        try:
            if df is not None and not df.empty:

                null_option = self.null_combo.currentText()
                if null_option == "Drop rows with ANY null":
                    df.dropna(how="any", inplace=True)
                elif null_option == "Drop rows with ALL nulls":
                    df.dropna(how="all", inplace=True)

                dup_option = self.dup_combo.currentText()
                if dup_option == "Keep first occurrence":
                    df.drop_duplicates(keep="first", inplace=True)
                elif dup_option == "Keep last occurrence":
                    df.drop_duplicates(keep="last", inplace=True)

                to_drop = [col for col, cb in self.drop_checkboxes.items() if cb.isChecked()]
                if to_drop:
                    df.drop(columns=to_drop, inplace=True, errors="ignore")

                if sheet_key is None:
                    self.merged_df = df
                    self.current_df = self.merged_df
                else:
                    self.sheets_dfs[sheet_key] = df
                    self.current_df = self.sheets_dfs[sheet_key]

        except Exception as e:
            self.show_msg(QMessageBox.Icon.Critical, "Error", f"Failed to apply wrangling actions:\n{e}")
            return
        
        rename_map = {}
        for old_name, edit_widget in self.rename_edits.items():
            if edit_widget is None:
                continue
            new_name = edit_widget.text().strip()
            if new_name and new_name != old_name:
                rename_map[old_name] = new_name

        try:
            if rename_map:
                if self.current_df is not None:  
                    self.current_df.rename(columns=rename_map, inplace=True)

                if self.merged_df is not None and self.current_df is not self.merged_df:
                    self.merged_df.rename(columns=rename_map, inplace=True)

                for sheet_name, sdf in self.sheets_dfs.items():
                    sdf.rename(columns=rename_map, inplace=True)

                renamed_list = "\n".join([f"'{old}' → '{new}'" for old, new in rename_map.items()])
                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Columns Renamed",
                    f"Successfully renamed {len(rename_map)} columns:\n\n{renamed_list}"
                )

            total_value_maps = 0
            if self.merged_df is not None:
                for col_name, mapping_widgets in self.value_edits.items():
                    if col_name not in self.merged_df.columns:
                        continue

                    replace_map = {}
                    for old_val, edit in mapping_widgets.items():
                        new_val = edit.text().strip()
                        if new_val and new_val != old_val:
                            replace_map[old_val] = new_val

                    if not replace_map:
                        continue

                    self.merged_df[col_name] = self.merged_df[col_name].replace(replace_map)

                    for sheet_name, sdf in self.sheets_dfs.items():
                        if col_name in sdf.columns:
                            sdf[col_name] = sdf[col_name].replace(replace_map)

                    total_value_maps += len(replace_map)

            if total_value_maps:
                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Value Mappings Applied",
                    f"Applied {total_value_maps} value mappings."
                )

            if self.merged_df is not None:
                self._normalize_values_inplace(self.merged_df)
                for sheet_name, sdf in self.sheets_dfs.items():
                    self._normalize_values_inplace(sdf)
            elif self.current_df is not None:
                self._normalize_values_inplace(self.current_df)

            if self.processed_path:
                p = Path(self.processed_path)
                renamed_path = p.with_name(p.stem + "_renamed.csv")
            else:
                renamed_path = Path.cwd() / "renamed.csv"

            df_to_save = self.merged_df if self.merged_df is not None else self.current_df
            df_to_save.to_csv(renamed_path, index=False, encoding="utf-8-sig")

            db_path = renamed_path.with_suffix(".db")
            db = CerebroXDB(db_path)
            table_name = db.save_snapshot(df=df_to_save, name="Cleaned Dataset", source_csv=renamed_path)

            if self.sheets_dfs:
                folder = renamed_path.parent / "cleaned_sheets"
                folder.mkdir(exist_ok=True)

                for sheet_name, sdf in self.sheets_dfs.items():
                    safe = self._safe_filename(sheet_name)

                    describe_path = folder / f"{safe}_describe.csv"
                    sdf.describe(include="all").to_csv(describe_path, encoding="utf-8-sig")

                    clean_path = folder / f"{safe}_cleaned.csv"
                    sdf.to_csv(clean_path, index=False, encoding="utf-8-sig")

                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Sheets Saved",
                    f"Each sheet has been saved along with its statistics inside:\n{folder}"
                )

            self.show_msg(
                QMessageBox.Icon.Information,
                "File Saved",
                f"Cleaned dataset saved as:\n{renamed_path}\n\n"
                f"Database snapshot saved in:\n{db_path}\nTable: {table_name}"
            )

        except Exception as e:
            self.show_msg(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed during cleaning/saving:\n{e}"
            )
            return

        self.show_columns()
        self._populate_value_col_combo()
        self.populate_drop_columns()


    def set_dataframes(self, merged_df: pd.DataFrame, sheets_dfs: dict):
        self.merged_df = merged_df
        self.sheets_dfs = sheets_dfs or {}
        self.current_sheet = None

        self._normalize_headers_inplace(self.merged_df)
        for sheet_name, df in self.sheets_dfs.items():
            self._normalize_headers_inplace(df)

        self.current_df = self.merged_df

        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")
        for sheet_name in self.sheets_dfs.keys():
            self.sheet_combo.addItem(sheet_name)

        self._populate_value_col_combo()
        self.show_columns()
        self.populate_drop_columns()

    def set_dataframe(self, df: pd.DataFrame):
        self.merged_df = df
        self.sheets_dfs = {}
        self.current_sheet = None

        self._normalize_headers_inplace(self.merged_df)
        self.current_df = self.merged_df

        self.sheet_combo.clear()
        self.sheet_combo.addItem("All sheets (merged)")

        self._populate_value_col_combo()
        self.show_columns()
        self.populate_drop_columns()

    def _clear_values_layout(self):
        while self.values_inner_layout.count():
            item = self.values_inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def show_values_for_column(self, col_name: str):
        self._clear_values_layout()
        self.value_edits = {}

        if not col_name or self.merged_df is None or col_name not in self.merged_df.columns:
            return

        series = self.merged_df[col_name]
        self.value_edits[col_name] = {}

        info_label = QLabel(f"Type: {series.dtype}, Missing: {series.isna().sum()}, Unique: {series.nunique()}")
        info_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        self.values_inner_layout.addWidget(info_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        self.values_inner_layout.addWidget(divider)

        uniques = series.dropna().unique()[:50]

        for val in uniques:
            orig = "" if pd.isna(val) else str(val)

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 4, 0, 4)
            hl.setSpacing(10)

            lbl = QLabel(orig)
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: {PALETTE['text']};
                    background: {PALETTE['surface']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 6px;
                    padding: 6px 10px;
                }}
            """)
            lbl.setFixedWidth(180)
            lbl.setWordWrap(True)

            edit = QLineEdit()
            edit.setPlaceholderText("New value...")
            edit.setFixedHeight(34)
            edit.setStyleSheet(f"""
                QLineEdit {{
                    color:{PALETTE['text']};
                    background:{PALETTE['panel']};
                    border:1px solid {PALETTE['border']};
                    border-radius:6px;
                    padding:6px 10px;
                }}
                QLineEdit:focus {{
                    border: 1px solid {PALETTE['primary']};
                }}
            """)

            hl.addWidget(lbl)
            hl.addWidget(edit, stretch=1)

            self.values_inner_layout.addWidget(row)
            self.value_edits[col_name][orig] = edit

        self.values_inner_layout.addStretch(1)

    def change_sheet(self, name: str):
        if name == "All sheets (merged)" or not name:
            self.current_sheet = None
            self.current_df = self.merged_df
        else:
            self.current_sheet = name
            self.current_df = self.sheets_dfs.get(name, self.merged_df)

        self.show_columns()
        self.populate_drop_columns()

    def showEvent(self, event):
        super().showEvent(event)
        main_window = self.window()
        if (self.current_df is None and hasattr(main_window, "loading") and
                getattr(main_window.loading, "current_df", None) is not None):
            sheets_dfs = getattr(main_window.loading, "sheets_dfs", {})
            self.processed_path = getattr(main_window.loading, "processed_path", None)
            self.set_dataframes(main_window.loading.current_df, sheets_dfs)

    def _clear_columns_layout(self):
        while self.columns_layout.count():
            item = self.columns_layout.takeAt(0)
            w = item.widget()
            lay = item.layout()
            if w is not None:
                w.deleteLater()
            elif lay is not None:
                while lay.count():
                    sub_item = lay.takeAt(0)
                    sub_w = sub_item.widget()
                    if sub_w is not None:
                        sub_w.deleteLater()

    def show_columns(self):
        self._clear_columns_layout()
        self.rename_edits.clear()

        if self.current_df is None:
            return

        header_widget = QWidget()
        header_widget.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        header_layout.setSpacing(12)

        header_left = QLabel("Original Column Name")
        header_left.setStyleSheet(f"color: {PALETTE['accent']}; font-weight: 600; font-size: 13px;")
        header_left.setFixedWidth(350)

        header_right = QLabel("New Column Name (Optional)")
        header_right.setStyleSheet(f"color: {PALETTE['accent']}; font-weight: 600; font-size: 13px;")
        header_right.setFixedWidth(350)

        header_layout.addWidget(header_left)
        header_layout.addWidget(header_right)
        header_layout.addStretch()

        self.columns_layout.addWidget(header_widget)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        divider.setMaximumWidth(750)
        self.columns_layout.addWidget(divider)

        for old_name in self.current_df.columns[1:]:
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 6, 0, 6)
            row_layout.setSpacing(12)

            raw = str(old_name)
            parts = [p.strip() for p in raw.split('/') if p.strip()]
            cleaned_parts = []
            for p in parts:
                if not cleaned_parts or cleaned_parts[-1] != p:
                    cleaned_parts.append(p)
            clean_name = " / ".join(cleaned_parts) if cleaned_parts else raw

            lbl = QLabel(clean_name)
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: {PALETTE['text']};
                    background: {PALETTE['surface']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 6px;
                    padding: 10px 12px;
                }}
            """)
            lbl.setFont(QFont("Segoe UI", 12))
            lbl.setFixedWidth(350)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            edit = QLineEdit()
            edit.setFont(QFont("Segoe UI", 12))
            edit.setPlaceholderText("Leave empty to keep original")
            edit.setFixedWidth(350)
            edit.setFixedHeight(42)
            edit.setMaxLength(80)
            edit.setStyleSheet(f"""
                QLineEdit {{
                    color:{PALETTE['text']};
                    background:{PALETTE['panel']};
                    border:1px solid {PALETTE['border']};
                    border-radius:6px;
                    padding:10px 12px;
                }}
                QLineEdit:focus {{
                    border: 1px solid {PALETTE['primary']};
                }}
            """)

            row_layout.addWidget(lbl)
            row_layout.addWidget(edit)
            row_layout.addStretch()

            self.columns_layout.addWidget(row_widget)
            self.rename_edits[old_name] = edit

        self.columns_layout.addStretch(1)

    def go_back(self):
        main_window = self.window()
        if hasattr(main_window, "stack"):
            main_window.stack.setCurrentIndex(1)

    def exit_app(self):
        QApplication.quit()

    def show_msg(self, icon: QMessageBox.Icon, title: str, text: str):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setStyleSheet(
            f"""
            QMessageBox {{
                background-color: {PALETTE['panel']};
                color: {PALETTE['text']};
                font-size: 14px;
            }}
            QMessageBox QLabel {{
                background-color: transparent;
                color: {PALETTE['text']};
            }}
            QMessageBox QPushButton {{
                background-color: {PALETTE['accent']};
                color: {PALETTE['text']};
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {PALETTE['accent_hover']};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: {PALETTE['accent_pressed']};
            }}
            """
        )
        msg.exec()
