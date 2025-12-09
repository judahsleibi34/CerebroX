from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QComboBox, QCheckBox, QHBoxLayout, QScrollArea, QFrame,
    QTabWidget, QLineEdit, QMessageBox, QStyledItemDelegate
)
from pathlib import Path
import pandas as pd

import csv
import os, sys, traceback
import json
from pathlib import Path

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
        
        self.db = CerebroXDB()     
        self.LOG_FILE = os.path.join(self.get_app_dir(), "binary_mapping_log.txt")

        try:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write("=== Logging Started ===\n")
        except:
            print("ERROR: cannot create log file")

        self.loaded_filename = None


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
                padding: 4px 6px;
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
                padding: 4px 6px;
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
        dup_layout.addWidget(dup_label)

        self.dup_combo = QComboBox()
        self.dup_combo.addItems(["Don't drop", "Keep first occurrence", "Keep last occurrence"])
        self.dup_combo.setFixedWidth(240)
        self.dup_combo.setStyleSheet(f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 4px 6px;
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

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedSize(140, 38)
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['panel']};
                color:{PALETTE['text']};
                border:1px solid {PALETTE['border']};
                border-radius:6px;
                padding:8px 16px;
                font-size:14px;
                font-weight:600;
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
        self.apply_btn.clicked.connect(self.apply_only)
        button_row.addWidget(self.apply_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedSize(140, 38)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet(
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
        self.save_btn.clicked.connect(self.save_to_database)
        button_row.addWidget(self.save_btn)


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

    def log(self, msg):
        try:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print("LOGGER ERROR:", e)

    def excepthook(type, value, tb):
        print("\n🔥🔥 UNCAUGHT ERROR IN QT CALLBACK 🔥🔥")
        print(value)
        traceback.print_tb(tb)

    sys.excepthook = excepthook

    def set_loaded_filename(self, filename: str):
        self.loaded_filename = filename

    def refresh_ui(self):

        self.show_columns()
        self.populate_drop_columns()
        self._populate_value_col_combo()

        current_col = self.value_col_combo.currentData()
        if current_col:
            self.show_values_for_column(current_col)

        self.repaint()
        self.update()

    def split_and_rename(self, value, rename_map, sep="،"):
        if pd.isna(value):
            return value

        parts = [p.strip() for p in str(value).split(sep)]

        renamed = [
            rename_map.get(part, part)
            for part in parts
        ]

        return f"{sep} ".join(renamed)

    def save_cleaned_csv(self):
        if not self.loaded_filename:
            return

        base_path = Path(self.loaded_filename).parent
        base_name = Path(self.loaded_filename).stem
        out_folder = base_path / f"{base_name}_cleaned"
        out_folder.mkdir(exist_ok=True)

        if self.merged_df is not None:
                merged_path = out_folder / f"{base_name}_merged.csv"
                self.merged_df.to_csv(
                    merged_path,
                    index=False,
                    encoding="utf-8-sig",
                    quoting=csv.QUOTE_MINIMAL,
                )

        for sheet, df in self.sheets_dfs.items():
                safe = "".join(c for c in sheet if c.isalnum() or c in ("_", "-"))
                sheet_path = out_folder / f"{base_name}_{safe}.csv"
                df.to_csv(
                    sheet_path,
                    index=False,
                    encoding="utf-8-sig",
                    quoting=csv.QUOTE_MINIMAL,
                )


        return str(out_folder)

    
    def _split_respecting_parentheses(self, text):
        if pd.isna(text):
            return []
        
        text = str(text)
        parts = []
        current = []
        paren_depth = 0
        
        for char in text:
            if char == '(':
                paren_depth += 1
                current.append(char)

            elif char == ')':
                paren_depth -= 1
                current.append(char)

            elif char in (',', '،') and paren_depth == 0:
                part = ''.join(current).strip()
                if part:
                    parts.append(part)
                current = []

            else:
                current.append(char)
        
        part = ''.join(current).strip()
        if part:
            parts.append(part)
        
        return parts

    def _rename_multi_values(self, value, rename_map):
        if pd.isna(value):
            return value
        
        if '،' in str(value):
            separator = '،'
        else:
            separator = ','
        
        parts = self._split_respecting_parentheses(value)
        renamed_parts = [rename_map.get(part, part) for part in parts]
        
        if separator == '،':
            return '، '.join(renamed_parts)
        else:
            return ', '.join(renamed_parts)

    def apply_only(self):
        selected_sheet = self.sheet_combo.currentText()

        if selected_sheet == "All sheets (merged)":
            df = self.merged_df
        else:
            df = self.sheets_dfs.get(selected_sheet)

        if df is None:
            self.show_msg(QMessageBox.Icon.Warning, "No Data", "No dataframe loaded.")
            return
        
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
        df.drop(columns=to_drop, inplace=True, errors="ignore")
        
        rename_map = {}
        value_rename_map = {}
        
        for col, edit_or_dict in self.rename_edits.items():
            if isinstance(edit_or_dict, dict):
                value_rename_map[col] = {}

                for old_val, edit in edit_or_dict.items():
                    new_val = edit.text().strip()

                    if new_val and new_val != old_val:
                        value_rename_map[col][old_val] = new_val
            else:
                new = edit_or_dict.text().strip()

                if new and new != col:
                    rename_map[col] = new

        if rename_map:
            for sheet, sdf in self.sheets_dfs.items():
                sdf.rename(columns=rename_map, inplace=True)
                
            if self.merged_df is not None:
                self.merged_df.rename(columns=rename_map, inplace=True)

        for col, val_map in value_rename_map.items():
            if not val_map:
                continue
            
            actual_col = rename_map.get(col, col)
            
            for sheet, sdf in self.sheets_dfs.items():
                if actual_col in sdf.columns:
                    sdf[actual_col] = sdf[actual_col].apply(
                    lambda x: self._apply_value_map(x, val_map)
                )
            
            if self.merged_df is not None and actual_col in self.merged_df.columns:
                self.merged_df[actual_col] = self.merged_df[actual_col].apply(
                    lambda x: self._apply_value_map(x, val_map)
                )

        for col, edits in self.value_edits.items():
            replace_map = {}
            
            for old_val, edit in edits.items():
                new_val = edit.text().strip()
                if new_val and new_val != old_val:
                    replace_map[old_val] = new_val

            if replace_map:
                actual_col = rename_map.get(col, col)
                
                for sheet, sdf in self.sheets_dfs.items():
                    if actual_col in sdf.columns:
                        sdf[actual_col] = sdf[actual_col].apply(
                            lambda x: self._apply_value_map(x, replace_map)
                        )

                if self.merged_df is not None and actual_col in self.merged_df.columns:
                    self.merged_df[actual_col] = self.merged_df[actual_col].apply(
                        lambda x: self._apply_value_map(x, replace_map)
                    )


        self.apply_one_hot_encoding()

        self.populate_drop_columns()
        self.show_msg(QMessageBox.Icon.Information, "Applied", "All changes applied successfully.")
        self.refresh_ui()

    def go_to_plot_frame(self):
        main_window = self.window()
        main_window.cleaning_window = self  

        main_window.plotting.set_dataframes(
            self.merged_df,
            self.sheets_dfs,
            self.processed_path
        )

        main_window.stack.setCurrentWidget(main_window.plotting)

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
                summary = f"Sheet: {selected_sheet}\n"
                summary += "\n".join(actions_performed)
                summary += f"\nTotal rows removed: {total_removed}"
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
            f"Dropped {len(to_drop)} column(s) from sheet '{selected_sheet}':\n" + "\n".join(to_drop)
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

                if len(parts) >= 2 and parts[-1] in ("Ø§Ù„Ø¹Ù„Ø§Ù…Ø§Øª", "Ø§Ù„Ø¹Ù€Ù„Ø§Ù…Ø§Øª"):
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
        if df is None:
            return
        df.columns = [Data_analysis._normalize_header(c) for c in df.columns]

    def _normalize_values_inplace(self, df: pd.DataFrame):
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
        if self.loaded_filename is None:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "No File",
                "No filename assigned. Make sure loading page calls set_loaded_filename()."
            )
            return

        selected_sheet = self.sheet_combo.currentText()

        if selected_sheet == "All sheets (merged)" or selected_sheet is None:
            df = self.merged_df
        else:
            df = self.sheets_dfs.get(selected_sheet)

        if df is None:
            self.show_msg(QMessageBox.Icon.Warning, "No data", "No dataframe loaded.")
            return

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

        rename_map = {}
        for old, edit in self.rename_edits.items():
            new = edit.text().strip()
            if new and new != old:
                rename_map[old] = new

        if rename_map:
            for sheet_name, sdf in self.sheets_dfs.items():
                sdf.rename(columns=rename_map, inplace=True)
            self.merged_df.rename(columns=rename_map, inplace=True)

        for col, edits in self.value_edits.items():
            replace_map = {}
            for old_val, edit in edits.items():
                new_val = edit.text().strip()
                if new_val and new_val != old_val:
                    replace_map[old_val] = new_val

            if replace_map:
                for sheet_name, sdf in self.sheets_dfs.items():
                    if col in sdf.columns:
                        sdf[col] = sdf[col].replace(replace_map)
                if col in self.merged_df.columns:
                    self.merged_df[col] = self.merged_df[col].replace(replace_map)

        sheets_final = {}

        if len(self.sheets_dfs) > 1:
            merged_final = self.merged_df
            sheets_final = {name: df for name, df in self.sheets_dfs.items()}
        else:
            merged_final = None
            sheets_final = self.sheets_dfs.copy()

        try:
            self.apply_one_hot_encoding()
            csv_folder = self.save_cleaned_csv()

            version, tables = self.db.save_run(
                filename=self.loaded_filename,
                merged_df=merged_final,
                sheets=sheets_final
            )

            table_list_str = "\n".join(tables)

            self.show_msg(
                QMessageBox.Icon.Information,
                "Saved Successfully",
                f"Version: v{version}\n\nSaved tables:\n{table_list_str}"
            )

        except Exception as e:
            self.show_msg(
                QMessageBox.Icon.Critical,
                "Database Error",
                f"Failed to save dataset:\n{e}"
            )
            return

        self.refresh_ui()
        self.go_to_plot_frame()


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
        
        all_individual_values = set()
        
        for val in series.dropna().unique():
            if pd.isna(val):
                continue
            
            parts = self._split_respecting_parentheses(val)
            
            for part in parts:
                if part.strip():
                    all_individual_values.add(part.strip())
        
        info_label = QLabel(
            f"Type: {series.dtype} | Missing: {series.isna().sum()} | "
            f"Unique cells: {series.nunique()} | Individual values: {len(all_individual_values)}"
        )
        info_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        self.values_inner_layout.addWidget(info_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        self.values_inner_layout.addWidget(divider)

        self.values_inner_layout.setAlignment(Qt.AlignTop)
        self.values_inner_layout.setSpacing(6)

        for part in sorted(all_individual_values):

            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setAlignment(Qt.AlignVCenter)
            hl.setContentsMargins(0, 2, 0, 2)
            hl.setSpacing(12)

            lbl = QLabel(part)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(160, 32)
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: {PALETTE['text']};
                    background: {PALETTE['surface']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 6px;
                    font-size: 13px;
                }}
            """)

            edit = QLineEdit()
            edit.setPlaceholderText("New value...")
            edit.setFixedHeight(32)
            edit.setStyleSheet(f"""
                QLineEdit {{
                    color:{PALETTE['text']};
                    background:{PALETTE['panel']};
                    border:1px solid {PALETTE['border']};
                    border-radius:6px;
                    padding:4px 8px;
                    font-size: 13px;
                }}
            """)

            hl.addWidget(lbl)
            hl.addWidget(edit, stretch=1)

            self.values_inner_layout.addWidget(row)

            self.value_edits[col_name][part] = edit

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

            raw = str(old_name)
            parts = [p.strip() for p in raw.split('/') if p.strip()]
            clean_name = " / ".join(parts) if parts else raw

            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 6, 0, 6)
            row_layout.setSpacing(12)

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

    


    def get_app_dir(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))   


    def _apply_value_map(self, value, value_map):
        if pd.isna(value):
            return value

        parts = self._split_respecting_parentheses(value)

        new_parts = [value_map.get(p, p) for p in parts]

        if "،" in str(value):
            return "، ".join(new_parts)
        else:
            return ", ".join(new_parts)

    def apply_one_hot_encoding(self):
        if self.merged_df is None:
            return

        df = self.merged_df
        encoding_map = {}  
        multi_cols = [
            col for col in df.columns
            if df[col].dtype == "object"
            and df[col].dropna().astype(str).str.contains(r"(،|,)\s*").any()
        ]

        if not multi_cols:
            self.log("No multi-value columns found for encoding")
            return

        self.log(f"Found {len(multi_cols)} multi-value columns to encode")

        for col in multi_cols:
            unique_parts = []
            seen = set()

            for val in df[col].dropna():
                for part in self._split_respecting_parentheses(val):
                    part = part.strip()
                    if part and part not in seen:
                        seen.add(part)
                        unique_parts.append(part)

            n_bits = len(unique_parts)
            if n_bits == 0:
                continue

            value_to_index = {v: i for i, v in enumerate(unique_parts)}
            
            encoding_map[col] = {
                'bit_length': n_bits,
                'mapping': {str(i): v for v, i in value_to_index.items()}
            }

            def encode_cell(cell):
                if pd.isna(cell):
                    bit_string = "0" * n_bits
                    return "'" + bit_string

                bits = ["0"] * n_bits
                parts = self._split_respecting_parentheses(cell)

                for p in parts:
                    p = p.strip()
                    if p in value_to_index:
                        bits[value_to_index[p]] = "1"

                bit_string = "".join(bits).zfill(n_bits)
                return "'" + bit_string

            df[col] = df[col].apply(encode_cell)
            
            self.log(f"✓ Encoded column '{col}' with {n_bits} bits")

        self.merged_df = df
        
        if encoding_map:
            self._save_encoding_map(encoding_map)

        return encoding_map


    def apply_one_hot_encoding_to_sheets(self):
        if not self.sheets_dfs:
            return {}
        
        all_encoding_maps = {}
        
        for sheet_name, df in self.sheets_dfs.items():
            multi_cols = [
                col for col in df.columns
                if df[col].dtype == "object"
                and df[col].dropna().astype(str).str.contains(r"(،|,)\s*").any()
            ]
            
            if not multi_cols:
                continue
            
            sheet_encoding_map = {}
            
            for col in multi_cols:
                unique_parts = []
                seen = set()
                
                for val in df[col].dropna():
                    for part in self._split_respecting_parentheses(val):
                        part = part.strip()
                        if part and part not in seen:
                            seen.add(part)
                            unique_parts.append(part)
                
                n_bits = len(unique_parts)
                if n_bits == 0:
                    continue
                
                value_to_index = {v: i for i, v in enumerate(unique_parts)}
                
                sheet_encoding_map[col] = {
                    'bit_length': n_bits,
                    'mapping': {str(i): v for v, i in value_to_index.items()}
                }
                
                def encode_cell(cell):
                    if pd.isna(cell):
                        return "'" + ("0" * n_bits)
                    
                    bits = ["0"] * n_bits
                    parts = self._split_respecting_parentheses(cell)
                    
                    for p in parts:
                        p = p.strip()
                        if p in value_to_index:
                            bits[value_to_index[p]] = "1"
                    
                    return "'" + "".join(bits).zfill(n_bits)
                
                df[col] = df[col].apply(encode_cell)
                
                self.log(f"✓ Sheet '{sheet_name}': Encoded '{col}' with {n_bits} bits")
            
            if sheet_encoding_map:
                all_encoding_maps[sheet_name] = sheet_encoding_map
        
        return all_encoding_maps


    def _save_encoding_map(self, encoding_map, sheet_maps=None):
        if not self.loaded_filename:
            return
        
        base_path = Path(self.loaded_filename).parent
        base_name = Path(self.loaded_filename).stem
        out_folder = base_path / f"{base_name}_cleaned"
        out_folder.mkdir(exist_ok=True)
        
        mapping_doc = {
            'filename': self.loaded_filename,
            'encoding_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'merged_sheet': encoding_map if encoding_map else {},
            'individual_sheets': sheet_maps if sheet_maps else {}
        }
        
        json_path = out_folder / f"{base_name}_binary_encoding_map.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_doc, f, ensure_ascii=False, indent=2)
        
        txt_path = out_folder / f"{base_name}_binary_encoding_map.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BINARY ENCODING MAPPING DOCUMENTATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.loaded_filename}\n")
            f.write(f"Encoding Date: {mapping_doc['encoding_date']}\n\n")
            
            if encoding_map:
                f.write("-" * 80 + "\n")
                f.write("MERGED SHEET ENCODINGS\n")
                f.write("-" * 80 + "\n\n")
                
                for col, info in encoding_map.items():
                    f.write(f"Column: {col}\n")
                    f.write(f"Bit Length: {info['bit_length']}\n")
                    f.write(f"Bit Position Mapping:\n")
                    for pos, value in sorted(info['mapping'].items(), key=lambda x: int(x[0])):
                        f.write(f"  Bit {pos}: {value}\n")
                    f.write("\n")
            
            if sheet_maps:
                for sheet_name, sheet_map in sheet_maps.items():
                    f.write("-" * 80 + "\n")
                    f.write(f"SHEET: {sheet_name}\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for col, info in sheet_map.items():
                        f.write(f"Column: {col}\n")
                        f.write(f"Bit Length: {info['bit_length']}\n")
                        f.write(f"Bit Position Mapping:\n")
                        for pos, value in sorted(info['mapping'].items(), key=lambda x: int(x[0])):
                            f.write(f"  Bit {pos}: {value}\n")
                        f.write("\n")
        
        self.log(f"✓ Saved encoding map to: {json_path}")
        self.log(f"✓ Saved readable map to: {txt_path}")
        
        print(f"\n📋 Binary encoding map saved to:\n   {json_path}\n   {txt_path}\n")


    def save_to_database(self):
        """Updated save_to_database with proper encoding map generation"""
        if self.loaded_filename is None:
            self.show_msg(QMessageBox.Icon.Warning, "No File", "No filename assigned.")
            return

        if self.merged_df is None and not self.sheets_dfs:
            self.show_msg(QMessageBox.Icon.Warning, "Empty", "Nothing to save.")
            return

        try:
            merged_map = self.apply_one_hot_encoding()   
            sheet_maps = self.apply_one_hot_encoding_to_sheets()   
            
            self._save_encoding_map(merged_map, sheet_maps)
            
            csv_folder = self.save_cleaned_csv()
            self.processed_path = csv_folder 
            
            self.log("Calling DB save...")
            
            version, tables = self.db.save_run(
                filename=self.loaded_filename,
                merged_df=self.merged_df,
                sheets=self.sheets_dfs
            )
            
            self.log("DB save success.")
            
            table_list_str = "\n".join(tables)
            
            self.show_msg(
                QMessageBox.Icon.Information,
                "Saved Successfully",
                f"CSV files saved to:\n{csv_folder}\n\n"
                f"Binary encoding map saved as:\n"
                f"  - {Path(self.loaded_filename).stem}_binary_encoding_map.json\n"
                f"  - {Path(self.loaded_filename).stem}_binary_encoding_map.txt\n\n"
                f"Database Version: v{version}\n\n"
                f"Saved tables:\n{table_list_str}"
            )
            
            self.go_to_plot_frame()
            
        except Exception as e:
            self.log(f"Save Error: {e}")
            self.show_msg(
                QMessageBox.Icon.Critical,
                "Save Error",
                f"Failed to save dataset:\n{e}"
            )