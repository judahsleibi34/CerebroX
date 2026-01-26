import subprocess

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
import os, sys, traceback, subprocess
import json, re
import datetime

import numpy as np

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

        # =========================
        # Encoding Tab (NEW)
        # =========================
        self.encoding_tab = QWidget()
        encoding_layout = QVBoxLayout(self.encoding_tab)
        encoding_layout.setContentsMargins(16, 16, 16, 16)
        encoding_layout.setSpacing(12)

        enc_title = QLabel("Encoding Selection")
        enc_title.setStyleSheet(f"color: {PALETTE['text']};")
        enc_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        encoding_layout.addWidget(enc_title)

        # Search filter (optional but useful)
        self.encoding_search = QLineEdit()
        self.encoding_search.setPlaceholderText("Search columns...")
        self.encoding_search.setStyleSheet(f"""
            QLineEdit {{
                color:{PALETTE['text']};
                background:{PALETTE['panel']};
                border:1px solid {PALETTE['border']};
                border-radius:6px;
                padding:6px 10px;
                font-size: 13px;
            }}
        """)
        self.encoding_search.textChanged.connect(self.populate_encoding_columns)
        encoding_layout.addWidget(self.encoding_search)

        # Select all/none row
        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(10)

        self.encoding_select_all = QCheckBox("Select / Deselect All")
        self.encoding_select_all.setStyleSheet(f"""
            QCheckBox {{
                color: {PALETTE['accent']};
                spacing: 8px;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 0px;
            }}
        """)
        self.encoding_select_all.stateChanged.connect(self._toggle_all_encoding_columns)
        row_l.addWidget(self.encoding_select_all)
        row_l.addStretch(1)

        encoding_layout.addWidget(row)

        # Scroll list of checkboxes
        self.encoding_scroll = QScrollArea()
        self.encoding_scroll.setWidgetResizable(True)
        self.encoding_scroll.setFrameShape(QFrame.NoFrame)
        self.encoding_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.encoding_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.encoding_scroll.setStyleSheet(
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

        self.encoding_inner = QWidget()
        self.encoding_inner.setStyleSheet("background: transparent;")
        self.encoding_inner_layout = QVBoxLayout(self.encoding_inner)
        self.encoding_inner_layout.setContentsMargins(0, 8, 0, 8)
        self.encoding_inner_layout.setSpacing(8)
        self.encoding_inner_layout.setAlignment(Qt.AlignTop)

        self.encoding_scroll.setWidget(self.encoding_inner)
        encoding_layout.addWidget(self.encoding_scroll, stretch=1)

        # Encoding mode selector row
        mode_row = QWidget()
        mode_l = QHBoxLayout(mode_row)
        mode_l.setContentsMargins(0, 0, 0, 0)
        mode_l.setSpacing(10)

        mode_lbl = QLabel("Encoding Mode:")
        mode_lbl.setStyleSheet(f"color:{PALETTE['text']}; font-size: 13px;")
        mode_l.addWidget(mode_lbl)

        self.encoding_mode_combo = QComboBox()
        self.encoding_mode_combo.addItems([
            "AUTO",
            "FORCE_LABEL",
            "FORCE_ONEHOT",
            "FORCE_MULTIHOT",
            "FORCE_BITMASK"
        ])
        self.encoding_mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                padding: 4px 6px;
                min-width: 220px;
            }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                selection-background-color: {PALETTE['primary']};
            }}
        """)
        mode_l.addWidget(self.encoding_mode_combo)
        mode_l.addStretch(1)

        encoding_layout.addWidget(mode_row)

        # Add tab to widget
        self.tabs.addTab(self.encoding_tab, "Encoding")

        # Storage for encoding checkboxes
        self.encoding_checkboxes = {}   # {real_col_name: QCheckBox}


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

        self.apply_btn.clicked.connect(self.apply_main_action)
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
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {msg}\n"
            
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            print(log_entry.strip())
        except Exception as e:
            print(f"⚠️ LOGGER ERROR: {e}")
            print(f"   Attempted to log: {msg}")

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

        # --- Quality actions ---
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

        # --- Drop selected columns ---
        to_drop = [col for col, cb in self.drop_checkboxes.items() if cb.isChecked()]
        df.drop(columns=to_drop, inplace=True, errors="ignore")

        # --- Collect rename maps (column rename + value rename) ---
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

        # --- Apply column rename to all dataframes ---
        if rename_map:
            for sheet, sdf in self.sheets_dfs.items():
                sdf.rename(columns=rename_map, inplace=True)
            if self.merged_df is not None:
                self.merged_df.rename(columns=rename_map, inplace=True)

        # --- Apply value rename_map for multi-value cells ---
        for col, val_map in value_rename_map.items():
            if not val_map:
                continue

            actual_col = rename_map.get(col, col)

            for sheet, sdf in self.sheets_dfs.items():
                if actual_col in sdf.columns:
                    sdf[actual_col] = sdf[actual_col].apply(lambda x: self._apply_value_map(x, val_map))

            if self.merged_df is not None and actual_col in self.merged_df.columns:
                self.merged_df[actual_col] = self.merged_df[actual_col].apply(lambda x: self._apply_value_map(x, val_map))

        # --- Apply value edits panel replacements ---
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
                        sdf[actual_col] = sdf[actual_col].apply(lambda x: self._apply_value_map(x, replace_map))

                if self.merged_df is not None and actual_col in self.merged_df.columns:
                    self.merged_df[actual_col] = self.merged_df[actual_col].apply(lambda x: self._apply_value_map(x, replace_map))

        # IMPORTANT: do NOT multi-hot encode here (keep dataset readable for UI)
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
        current_widget = self.tabs.widget(index)

        if current_widget is self.cleaning_tab:
            self.populate_drop_columns()

        elif hasattr(self, "encoding_tab") and current_widget is self.encoding_tab:
            self.populate_encoding_columns()


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

        # --- Quality actions ---
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

        # --- Drop columns ---
        to_drop = [col for col, cb in self.drop_checkboxes.items() if cb.isChecked()]
        if to_drop:
            df.drop(columns=to_drop, inplace=True, errors="ignore")

        # --- Rename columns ---
        rename_map = {}
        for old, edit in self.rename_edits.items():
            new = edit.text().strip()
            if new and new != old:
                rename_map[old] = new

        if rename_map:
            for sheet_name, sdf in self.sheets_dfs.items():
                sdf.rename(columns=rename_map, inplace=True)
            if self.merged_df is not None:
                self.merged_df.rename(columns=rename_map, inplace=True)

        # --- Value mapping panel replacements ---
        for col, edits in self.value_edits.items():
            replace_map = {}
            for old_val, edit in edits.items():
                new_val = edit.text().strip()
                if new_val and new_val != old_val:
                    replace_map[old_val] = new_val

            if replace_map:
                for sheet_name, sdf in self.sheets_dfs.items():
                    if col in sdf.columns:
                        sdf[col] = sdf[col].apply(lambda x: self._apply_value_map(x, replace_map))
                if self.merged_df is not None and col in self.merged_df.columns:
                    self.merged_df[col] = self.merged_df[col].apply(lambda x: self._apply_value_map(x, replace_map))

        try:
            # # --- Multi-hot encode (this changes self.merged_df in-place) ---
            # merged_map = self.apply_multi_hot_encoding(drop_original=True)
            # self._save_multi_hot_map(merged_map)

            # Save CSVs AFTER encoding
            csv_folder = self.save_cleaned_csv()
            self.processed_path = csv_folder

            # Save to DB using UPDATED dfs
            version, tables = self.db.save_run(
                filename=self.loaded_filename,
                merged_df=self.merged_df,
                sheets=self.sheets_dfs
            )

            table_list_str = "\n".join(tables)

            self.show_msg(
                QMessageBox.Icon.Information,
                "Saved Successfully",
                f"CSV files saved to:\n{csv_folder}\n\n"
                f"Multi-hot map saved as:\n"
                f"  - {Path(self.loaded_filename).stem}_multi_hot_map.json\n\n"
                f"Database Version: v{version}\n\n"
                f"Saved tables:\n{table_list_str}"
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
        
        if hasattr(self, "encoding_tab") and self.tabs.currentWidget() is self.encoding_tab:
            self.populate_encoding_columns()


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
        self.log("=" * 60)
        self.log("STARTING ONE-HOT ENCODING FOR MERGED DATAFRAME")
        self.log("=" * 60)
        
        if self.merged_df is None:
            self.log("⚠️ No merged_df to encode - SKIPPING")
            return {}

        df = self.merged_df.copy()   
        encoding_map = {}   
        
        self.log(f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")

        multi_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                has_separator = df[col].dropna().astype(str).str.contains(r"(،|,)\s*").any()
                if has_separator:
                    multi_cols.append(col)
                    self.log(f"  ✓ Found multi-value column: {col}")

        if not multi_cols:
            self.log("⚠️ No multi-value columns found for encoding")
            return {}

        self.log(f"\n📊 Total multi-value columns to encode: {len(multi_cols)}\n")

        for col in multi_cols:
            self.log(f"Processing column: '{col}'")
            
            unique_parts = []
            seen = set()

            cell_count = 0
            for val in df[col].dropna():
                cell_count += 1
                parts = self._split_respecting_parentheses(val)
                for part in parts:
                    part = part.strip()
                    if part and part not in seen:
                        seen.add(part)
                        unique_parts.append(part)

            n_bits = len(unique_parts)
            if n_bits == 0:
                self.log(f"  ⚠️ No unique values found in '{col}' - SKIPPING")
                continue

            self.log(f"  • Processed {cell_count} non-null cells")
            self.log(f"  • Found {n_bits} unique individual values")

            value_to_index = {v: i for i, v in enumerate(unique_parts)}
            
            encoding_map[col] = {
                'bit_length': n_bits,
                'mapping': {str(i): v for v, i in value_to_index.items()}
            }
            
            self.log(f"  • Bit position mapping:")
            for v, i in sorted(value_to_index.items(), key=lambda x: x[1]):
                self.log(f"    Bit {i}: {v}")

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

            old_sample = df[col].iloc[0] if len(df) > 0 else None
            df[col] = df[col].apply(encode_cell)
            new_sample = df[col].iloc[0] if len(df) > 0 else None
            
            self.log(f"  ✓ Encoded '{col}' with {n_bits} bits")
            self.log(f"    Example: '{old_sample}' -> '{new_sample}'")
            self.log("")

        self.merged_df = df
        
        self.log(f"✅ ONE-HOT ENCODING COMPLETE FOR MERGED DATAFRAME")
        self.log(f"   Encoded {len(encoding_map)} columns")
        self.log("=" * 60 + "\n")

        return encoding_map

    def _save_encoding_map(self, encoding_map, sheet_maps=None):
        self.log("=" * 60)
        self.log("SAVING ENCODING MAPS")
        self.log("=" * 60)
        
        if not self.loaded_filename:
            self.log("⚠️ No filename available - cannot save mapping files")
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
        
        self.log(f"✓ Saved JSON map to: {json_path}")
        
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
        
        self.log(f"✓ Saved TXT map to: {txt_path}")
        self.log("=" * 60 + "\n")
        
        print(f"\n📋 Binary encoding map saved to:\n   📄 {json_path}\n   📄 {txt_path}\n")

    def save_to_database(self):
        self.log("\n" + "=" * 80)
        self.log("STARTING SAVE TO DATABASE PROCESS")
        self.log("=" * 80)

        if self.loaded_filename is None:
            self.show_msg(QMessageBox.Icon.Warning, "No File", "No filename assigned.")
            return

        if self.merged_df is None and not self.sheets_dfs:
            self.show_msg(QMessageBox.Icon.Warning, "Empty", "Nothing to save.")
            return

        try:
            # Save CSVs first
            csv_folder = self.save_cleaned_csv()
            self.processed_path = csv_folder

            # DB save
            version, tables = self.db.save_run(
                filename=self.loaded_filename,
                merged_df=self.merged_df,
                sheets=self.sheets_dfs
            )

            try:
                self.run_backup_and_push()
            except Exception as e:
                self.show_msg(QMessageBox.Icon.Warning,
                            "Saved to DB but GitHub push failed",
                            str(e))

            table_list_str = "\n".join(tables)
            self.show_msg(
                QMessageBox.Icon.Information,
                "Saved Successfully",
                f"CSV files saved to:\n{csv_folder}\n\n"
                f"Database Version: v{version}\n\n"
                f"Saved tables:\n{table_list_str}"
            )

            self.go_to_plot_frame()

        except Exception as e:
            self.log(f"❌ DB SAVE ERROR: {e}")
            self.log(traceback.format_exc())

            msg = str(e).lower()

            # Friendly messages for typical SQL problems
            if "sql" in msg and ("syntax" in msg or "near" in msg):
                friendly = (
                    "Database save failed because some column names are not SQL-safe.\n\n"
                    "Fix: Use the Encoding tab (it now makes SQL-safe column names), then save again."
                )
            elif "too many columns" in msg:
                friendly = (
                    "Database save failed because the table has too many columns.\n\n"
                    "Fix: Encode fewer columns at a time, or use BITMASK mode for multi-value columns."
                )
            elif "duplicate column" in msg or "already exists" in msg:
                friendly = (
                    "Database save failed due to duplicate column names.\n\n"
                    "Fix: Re-encode (the new encoder prevents collisions), then save again."
                )
            else:
                friendly = "Database save failed due to an unexpected error."

            self.show_msg(
                QMessageBox.Icon.Critical,
                "Save Error",
                f"{friendly}\n\nTechnical details:\n{e}"
            )

    def run_backup_and_push(self):
        base_dir = Path(self.get_app_dir()).resolve()
        repo_dir = base_dir / "DB_Cerebrox"
        bat_path = repo_dir / "run_backup.bat"

        if not bat_path.exists():
            raise FileNotFoundError(f"run_backup.bat not found at:\n{bat_path}")

        p = subprocess.run(
            ["cmd.exe", "/c", str(bat_path)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True
        )

        if p.returncode != 0:
            raise RuntimeError(p.stderr or p.stdout or f"Backup failed with code {p.returncode}")



    def apply_one_hot_encoding_to_sheets(self):
        self.log("=" * 60)
        self.log("STARTING ONE-HOT ENCODING FOR INDIVIDUAL SHEETS")
        self.log("=" * 60)
        
        if not self.sheets_dfs:
            self.log("⚠️ No individual sheets to encode - SKIPPING")
            return {}
        
        all_encoding_maps = {}
        
        for sheet_name, df in self.sheets_dfs.items():
            self.log(f"\n📄 Processing sheet: '{sheet_name}'")
            self.log(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            df = df.copy()   
            
            multi_cols = []
            for col in df.columns:
                if df[col].dtype == "object":
                    has_separator = df[col].dropna().astype(str).str.contains(r"(،|,)\s*").any()
                    if has_separator:
                        multi_cols.append(col)
            
            if not multi_cols:
                self.log(f"   ⚠️ No multi-value columns in sheet '{sheet_name}'")
                continue
            
            self.log(f"   Found {len(multi_cols)} multi-value columns")
            
            sheet_encoding_map = {}
            
            for col in multi_cols:
                self.log(f"\n   Processing column: '{col}'")
                
                unique_parts = []
                seen = set()
                
                cell_count = 0
                for val in df[col].dropna():
                    cell_count += 1
                    parts = self._split_respecting_parentheses(val)
                    for part in parts:
                        part = part.strip()
                        if part and part not in seen:
                            seen.add(part)
                            unique_parts.append(part)
                
                n_bits = len(unique_parts)
                if n_bits == 0:
                    self.log(f"     ⚠️ No unique values - SKIPPING")
                    continue
                
                self.log(f"     • Processed {cell_count} cells")
                self.log(f"     • Found {n_bits} unique values")
                
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
                
                old_sample = df[col].iloc[0] if len(df) > 0 else None
                df[col] = df[col].apply(encode_cell)
                new_sample = df[col].iloc[0] if len(df) > 0 else None
                
                self.log(f"     ✓ Encoded with {n_bits} bits")
                self.log(f"       Example: '{old_sample}' -> '{new_sample}'")
            
            self.sheets_dfs[sheet_name] = df
            
            if sheet_encoding_map:
                all_encoding_maps[sheet_name] = sheet_encoding_map
                self.log(f"\n   ✅ Sheet '{sheet_name}': Encoded {len(sheet_encoding_map)} columns")
        
        self.log(f"\n✅ ONE-HOT ENCODING COMPLETE FOR ALL SHEETS")
        self.log(f"   Total sheets processed: {len(all_encoding_maps)}")
        self.log("=" * 60 + "\n")
        
        return all_encoding_maps

    def apply_multi_hot_encoding(self, drop_original: bool = True, sparse: bool = True, only_columns=None):
        """
        Backward compatible: encodes self.merged_df in place (but uses helper internally).
        """
        if self.merged_df is None or self.merged_df.empty:
            self.log("⚠️ No merged_df to encode - SKIPPING")
            return {}

        df_copy = self.merged_df.copy()
        encoded_df, enc_map = self._multi_hot_encode_df(
            df_copy,
            drop_original=drop_original,
            only_columns=only_columns
        )
        self.merged_df = encoded_df
        return enc_map

    def _save_multi_hot_map(self, encoding_map: dict):
        if not self.loaded_filename or not encoding_map:
            return

        base_path = Path(self.loaded_filename).parent
        base_name = Path(self.loaded_filename).stem
        out_folder = base_path / f"{base_name}_cleaned"
        out_folder.mkdir(exist_ok=True)

        out_path = out_folder / f"{base_name}_multi_hot_map.json"
        payload = {
            "filename": self.loaded_filename,
            "encoding_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "encoding_map": encoding_map
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.log(f"✓ Saved multi-hot map to: {out_path}")


    # =========================
    # Encoding Tab Helpers (NEW)
    # =========================
    def _clear_encoding_layout(self):
        while self.encoding_inner_layout.count():
            item = self.encoding_inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.encoding_checkboxes.clear()

    def _toggle_all_encoding_columns(self, state):
        checked = (state == Qt.Checked)
        for cb in self.encoding_checkboxes.values():
            cb.setChecked(checked)

    # def populate_encoding_columns(self):
    #     """
    #     Shows ONLY columns that are:
    #     - dtype object
    #     - contain multi-value separators (، or ,)
    #     Supports search filter.
    #     """
    #     df = self.current_df if self.current_df is not None else self.merged_df
    #     self._clear_encoding_layout()

    #     if df is None or df.empty:
    #         return

    #     query = (self.encoding_search.text() if hasattr(self, "encoding_search") else "") or ""
    #     query = query.strip().lower()

    #     # Only object columns + multi-value candidates
    #     candidate_cols = []
    #     obj_cols = list(df.select_dtypes(include="object").columns)

    #     for col in obj_cols:
    #         s = df[col].dropna().astype(str)
    #         if s.empty:
    #             continue
    #         if s.str.contains(r"(،|,)\s*").any():
    #             candidate_cols.append(col)

    #     # Apply search filter
    #     for col in candidate_cols:
    #         col_str = str(col)
    #         if query and query not in col_str.lower():
    #             continue

    #         cb = QCheckBox(col_str)
    #         cb.setStyleSheet(f"""
    #             QCheckBox {{
    #                 color: {PALETTE['text']};
    #                 spacing: 8px;
    #                 font-size: 13px;
    #                 padding: 4px 0px;
    #             }}
    #             QCheckBox::indicator {{
    #                 width: 16px;
    #                 height: 16px;
    #                 border: 1px solid {PALETTE['border']};
    #                 border-radius: 3px;
    #             }}
    #             QCheckBox::indicator:hover {{
    #                 border: 1px solid {PALETTE['primary']};
    #             }}
    #             QCheckBox::indicator:checked {{
    #                 background: {PALETTE['primary']};
    #                 border: 1px solid {PALETTE['primary']};
    #             }}
    #         """)
    #         self.encoding_inner_layout.addWidget(cb)
    #         self.encoding_checkboxes[col] = cb

    #     self.encoding_inner_layout.addStretch(1)

    #     # Keep "Select all" checkbox consistent with current view
    #     if hasattr(self, "encoding_select_all"):
    #         self.encoding_select_all.blockSignals(True)
    #         self.encoding_select_all.setChecked(False)
    #         self.encoding_select_all.blockSignals(False)

    def populate_encoding_columns(self):
        """
        Shows ALL object columns (single-value + multi-value).
        - Multi-value columns are marked with [MULTI]
        Supports search filter.
        """
        df = self.current_df if self.current_df is not None else self.merged_df
        self._clear_encoding_layout()

        if df is None or df.empty:
            return

        query = (self.encoding_search.text() if hasattr(self, "encoding_search") else "") or ""
        query = query.strip().lower()

        obj_cols = list(df.select_dtypes(include="object").columns)

        for col in obj_cols:
            col_str = str(col)
            if query and query not in col_str.lower():
                continue

            is_multi = self._is_multi_value_column(df[col])
            display = f"{col_str}  [MULTI]" if is_multi else col_str

            cb = QCheckBox(display)
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
            self.encoding_inner_layout.addWidget(cb)

            # store real column name as key
            self.encoding_checkboxes[col] = cb

        self.encoding_inner_layout.addStretch(1)

        if hasattr(self, "encoding_select_all"):
            self.encoding_select_all.blockSignals(True)
            self.encoding_select_all.setChecked(False)
            self.encoding_select_all.blockSignals(False)


    # def apply_selected_encoding(self):
    #     """
    #     Applies multi-hot encoding ONLY to selected columns,
    #     and applies it to:
    #     - the currently selected sheet dataframe (if a sheet is selected)
    #     - otherwise, the merged dataframe
    #     """
    #     df = self.current_df if self.current_df is not None else self.merged_df
    #     if df is None or df.empty:
    #         self.show_msg(QMessageBox.Icon.Warning, "No Data", "No dataframe loaded.")
    #         return

    #     selected_cols = [col for col, cb in self.encoding_checkboxes.items() if cb.isChecked()]
    #     if not selected_cols:
    #         self.show_msg(QMessageBox.Icon.Warning, "No Columns Selected",
    #                     "Please select at least one column to encode.")
    #         return

    #     # Decide target: sheet or merged
    #     is_sheet = bool(self.current_sheet and self.current_sheet in self.sheets_dfs)
    #     target_name = self.current_sheet if is_sheet else "All sheets (merged)"

    #     try:
    #         # Encode on a COPY to avoid weird reference side-effects
    #         df_copy = df.copy()

    #         # Run encoding on df_copy without touching self.merged_df as a workspace
    #         encoded_df, enc_map = self._multi_hot_encode_df(
    #             df_copy,
    #             drop_original=True,
    #             only_columns=selected_cols
    #         )

    #         if not enc_map:
    #             self.show_msg(QMessageBox.Icon.Information, "No Encoding Applied",
    #                         "None of the selected columns contain multi-value data (comma/Arabic comma).")
    #             return

    #         # Save mapping (optional: per sheet naming could be improved later)
    #         self._save_multi_hot_map(enc_map)

    #         # Write back to correct place
    #         if is_sheet:
    #             self.sheets_dfs[self.current_sheet] = encoded_df
    #             self.current_df = self.sheets_dfs[self.current_sheet]
    #         else:
    #             self.merged_df = encoded_df
    #             self.current_df = self.merged_df

    #         self.show_msg(
    #             QMessageBox.Icon.Information,
    #             "Encoding Applied",
    #             f"Sheet: {target_name}\nEncoded {len(enc_map)} column(s).\nOnly selected columns were encoded."
    #         )

    #         # UI refresh + lists update
    #         self.refresh_ui()
    #         self.populate_drop_columns()
    #         self.populate_encoding_columns()

    #     except Exception as e:
    #         self.show_msg(QMessageBox.Icon.Critical, "Encoding Error", str(e))

    def apply_selected_encoding(self):
        """
        HYBRID encoding for selected columns:
        - If column contains multi-values (comma/Arabic comma): multi-hot (default) OR bitmask if user chose it
        - Else single categorical:
            * 2 unique -> binary label encoding (0/1)
            * >2 unique -> one-hot encoding
        Also does safe SQL-friendly column naming + friendly error messages.
        """
        df = self.current_df if self.current_df is not None else self.merged_df
        if df is None or df.empty:
            self.show_msg(QMessageBox.Icon.Warning, "No Data", "No dataframe loaded.")
            return

        selected_cols = [col for col, cb in self.encoding_checkboxes.items() if cb.isChecked()]
        if not selected_cols:
            self.show_msg(QMessageBox.Icon.Warning, "No Columns Selected",
                        "Please select at least one column to encode.")
            return

        # Decide target: sheet or merged
        is_sheet = bool(self.current_sheet and self.current_sheet in self.sheets_dfs)
        target_name = self.current_sheet if is_sheet else "All sheets (merged)"

        try:
            # Work on a copy to prevent weird Qt/DataFrame reference issues
            df_copy = df.copy()

            # Decide encoding mode (Auto / Force choices)
            mode = "AUTO"
            if hasattr(self, "encoding_mode_combo"):
                mode = (self.encoding_mode_combo.currentText() or "AUTO").strip().upper()

            # Apply hybrid encoding
            encoded_df, enc_map, report = self._hybrid_encode_df(
                df_copy,
                selected_cols=selected_cols,
                mode=mode
            )

            if not enc_map:
                self.show_msg(QMessageBox.Icon.Information, "No Encoding Applied",
                            "No suitable encoding was applied.\n\n"
                            "Tip: Make sure the selected columns are categorical (text) or multi-value columns.")
                return

            # Save mapping
            self._save_hybrid_encoding_map(enc_map)

            # Write back to correct place
            if is_sheet:
                self.sheets_dfs[self.current_sheet] = encoded_df
                self.current_df = self.sheets_dfs[self.current_sheet]
            else:
                self.merged_df = encoded_df
                self.current_df = self.merged_df

            # Friendly success message
            summary_lines = [
                f"Sheet: {target_name}",
                f"Encoded columns: {len(enc_map)}",
                "",
                "What happened:"
            ] + report[:30]  # avoid super long popup

            if len(report) > 30:
                summary_lines.append(f"... (+{len(report) - 30} more)")

            self.show_msg(
                QMessageBox.Icon.Information,
                "Encoding Applied",
                "\n".join(summary_lines)
            )

            # Refresh UI lists
            self.refresh_ui()
            self.populate_drop_columns()
            self.populate_encoding_columns()

        except Exception as e:
            self._show_friendly_encoding_error(e)


    # def _multi_hot_encode_df(self, df: pd.DataFrame, drop_original: bool = True, only_columns=None):
    #     """
    #     Multi-hot encode on a PROVIDED dataframe (no self.merged_df usage).
    #     Returns: (encoded_df, encoding_map)
    #     """
    #     if df is None or df.empty:
    #         return df, {}

    #     encoding_map = {}

    #     # Detect multi-value columns
    #     multi_cols = []
    #     for col in df.columns:
    #         if only_columns is not None and col not in only_columns:
    #             continue

    #         if df[col].dtype == "object":
    #             s = df[col].dropna().astype(str)
    #             if not s.empty and s.str.contains(r"(،|,)\s*").any():
    #                 multi_cols.append(col)

    #     if not multi_cols:
    #         return df, {}

    #     def safe_token_name(token: str) -> str:
    #         token = str(token).strip()
    #         token = re.sub(r"\s+", "_", token)
    #         token = re.sub(r"[^\w\u0600-\u06FF]+", "_", token)
    #         return token.strip("_")

    #     for col in multi_cols:
    #         self.log(f"Multi-hot encoding column: {col}")

    #         labels_series = df[col].apply(
    #             lambda x: [p.strip() for p in self._split_respecting_parentheses(x) if p.strip()]
    #             if pd.notna(x) else []
    #         )

    #         uniques = sorted({lab for row in labels_series for lab in row})
    #         if not uniques:
    #             self.log(f"  ⚠️ No labels found in {col} - SKIPPING")
    #             continue

    #         mat = np.zeros((len(df), len(uniques)), dtype=np.uint8)
    #         idx = {lab: i for i, lab in enumerate(uniques)}

    #         for r, row_labels in enumerate(labels_series):
    #             for lab in row_labels:
    #                 mat[r, idx[lab]] = 1

    #         new_cols = []
    #         for i, lab in enumerate(uniques):
    #             new_name = f"{col}__{safe_token_name(lab)}"
    #             df[new_name] = mat[:, i]
    #             new_cols.append(new_name)

    #         if drop_original:
    #             df.drop(columns=[col], inplace=True, errors="ignore")

    #         encoding_map[col] = {
    #             "type": "multi_hot",
    #             "new_columns": new_cols,
    #             "labels": uniques
    #         }

    #         self.log(f"  ✓ Created {len(new_cols)} columns for {col}")

    #     return df, encoding_map

    def _multi_hot_encode_df(self, df: pd.DataFrame, drop_original: bool = True, only_columns=None):
        """
        Safer multi-hot encoder:
        - SQL-safe column naming
        - guaranteed unique column names
        - keeps Arabic letters but removes SQL-breaking punctuation like / ? etc
        """
        if df is None or df.empty:
            return df, {}

        df = df.copy()
        encoding_map = {}
        used_cols = set(df.columns)

        # detect multi-value columns
        multi_cols = []
        for col in df.columns:
            if only_columns is not None and col not in only_columns:
                continue
            if self._is_multi_value_column(df[col]):
                multi_cols.append(col)

        if not multi_cols:
            return df, {}

        for col in multi_cols:
            self.log(f"Multi-hot encoding column: {col}")

            new_cols, labels, created = self._multi_hot_encode_series(df[col], col, used_cols)
            if not created:
                self.log(f"  ⚠️ No labels found in {col} - SKIPPING")
                continue

            for nc, values in new_cols.items():
                df[nc] = values

            if drop_original:
                df.drop(columns=[col], inplace=True, errors="ignore")

            encoding_map[str(col)] = {
                "type": "multi_hot",
                "new_columns": created,
                "labels": labels
            }

            self.log(f"  ✓ Created {len(created)} columns for {col}")

        return df, encoding_map

    def _show_friendly_encoding_error(self, e: Exception):
        msg = str(e)

        # Common pandas / encoding issues
        if "cannot reindex" in msg.lower():
            nice = "Encoding failed because the dataframe index/columns got out of sync."
        elif "duplicate" in msg.lower() and "columns" in msg.lower():
            nice = "Encoding failed because it tried to create duplicate column names."
        else:
            nice = "Encoding failed due to an unexpected error."

        self.log(f"❌ ENCODING ERROR: {e}")
        self.log(traceback.format_exc())

        self.show_msg(
            QMessageBox.Icon.Critical,
            "Encoding Error",
            f"{nice}\n\nTechnical details:\n{e}"
        )


    def _is_multi_value_column(self, s: pd.Series) -> bool:
        """True if series contains multi-value separator in any non-null cell."""
        if s is None:
            return False
        if s.dtype != "object":
            return False
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        return ss.str.contains(r"(،|,)\s*").any()


    def _safe_sql_identifier(self, name: str) -> str:
        """
        Make column names DB-safe:
        - replace spaces with _
        - remove punctuation like / ? etc
        - keep Arabic letters and digits and underscore
        - ensure not empty and not starting with digit
        """
        name = str(name).strip()
        name = re.sub(r"\s+", "_", name)

        # keep: letters/digits/_ and Arabic range
        name = re.sub(r"[^\w\u0600-\u06FF]+", "_", name)
        name = name.strip("_")

        if not name:
            name = "col"

        if re.match(r"^\d", name):
            name = f"c_{name}"

        # optional: limit length for DB friendliness
        if len(name) > 80:
            name = name[:80]

        return name


    def _make_unique_name(self, base: str, used: set) -> str:
        """Ensure no collisions."""
        if base not in used:
            used.add(base)
            return base
        k = 2
        while True:
            candidate = f"{base}__{k}"
            if candidate not in used:
                used.add(candidate)
                return candidate
            k += 1


    def _binary_label_encode_series(self, s: pd.Series):
        """
        Binary encode 2 unique (non-null) values into 0/1.
        Returns (encoded_series, mapping_dict)
        """
        vals = [v for v in s.dropna().unique()]
        # keep stable order
        vals = list(map(lambda x: str(x), vals))
        vals = list(dict.fromkeys(vals))

        mapping = {vals[0]: 0, vals[1]: 1} if len(vals) == 2 else {}
        def enc(x):
            if pd.isna(x):
                return np.nan
            key = str(x)
            return mapping.get(key, np.nan)

        out = s.map(enc).astype("float")  # keep NaN allowed
        return out, mapping


    def _one_hot_encode_series(self, s: pd.Series, base_col: str, used_cols: set):
        """
        One-hot encode single-category column.
        Returns (new_df_cols_dict, labels)
        """
        ss = s.fillna("__MISSING__").astype(str)
        labels = sorted(ss.unique())

        new_cols = {}
        for lab in labels:
            safe_base = self._safe_sql_identifier(base_col)
            safe_lab = self._safe_sql_identifier(lab)
            col_name = self._make_unique_name(f"{safe_base}__{safe_lab}", used_cols)
            new_cols[col_name] = (ss == lab).astype(np.uint8)

        return new_cols, labels


    def _multi_hot_encode_series(self, s: pd.Series, base_col: str, used_cols: set, drop_original=True):
        """
        Multi-hot encode multi-value cells into multiple 0/1 columns.
        Returns (new_cols_dict, labels, created_colnames)
        """
        labels_series = s.apply(
            lambda x: [p.strip() for p in self._split_respecting_parentheses(x) if p.strip()]
            if pd.notna(x) else []
        )
        uniques = sorted({lab for row in labels_series for lab in row})
        if not uniques:
            return {}, [], []

        mat = np.zeros((len(s), len(uniques)), dtype=np.uint8)
        idx = {lab: i for i, lab in enumerate(uniques)}

        for r, row_labels in enumerate(labels_series):
            for lab in row_labels:
                mat[r, idx[lab]] = 1

        new_cols = {}
        created = []
        safe_base = self._safe_sql_identifier(base_col)

        for i, lab in enumerate(uniques):
            safe_lab = self._safe_sql_identifier(lab)
            col_name = self._make_unique_name(f"{safe_base}__{safe_lab}", used_cols)
            new_cols[col_name] = mat[:, i]
            created.append(col_name)

        return new_cols, uniques, created


    def _bitmask_encode_series(self, s: pd.Series):
        """
        Compact numeric encoding for multi-value cells:
        each label gets a bit position -> output integer bitmask.
        Returns (encoded_series_int, labels_in_order)
        """
        labels_series = s.apply(
            lambda x: [p.strip() for p in self._split_respecting_parentheses(x) if p.strip()]
            if pd.notna(x) else []
        )
        uniques = sorted({lab for row in labels_series for lab in row})
        if not uniques:
            return None, []

        idx = {lab: i for i, lab in enumerate(uniques)}

        def enc(row_labels):
            total = 0
            for lab in row_labels:
                total |= (1 << idx[lab])
            return total

        out = labels_series.map(enc).astype(np.int64)
        return out, uniques


    def _hybrid_encode_df(self, df: pd.DataFrame, selected_cols: list, mode: str = "AUTO"):
        """
        mode:
        - AUTO (recommended)
        - FORCE_LABEL   -> label/binary (2 uniques) else one-hot
        - FORCE_ONEHOT  -> one-hot even if 2 uniques
        - FORCE_MULTIHOT-> multi-hot (requires multi-value)
        - FORCE_BITMASK -> bitmask int (requires multi-value)
        Returns: (df_encoded, encoding_map, report_lines)
        """
        df = df.copy()
        encoding_map = {}
        report = []

        used_cols = set(df.columns)

        for col in selected_cols:
            if col not in df.columns:
                report.append(f"• Skipped '{col}' (not found).")
                continue

            s = df[col]
            is_multi = self._is_multi_value_column(s)

            # ---------- MULTI VALUE ----------
            if is_multi:
                if mode == "FORCE_BITMASK":
                    encoded, labels = self._bitmask_encode_series(s)
                    if encoded is None:
                        report.append(f"• '{col}': no labels found (skipped).")
                        continue

                    safe_col = self._safe_sql_identifier(col)
                    safe_col = self._make_unique_name(f"{safe_col}__BITMASK", used_cols)

                    df[safe_col] = encoded
                    df.drop(columns=[col], inplace=True, errors="ignore")

                    encoding_map[str(col)] = {
                        "type": "bitmask",
                        "new_column": safe_col,
                        "labels": labels
                    }
                    report.append(f"✓ '{col}': multi-value → BITMASK int (1 column).")
                    continue

                # AUTO or FORCE_MULTIHOT
                if mode in ("AUTO", "FORCE_MULTIHOT"):
                    new_cols, labels, created = self._multi_hot_encode_series(s, col, used_cols)
                    if not created:
                        report.append(f"• '{col}': no labels found (skipped).")
                        continue

                    for nc, values in new_cols.items():
                        df[nc] = values

                    df.drop(columns=[col], inplace=True, errors="ignore")

                    encoding_map[str(col)] = {
                        "type": "multi_hot",
                        "new_columns": created,
                        "labels": labels
                    }
                    report.append(f"✓ '{col}': multi-value → MULTI-HOT ({len(created)} columns).")
                    continue

                # If forced label/onehot on multi-value, warn and skip
                report.append(f"• '{col}': multi-value column, but mode '{mode}' doesn't support it (skipped).")
                continue

            # ---------- SINGLE VALUE ----------
            # Work with unique non-null categories
            uniques = [v for v in s.dropna().astype(str).unique()]
            uniques = list(dict.fromkeys(uniques))
            n_uniques = len(uniques)

            if n_uniques == 0:
                report.append(f"• '{col}': only nulls (skipped).")
                continue

            # FORCE_ONEHOT always one-hot
            if mode == "FORCE_ONEHOT":
                new_cols, labels = self._one_hot_encode_series(s, col, used_cols)
                for nc, ser in new_cols.items():
                    df[nc] = ser
                df.drop(columns=[col], inplace=True, errors="ignore")

                encoding_map[str(col)] = {
                    "type": "one_hot",
                    "new_columns": list(new_cols.keys()),
                    "labels": labels
                }
                report.append(f"✓ '{col}': single-value → ONE-HOT ({len(new_cols)} columns).")
                continue

            # AUTO or FORCE_LABEL:
            # 2 uniques -> binary label; else -> one-hot
            if n_uniques == 2 and mode in ("AUTO", "FORCE_LABEL"):
                encoded, mapping = self._binary_label_encode_series(s)

                safe_col = self._safe_sql_identifier(col)
                safe_col = self._make_unique_name(safe_col, used_cols)

                df[safe_col] = encoded.astype("Int64")  # nullable int
                df.drop(columns=[col], inplace=True, errors="ignore")

                encoding_map[str(col)] = {
                    "type": "binary_label",
                    "new_column": safe_col,
                    "mapping": mapping
                }
                report.append(f"✓ '{col}': 2 values → BINARY LABEL (0/1).")
                continue

            # otherwise one-hot
            new_cols, labels = self._one_hot_encode_series(s, col, used_cols)
            for nc, ser in new_cols.items():
                df[nc] = ser
            df.drop(columns=[col], inplace=True, errors="ignore")

            encoding_map[str(col)] = {
                "type": "one_hot",
                "new_columns": list(new_cols.keys()),
                "labels": labels
            }
            report.append(f"✓ '{col}': {n_uniques} values → ONE-HOT ({len(new_cols)} columns).")

        return df, encoding_map, report


    def _save_hybrid_encoding_map(self, encoding_map: dict):
        """
        Saves ONE file and merges with existing mapping (so it doesn't overwrite previous encodes).
        """
        if not self.loaded_filename or not encoding_map:
            return

        base_path = Path(self.loaded_filename).parent
        base_name = Path(self.loaded_filename).stem
        out_folder = base_path / f"{base_name}_cleaned"
        out_folder.mkdir(exist_ok=True)

        out_path = out_folder / f"{base_name}_encoding_map.json"

        payload = {
            "filename": self.loaded_filename,
            "encoding_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "encoding_map": {}
        }

        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        payload["encoding_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        payload.setdefault("encoding_map", {})
        payload["encoding_map"].update(encoding_map)

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"✓ Updated encoding map: {out_path}")


    def apply_main_action(self):
        """
        Single 'Apply' button behavior (the one near Save):
        - If user is currently on Encoding tab -> apply encoding
        - Otherwise -> apply normal cleaning (drop/rename/value mapping)
        """
        try:
            current_widget = self.tabs.currentWidget()

            # If on encoding tab, run encoding
            if hasattr(self, "encoding_tab") and current_widget is self.encoding_tab:
                self.apply_selected_encoding()
                return

            # Otherwise normal Apply behavior
            self.apply_only()

        except Exception as e:
            self.log(f"❌ APPLY MAIN ACTION ERROR: {e}")
            self.log(traceback.format_exc())
            self.show_msg(
                QMessageBox.Icon.Critical,
                "Apply Error",
                f"Failed to apply changes.\n\nTechnical details:\n{e}"
            )

