import os, sys
def fix_qt_plugin_path():
    try:
        import PySide6
        pyside_dir = os.path.dirname(PySide6.__file__)
        plugin_path = os.path.join(pyside_dir, 'plugins')
        os.environ['QT_PLUGIN_PATH'] = plugin_path
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(plugin_path, 'platforms')
    except Exception as e:
        print(f"Warning: Could not set Qt plugin path: {e}")

fix_qt_plugin_path()

import sys
from pathlib import Path  
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QStyledItemDelegate

from config import PALETTE
from home_page import LandingPage
from loading_page import DatasetLoding_Window
from ploting_page import Plotting_Page
from cleaning_page import DatasetCleaning_Window
from database import CerebroXDB   

class CenterDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CerebroX")
        self.setMinimumSize(QSize(980, 640))

        self.setStyleSheet(f"""
            QMainWindow {{
                background: {PALETTE['bg']};
            }}
            QStackedWidget {{
                background: {PALETTE['bg']};
            }}
        """)

        print("[DB] Database connected to MySQL: cerebrox_data")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.landing = LandingPage(self.go_loading)
        self.loading = DatasetLoding_Window()
        self.cleaning = DatasetCleaning_Window()
        self.plotting = Plotting_Page()

        self.loading.main_window = self
        self.cleaning.main_window = self
        self.plotting.main_window = self

        self.stack.addWidget(self.landing)           
        self.stack.addWidget(self.loading)           
        self.stack.addWidget(self.cleaning)   
        self.stack.addWidget(self.plotting)          

        self._connect_navigation()

    def _connect_navigation(self):
        if hasattr(self.loading, "next_clicked"):
            self.loading.next_clicked.connect(self.go_cleaning)
        
        if hasattr(self.cleaning, "next_btn"):
            self.cleaning.next_btn.clicked.connect(self.go_plotting)
        if hasattr(self.cleaning, "back_btn"):
            self.cleaning.back_btn.clicked.connect(self.go_loading)
        
        if hasattr(self.plotting, "back_btn"):
            self.plotting.back_btn.clicked.connect(self.go_cleaning)

    def go_loading(self):
        self.stack.setCurrentWidget(self.loading)

    def go_cleaning(self):
        print("=== Navigating to Cleaning Page ===")

        if getattr(self.loading, "current_df", None) is not None:
            sheets_dfs = getattr(self.loading, "sheets_dfs", {})
            processed_path = getattr(self.loading, "processed_path", None)

            print(f"Passing data to cleaning: shape={self.loading.current_df.shape}")
            print(f"Number of sheets: {len(sheets_dfs)}")

            self.cleaning.set_dataframes(self.loading.current_df, sheets_dfs)
            self.cleaning.processed_path = processed_path
        else:
            print("⚠ Warning: No data in loading page")

        self.stack.setCurrentWidget(self.cleaning)

    def go_plotting(self):
        print("=== Navigating to Plotting Page ===")

        if getattr(self.cleaning, "merged_df", None) is not None:
            sheets_dfs = getattr(self.cleaning, "sheets_dfs", {})
            processed_path = getattr(self.cleaning, "processed_path", None)

            print(f"Passing data to plotting: shape={self.cleaning.merged_df.shape}")
            print(f"Number of sheets: {len(sheets_dfs)}")

            self.plotting.set_dataframes(self.cleaning.merged_df, sheets_dfs, processed_path)
        else:
            print("⚠ Warning: No data in cleaning window")

        self.stack.setCurrentWidget(self.plotting)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
