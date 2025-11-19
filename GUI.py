import sys
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget
)

from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtCore import Qt

from config import PALETTE
from home_page import LandingPage
from loading_page import DatasetLoding_Window
from ploting_page import Plotting_Page 
from cleaning_page import DatasetCleaning_Window


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

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create pages
        self.landing = LandingPage(self.go_loading)
        self.loading = DatasetLoding_Window()
        self.cleaning_window = DatasetCleaning_Window()  # Changed from self.cleaning
        self.plotting = Plotting_Page()   

        self.stack.addWidget(self.landing)
        self.stack.addWidget(self.loading)
        self.stack.addWidget(self.cleaning_window)  
        self.stack.addWidget(self.plotting)
        
        # Connect navigation signals
        self._connect_navigation()
    
    def _connect_navigation(self):
        """Connect page navigation signals"""
        # Add navigation buttons/signals from your pages
        # Example: if loading page has a "next" button
        if hasattr(self.loading, 'next_clicked'):
            self.loading.next_clicked.connect(self.go_cleaning)
        
        # If cleaning page has a "plot" or "next" button
        if hasattr(self.cleaning_window, 'plot_clicked'):
            self.cleaning_window.plot_clicked.connect(self.go_plotting)
    
    def go_loading(self):
        """Navigate to loading page"""
        self.stack.setCurrentWidget(self.loading)
    
    def go_cleaning(self):
        """Navigate to cleaning page and pass data"""
        print("=== Navigating to Cleaning Page ===")
        
        if self.loading.current_df is not None:
            sheets_dfs = getattr(self.loading, "sheets_dfs", {})
            print(f"Passing data to cleaning: shape={self.loading.current_df.shape}")
            print(f"Number of sheets: {len(sheets_dfs)}")
            
            self.cleaning_window.set_dataframes(self.loading.current_df, sheets_dfs)
            self.cleaning_window.processed_path = getattr(self.loading, "processed_path", None)
        else:
            print("⚠ Warning: No data in loading page")

        self.stack.setCurrentWidget(self.cleaning_window)
    
    def go_plotting(self):
        """Navigate to plotting page and pass data"""
        print("=== Navigating to Plotting Page ===")
        
        if hasattr(self.cleaning_window, 'merged_df') and self.cleaning_window.merged_df is not None:
            sheets_dfs = getattr(self.cleaning_window, "sheets_dfs", {})
            print(f"Passing data to plotting: shape={self.cleaning_window.merged_df.shape}")
            print(f"Number of sheets: {len(sheets_dfs)}")
            
            self.plotting.set_dataframes(self.cleaning_window.merged_df, sheets_dfs)
        else:
            print("⚠ Warning: No data in cleaning window")
        
        self.stack.setCurrentWidget(self.plotting)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())