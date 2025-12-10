from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QTableWidgetItem, 
    QHBoxLayout, QLineEdit, QFileDialog, QTableWidget, QMessageBox
)

from PySide6.QtCore import Qt
from config import PALETTE

import os
import pandas as pd

from dataset_loading import Data_set

class DatasetLoding_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Loader")
        self.setStyleSheet(f"background: {PALETTE['bg']};")

        self.current_df = None
        self.processed_path = None
        self.sheets_dfs = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(20)

        title = QLabel("Dataset Loader")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title.setStyleSheet(f"color: {PALETTE['text']};")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        main_layout.addWidget(title)

        file_section = QVBoxLayout()

        subtitle_load = QLabel("Select Dataset File")
        subtitle_load.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle_load.setStyleSheet(f"color: {PALETTE['text']};")
        subtitle_load.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        file_section.addWidget(subtitle_load)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)

        self.file_btn = QPushButton("Choose file…")
        self.file_btn.setFixedSize(140, 36)
        self.file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.file_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['primary']};
                color:{PALETTE['text']};
                border:none; border-radius:6px; padding:6px 14px;
                font-family:'Segoe UI';
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
        self.file_btn.clicked.connect(self.pick_file)

        self.file_path = QLineEdit()
        self.file_path.setFixedHeight(40)
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("No file selected")
        self.file_path.setStyleSheet(
            f"""
            QLineEdit {{
                color:{PALETTE['text']};
                background:{PALETTE['panel']};
                border:1px solid {PALETTE['border']};
                border-radius:6px; padding:6px;
            }}
            """
        )

        file_row.addWidget(self.file_btn)
        file_row.addWidget(self.file_path, stretch=1)
        file_section.addLayout(file_row)

        note = QLabel("Note: Accepted file formats: CSV, XLSX only.")
        note.setStyleSheet(f"color: {PALETTE['muted']};")
        note.setFont(QFont("Segoe UI", 11))
        file_section.addWidget(note)

        main_layout.addLayout(file_section)

        self.process_btn = QPushButton("Process Dataset")
        self.file_btn.setFixedSize(140, 36)
        self.process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['primary']};
                color:{PALETTE['text']};
                border:none; border-radius:6px; padding:6px 14px;
                font-family:'Segoe UI';
                font-size:14px;
            }}
            QPushButton:hover {{
                background:{PALETTE['accent_hover']};
            }}
            QPushButton:pressed {{
                background:{PALETTE['glow']};
            }}
            """
        )
        self.process_btn.clicked.connect(self.process_file)
        self.process_btn.hide()
        self.process_btn.setEnabled(False)
        main_layout.addWidget(self.process_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.table = QTableWidget()
        self.table.setStyleSheet(
            f"""
            QTableWidget {{
                color:{PALETTE['text']};
                background:{PALETTE['panel']};
                gridline-color:{PALETTE['grid']};
                border:1px solid {PALETTE['border']};
                border-radius:6px;
            }}
            QHeaderView::section {{
                background:{PALETTE['panel']};
                color:{PALETTE['text']};
                border:1px solid {PALETTE['border']};
                padding:4px;
            }}
            QTableWidget::item:selected {{
                background: #3b1f6b;
                color: {PALETTE['text']};
            }}
            """
        )
        main_layout.addWidget(self.table, stretch=1)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(130, 36)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setStyleSheet(
            f"""
            QPushButton {{
                background:{PALETTE['primary']};
                color:{PALETTE['text']};
                border:none; border-radius:6px; padding:6px 14px;
                font-family:'Segoe UI';
                font-size:14px;
            }}
            QPushButton:hover {{
                background:{PALETTE['accent_hover']};
            }}
            QPushButton:pressed {{
                background:{PALETTE['glow']};
            }}
            """
        )
        self.next_btn.clicked.connect(self.go_next)
        self.next_btn.hide()
        button_row.addWidget(self.next_btn)
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

        self.setAcceptDrops(True)

    def go_back(self):
        main_window = self.window()
        if hasattr(main_window, "stack"):
            main_window.stack.setCurrentIndex(0)

    def go_next(self):
        main_window = self.window()

        print("DEBUG → processed_path =", self.processed_path)
        print("DEBUG → original_path  =", self.file_path.text())

        filename_to_pass = self.processed_path or self.file_path.text()
        print("DEBUG → filename_to_pass =", filename_to_pass)

        if hasattr(main_window, "cleaning"):
            print("DEBUG → setting filename on cleaning window")
            main_window.cleaning.set_loaded_filename(filename_to_pass)
        else:
            print("DEBUG → cleaning window NOT FOUND")

        if hasattr(main_window, "go_cleaning"):
            print("DEBUG → navigating to cleaning")
            main_window.go_cleaning()
        else:
            print("DEBUG → go_cleaning NOT FOUND")
   

        
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
            /* 🔑 remove the black strip behind the text */
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


    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
        )
        if path:
            self.handle_file(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
            if urls:
                e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            path = urls[0].toLocalFile()
            self.handle_file(path)

    def handle_file(self, path):
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext not in [".csv", ".xlsx"]:
            self.show_msg(
                QMessageBox.Icon.Warning,
                "Invalid file type",
                "Only CSV and XLSX files are supported.\nPlease select a valid dataset file."
            )
            return

        self.file_path.setText(path)

        try:
            if ext == ".csv":
                df = pd.read_csv(path)
                self.current_df = df
                self.sheets_dfs = {} 
                self.show_table_preview(df)
                self.show_msg(
                    QMessageBox.Icon.Information,
                    "CSV Loaded",
                    f"CSV file loaded successfully!\nNo processing needed.\n\nPath:\n{path}"
                )
                self.process_btn.hide()
                self.next_btn.show()

            else:
                self.current_df = None
                self.process_btn.show()
                self.process_btn.setEnabled(True)
                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Excel File Detected",
                    "Excel file selected. Click 'Process Dataset' to continue."
                )

        except Exception as e:
            self.show_msg(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed to load file:\n{e}"
            )
            self.process_btn.setEnabled(False)

    def process_file(self):
        file_path = self.file_path.text()
        if not file_path or not os.path.exists(file_path):
            self.show_msg(QMessageBox.Icon.Warning, "No file", "Please select a file first.")
            return

        root, ext = os.path.splitext(file_path)
        ext = ext.lower()

        base_dir = os.path.dirname(file_path)
        base_name = os.path.basename(root)
        run_folder = os.path.join(base_dir, base_name + "_run")
        os.makedirs(run_folder, exist_ok=True)

        output_path = os.path.join(run_folder, base_name + "_processed.csv")
        self.processed_path = output_path

        if os.path.exists(output_path):
            os.remove(output_path)

        try:
            data_processor = Data_set(dataset_path_old=file_path,
                                    dataset_path_new=output_path)

            if ext == ".xlsx":
                df = data_processor.load_google_excel(file_path)
                df.to_csv(output_path, index=False, encoding="utf-8-sig")
                self.current_df = df

                self.sheets_dfs = getattr(data_processor, "sheets_dfs", {})
                self.show_table_preview(self.current_df)
                self.next_btn.show()

                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Success",
                    f"Excel dataset processed successfully!\n\n"
                    f"Folder:\n{run_folder}\n\n"
                    f"Combined file:\n{output_path}"
                )
                return

            elif ext == ".csv":
                df = pd.read_csv(file_path)
                df.to_csv(output_path, index=False, encoding="utf-8-sig")

                self.current_df = df
                self.show_table_preview(self.current_df)
                self.next_btn.show()

                self.show_msg(
                    QMessageBox.Icon.Information,
                    "Success",
                    f"CSV file processed successfully!\n\n"
                    f"Saved to:\n{output_path}"
                )
                return

            else:
                self.show_msg(
                    QMessageBox.Icon.Warning,
                    "Unsupported file",
                    "Only CSV and XLSX files are supported."
                )
                return

        except Exception as e:
            self.show_msg(QMessageBox.Icon.Critical, "Error", f"Failed to process file:\n{e}")

    def show_table_preview(self, df):
        preview = df.head(50)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])

        for i in range(len(preview)):
            for j in range(len(preview.columns)):
                val = "" if pd.isna(preview.iat[i, j]) else str(preview.iat[i, j])
                self.table.setItem(i, j, QTableWidgetItem(val))

        self.table.resizeColumnsToContents()
