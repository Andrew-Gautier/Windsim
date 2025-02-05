import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget, QCheckBox

class CSVPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Column Plotter")
        self.setGeometry(100, 100, 400, 300)
        
        self.layout = QVBoxLayout()
        
        self.load_frt_button = QPushButton("Load _frt CSV File")
        self.load_frt_button.clicked.connect(lambda: self.load_csv(True))
        self.layout.addWidget(self.load_frt_button)
        
        self.load_non_frt_button = QPushButton("Load Non-_frt CSV File")
        self.load_non_frt_button.clicked.connect(lambda: self.load_csv(False))
        self.layout.addWidget(self.load_non_frt_button)
        
        self.column_list = QListWidget()
        self.layout.addWidget(self.column_list)
        
        self.highlight_checkbox = QCheckBox("Highlight Selected Columns")
        self.layout.addWidget(self.highlight_checkbox)
        
        self.plot_button = QPushButton("Plot Selected Columns")
        self.plot_button.clicked.connect(self.plot_columns)
        self.layout.addWidget(self.plot_button)
        
        self.setLayout(self.layout)
        self.df = None
    
    def load_csv(self, is_frt):
        file_dialog = QFileDialog()
        file_filter = "CSV Files (*.csv)"
        directory = "Wind_Faults"
        
        file_path, _ = file_dialog.getOpenFileName(self, "Open CSV File", directory, file_filter)
        
        if file_path:
            filename = os.path.basename(file_path)
            if ("_frt" in filename) == is_frt:
                self.df = pd.read_csv(file_path)
                self.column_list.clear()
                self.column_list.addItems(self.df.columns)
            else:
                print("Selected file type does not match the chosen category.")
    
    def plot_columns(self):
        if self.df is None:
            return
        
        selected_items = self.column_list.selectedItems()
        selected_columns = [item.text() for item in selected_items]
        
        plt.figure(figsize=(10, 6))
        
        for column in self.df.columns:
            if column in selected_columns:
                plt.plot(self.df[column], label=column, linewidth=2.5)
            else:
                plt.plot(self.df[column], label=column, alpha=0.2 if self.highlight_checkbox.isChecked() else 1.0)
        
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.title("Selected Column Plots")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVPlotter()
    window.show()
    sys.exit(app.exec())
