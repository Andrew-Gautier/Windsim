import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QListWidget, QWidget, QLabel, QComboBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class CSVPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSV Plotter')
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # File selection
        self.file_button = QPushButton('Select CSV Files', self)
        self.file_button.clicked.connect(self.select_files)
        self.layout.addWidget(self.file_button)

        # List to display selected files
        self.file_list = QListWidget(self)
        self.layout.addWidget(self.file_list)

        # Column selection
        self.column_label = QLabel('Select Columns to Plot:', self)
        self.layout.addWidget(self.column_label)

        self.column_list = QListWidget(self)
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.column_list)

        # Highlight selection
        self.highlight_label = QLabel('Select Column to Highlight:', self)
        self.layout.addWidget(self.highlight_label)

        self.highlight_combo = QComboBox(self)
        self.layout.addWidget(self.highlight_combo)

        # Plot button
        self.plot_button = QPushButton('Plot', self)
        self.plot_button.clicked.connect(self.plot_data)
        self.layout.addWidget(self.plot_button)

        # Matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def select_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if files:
            self.file_list.clear()
            self.file_list.addItems(files)
            self.load_columns(files[0])  # Load columns from the first file

    def load_columns(self, file_path):
        self.column_list.clear()
        self.highlight_combo.clear()
        df = pd.read_csv(file_path)
        self.column_list.addItems(df.columns)
        self.highlight_combo.addItems(df.columns)

    def plot_data(self):
        selected_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        selected_columns = [item.text() for item in self.column_list.selectedItems()]
        highlight_column = self.highlight_combo.currentText()

        if not selected_files or not selected_columns:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for file in selected_files:
            df = pd.read_csv(file)
            for column in selected_columns:
                if column in df.columns:
                    if column == highlight_column:
                        ax.plot(df[column], label=f'{os.path.basename(file)} - {column}', linewidth=2.5)
                    else:
                        ax.plot(df[column], label=f'{os.path.basename(file)} - {column}', alpha=0.3)

        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Comparison of Selected Columns')
        ax.legend()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CSVPlotterApp()
    ex.show()
    sys.exit(app.exec_())

