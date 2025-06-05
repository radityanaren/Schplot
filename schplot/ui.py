import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QCheckBox, QLabel, QListWidget, QGroupBox, QLineEdit, QScrollArea, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from .plot_settings import PlotSettings
from .logic import calculate_regression_stats

class RegressionDetailsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regression Details")
        self.setGeometry(200, 200, 800, 400)
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Property", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    def display_regression_details(self, plot, x, y, regression_type, stats):
        self.table.setRowCount(0)
        self.setWindowTitle(f"Regression Details: {plot.label} - {regression_type}")
        self.add_table_row("Regression Type", regression_type)
        self.add_table_row("Number of Points", str(len(x)))
        self.add_table_row("X Mean", f"{np.mean(x):.6f}")
        self.add_table_row("Y Mean", f"{np.mean(y):.6f}")
        self.add_table_row("X Standard Deviation", f"{np.std(x):.6f}")
        self.add_table_row("Y Standard Deviation", f"{np.std(y):.6f}")
        if regression_type == "Linear":
            m, b = stats["coefficients"]
            self.add_table_row("Formula", f"y = {m:.6f}x + {b:.6f}")
            self.add_table_row("Slope (m)", f"{m:.6f}")
            self.add_table_row("Y-Intercept (b)", f"{b:.6f}")
            self.add_table_row("Standard Error of Slope", f"{stats.get('stderr_slope', 'N/A')}")
        elif regression_type == "Exponential":
            a, b = stats["coefficients"]
            self.add_table_row("Formula", f"y = {np.exp(a):.6f} * e^({b:.6f}x)")
            self.add_table_row("Amplitude (A)", f"{np.exp(a):.6f}")
            self.add_table_row("Growth Rate (b)", f"{b:.6f}")
        elif regression_type == "Logarithmic":
            m, b = stats["coefficients"]
            self.add_table_row("Formula", f"y = {m:.6f} * ln(x) + {b:.6f}")
            self.add_table_row("Coefficient (m)", f"{m:.6f}")
            self.add_table_row("Y-Intercept (b)", f"{b:.6f}")
        self.add_table_row("R²", f"{stats['r2']:.6f}")
        self.add_table_row("Adjusted R²", f"{stats.get('adj_r2', 'N/A')}")
        self.add_table_row("Root Mean Square Error", f"{stats['rmse']:.6f}")
        self.add_table_row("Mean Absolute Error", f"{stats['mae']:.6f}")
    def add_table_row(self, property_name, value):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(property_name))
        self.table.setItem(row_position, 1, QTableWidgetItem(value))

class DataPlottingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Schplot")
        self.setGeometry(100, 100, 1400, 800)
        self.plots = []
        self.data = None
        self.filename = ""
        self.regression_stats = {}
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_widget, stretch=7)
        control_widget = QWidget()
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control_widget)
        control_layout = QVBoxLayout(control_widget)
        main_layout.addWidget(control_scroll, stretch=3)
        self.file_btn = QPushButton("Choose Files")
        self.file_btn.clicked.connect(self.select_file)
        control_layout.addWidget(self.file_btn)
        self.plot_list = QListWidget()
        self.plot_list.itemClicked.connect(self.update_plot_settings)
        control_layout.addWidget(QLabel("Plots:"))
        control_layout.addWidget(self.plot_list)
        btn_layout = QHBoxLayout()
        self.add_plot_btn = QPushButton("+")
        self.add_plot_btn.clicked.connect(self.add_plot)
        self.remove_plot_btn = QPushButton("-")
        self.remove_plot_btn.clicked.connect(self.remove_plot)
        btn_layout.addWidget(self.add_plot_btn)
        btn_layout.addWidget(self.remove_plot_btn)
        control_layout.addLayout(btn_layout)
        self.settings_group = QGroupBox("Plot Settings")
        settings_layout = QVBoxLayout()
        self.settings_group.setLayout(settings_layout)
        control_layout.addWidget(self.settings_group)
        self.plot_name_edit = QLineEdit()
        settings_layout.addWidget(QLabel("Plot Name:"))
        settings_layout.addWidget(self.plot_name_edit)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        settings_layout.addWidget(QLabel("X Variable:"))
        settings_layout.addWidget(self.x_combo)
        settings_layout.addWidget(QLabel("Y Variable:"))
        settings_layout.addWidget(self.y_combo)
        self.line_check = QCheckBox("Connected Lines")
        settings_layout.addWidget(self.line_check)
        self.regression_combo = QComboBox()
        self.regression_combo.addItems(["None", "Linear", "Exponential", "Logarithmic"])
        settings_layout.addWidget(QLabel("Regression Line:"))
        settings_layout.addWidget(self.regression_combo)
        self.reg_details_btn = QPushButton("Show Regression Details")
        self.reg_details_btn.clicked.connect(self.show_regression_details)
        settings_layout.addWidget(self.reg_details_btn)
        self.main_color_combo = QComboBox()
        self.main_color_combo.addItems(["Blue", "Green", "Red", "Cyan", "Magenta", "Yellow", "Black"])
        settings_layout.addWidget(QLabel("Data Color:"))
        settings_layout.addWidget(self.main_color_combo)
        self.regression_color_combo = QComboBox()
        self.regression_color_combo.addItems(["Blue", "Green", "Red", "Cyan", "Magenta", "Yellow", "Black"])
        settings_layout.addWidget(QLabel("Regression Color:"))
        settings_layout.addWidget(self.regression_color_combo)
        self.labels_group = QGroupBox("Plot Labels")
        labels_layout = QVBoxLayout()
        self.labels_group.setLayout(labels_layout)
        control_layout.addWidget(self.labels_group)
        self.title_edit = QLineEdit()
        labels_layout.addWidget(QLabel("Plot Title:"))
        labels_layout.addWidget(self.title_edit)
        self.x_label_edit = QLineEdit()
        labels_layout.addWidget(QLabel("X:"))
        labels_layout.addWidget(self.x_label_edit)
        self.y_label_edit = QLineEdit()
        labels_layout.addWidget(QLabel("Y:"))
        labels_layout.addWidget(self.y_label_edit)
        self.marker_check = QCheckBox("Data Markers")
        self.marker_check.setChecked(True)
        settings_layout.addWidget(self.marker_check)
        self.grid_check = QCheckBox("Grid")
        control_layout.addWidget(self.grid_check)
        self.update_btn = QPushButton("Update Plot")
        self.update_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(self.update_btn)
        self.save_code_btn = QPushButton("Save")
        self.save_code_btn.clicked.connect(self.save_simple_plot_code)
        control_layout.addWidget(self.save_code_btn)
        self.regression_details_window = RegressionDetailsWindow(self)

    def show_regression_details(self):
        current_row = self.plot_list.currentRow()
        if current_row < 0 or not self.plots:
            QMessageBox.warning(self, "Warning", "Please select a plot first")
            return
        plot = self.plots[current_row]
        if plot.regression == "None":
            QMessageBox.warning(self, "Warning", "No regression selected for this plot")
            return
        plot_key = f"{plot.label}_{plot.regression}"
        if plot_key not in self.regression_stats:
            QMessageBox.warning(self, "Warning", "Please update the plot to calculate regression stats first")
            return
        stats = self.regression_stats[plot_key]
        x = stats["x"]
        y = stats["y"]
        self.regression_details_window.display_regression_details(plot, x, y, plot.regression, stats)
        self.regression_details_window.exec_()

    def remove_plot(self):
        current_row = self.plot_list.currentRow()
        if current_row >= 0:
            del self.plots[current_row]
            self.plot_list.takeItem(current_row)
            if self.plots:
                self.plot_list.setCurrentRow(min(current_row, len(self.plots) - 1))
                self.update_plot_settings()
            else:
                self.clear_plot_settings()

    def add_plot(self):
        if self.data is None:
            return
        new_plot = PlotSettings()
        self.plots.append(new_plot)
        plot_name = f"Plot {len(self.plots)}"
        new_plot.label = plot_name
        self.plot_list.addItem(plot_name)
        self.plot_list.setCurrentRow(len(self.plots) - 1)
        self.update_plot_settings()

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_name:
            self.filename = file_name
            if file_name.endswith('.csv'):
                self.data = pd.read_csv(file_name, decimal=',', thousands='.', encoding='utf-8')
                for col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            elif file_name.endswith('.xlsx'):
                self.data = pd.read_excel(file_name)
                unnamed_cols = [col for col in self.data.columns if 'Unnamed:' in str(col)]
                self.data = self.data.drop(columns=unnamed_cols)
            self.x_combo.clear()
            self.y_combo.clear()
            self.x_combo.addItems(self.data.columns)
            self.y_combo.addItems(self.data.columns)

    def update_plot_settings(self):
        current_row = self.plot_list.currentRow()
        if current_row >= 0:
            plot = self.plots[current_row]
            self.plot_name_edit.setText(plot.label)
            self.x_combo.setCurrentText(plot.x_var)
            self.y_combo.setCurrentText(plot.y_var)
            self.line_check.setChecked(plot.show_line)
            self.marker_check.setChecked(plot.show_markers)
            self.regression_combo.setCurrentText(plot.regression)
            self.main_color_combo.setCurrentText(plot.main_color)
            self.regression_color_combo.setCurrentText(plot.regression_color)

    def clear_plot_settings(self):
        self.plot_name_edit.clear()
        self.x_combo.setCurrentIndex(-1)
        self.y_combo.setCurrentIndex(-1)
        self.line_check.setChecked(False)
        self.marker_check.setChecked(True)
        self.regression_combo.setCurrentIndex(0)
        self.main_color_combo.setCurrentIndex(0)
        self.regression_color_combo.setCurrentIndex(0)

    def update_plot(self):
        if self.data is None or not self.plots:
            return
        current_row = self.plot_list.currentRow()
        if current_row >= 0:
            plot = self.plots[current_row]
            plot.label = self.plot_name_edit.text()
            plot.x_var = self.x_combo.currentText()
            plot.y_var = self.y_combo.currentText()
            plot.show_line = self.line_check.isChecked()
            plot.show_markers = self.marker_check.isChecked()
            plot.regression = self.regression_combo.currentText()
            plot.main_color = self.main_color_combo.currentText()
            plot.regression_color = self.regression_color_combo.currentText()
            self.plot_list.item(current_row).setText(plot.label)
        self.ax.clear()
        self.regression_stats = {}
        for plot in self.plots:
            x = pd.to_numeric(self.data[plot.x_var], errors='coerce').values
            y = pd.to_numeric(self.data[plot.y_var], errors='coerce').values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) == 0 or len(y) == 0:
                QMessageBox.warning(self, "Data Error", f"No valid data points for {plot.label}")
                continue
            if plot.show_line and plot.show_markers:
                self.ax.plot(x, y, color=plot.main_color, label=plot.label, marker='o')
            elif plot.show_line:
                self.ax.plot(x, y, color=plot.main_color, label=plot.label)
            elif plot.show_markers:
                self.ax.scatter(x, y, color=plot.main_color, label=plot.label)
            if plot.regression != "None":
                try:
                    stats = calculate_regression_stats(x, y, plot.regression)
                    if stats:
                        if plot.regression == "Linear":
                            y_pred = np.poly1d(stats["coefficients"])(x)
                            formula = stats["formula"]
                            r2 = stats["r2"]
                            self.ax.plot(x, y_pred, color=plot.regression_color, linestyle='--', label=f"{plot.label} (Linear: {formula}, R-squared = {r2:.5f})")
                            self.regression_stats[f"{plot.label}_Linear"] = stats
                        elif plot.regression == "Exponential":
                            mask = y > 0
                            x_clean, y_clean = x[mask], y[mask]
                            y_pred = np.exp(np.poly1d((stats["coefficients"][1], stats["coefficients"][0]))(x_clean))
                            formula = stats["formula"]
                            r2 = stats["r2"]
                            self.ax.plot(x_clean, y_pred, color=plot.regression_color, linestyle='--', label=f"{plot.label} (Exp: {formula}, R-squared = {r2:.5f})")
                            self.regression_stats[f"{plot.label}_Exponential"] = stats
                        elif plot.regression == "Logarithmic":
                            mask = x > 0
                            x_clean, y_clean = x[mask], y[mask]
                            y_pred = np.poly1d((stats["coefficients"][0], stats["coefficients"][1]))(np.log(x_clean))
                            formula = stats["formula"]
                            r2 = stats["r2"]
                            self.ax.plot(x_clean, y_pred, color=plot.regression_color, linestyle='--', label=f"{plot.label} (Log: {formula}, R-squared = {r2:.5f})")
                            self.regression_stats[f"{plot.label}_Logarithmic"] = stats
                except np.linalg.LinAlgError:
                    QMessageBox.warning(self, "Regression Error", f"Could not perform regression for {plot.label}. The data might be constant or have other issues.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"An error occurred while processing {plot.label}: {str(e)}")
        self.ax.grid(self.grid_check.isChecked())
        self.ax.legend()
        self.ax.set_xlabel(self.x_label_edit.text())
        self.ax.set_ylabel(self.y_label_edit.text())
        self.ax.set_title(self.title_edit.text())
        self.canvas.draw()
        self.toolbar.update()

    def save_simple_plot_code(self):
        if not self.plots:
            return
        code = []
        code.append("# -*- coding: utf-8 -*-")
        code.append("import pandas as pd")
        code.append("import matplotlib.pyplot as plt")
        code.append("import numpy as np")
        code.append("from sklearn.metrics import r2_score\n")
        if self.filename.endswith('.csv'):
            code.append(f"data = pd.read_csv('{self.filename}')")
        elif self.filename.endswith('.xlsx'):
            code.append(f"data = pd.read_excel('{self.filename}')")
        code.append("")
        code.append("plt.figure(figsize=(10, 6))")
        for plot in self.plots:
            code.append(f"x = data['{plot.x_var}']")
            code.append(f"y = data['{plot.y_var}']")
            if plot.show_line and plot.show_markers:
                code.append(f"plt.plot(x, y, color='{plot.main_color}', label='{plot.label}', marker='o')")
            elif plot.show_line:
                code.append(f"plt.plot(x, y, color='{plot.main_color}', label='{plot.label}')")
            elif plot.show_markers:
                code.append(f"plt.scatter(x, y, color='{plot.main_color}', label='{plot.label}')")
            if plot.regression != "None":
                if plot.regression == "Linear":
                    code.append("z = np.polyfit(x, y, 1)")
                    code.append("p = np.poly1d(z)")
                    code.append("y_pred = p(x)")
                    code.append("r2 = r2_score(y, y_pred)")
                    code.append("formula = f'y = {z[0]:.7f}x + {z[1]:.7f}'")
                    code.append(f"plt.plot(x, p(x), color='{plot.regression_color}', linestyle='--', label=f'{plot.label} (Linear: {{formula}}, R-squared = {{r2:.5f}}')")
                elif plot.regression == "Exponential":
                    code.append("# Remove non-positive values")
                    code.append("mask = y > 0")
                    code.append("x_clean, y_clean = x[mask], y[mask]")
                    code.append("z = np.polyfit(x_clean, np.log(y_clean), 1)")
                    code.append("p = np.poly1d(z)")
                    code.append("y_pred = np.exp(p(x_clean))")
                    code.append("r2 = r2_score(y_clean, y_pred)")
                    code.append("formula = f'y = {np.exp(z[1]):.7f} * e^({z[0]:.7f}x)'")
                    code.append(f"plt.plot(x_clean, np.exp(p(x_clean)), color='{plot.regression_color}', linestyle='--', label=f'{plot.label} (Exp: {{formula}}, R-squared = {{r2:.5f}}')")
                elif plot.regression == "Logarithmic":
                    code.append("# Remove non-positive values")
                    code.append("mask = x > 0")
                    code.append("x_clean, y_clean = x[mask], y[mask]")
                    code.append("z = np.polyfit(np.log(x_clean), y_clean, 1)")
                    code.append("p = np.poly1d(z)")
                    code.append("y_pred = p(np.log(x_clean))")
                    code.append("r2 = r2_score(y_clean, y_pred)")
                    code.append("formula = f'y = {z[0]:.7f} * ln(x) + {z[1]:.7f}'")
                    code.append(f"plt.plot(x_clean, p(np.log(x_clean)), color='{plot.regression_color}', linestyle='--', label=f'{plot.label} (Log: {{formula}}, R-squared = {{r2:.5f}}')")
        code.append(f"plt.grid({self.grid_check.isChecked()})")
        code.append("plt.legend()")
        code.append(f"plt.xlabel('{self.x_label_edit.text()}')")
        code.append(f"plt.ylabel('{self.y_label_edit.text()}')")
        code.append(f"plt.title('{self.title_edit.text()}')")
        code.append("plt.show()")
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Simple Plot Code", "", "Python Files (*.py)")
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join(code)) 