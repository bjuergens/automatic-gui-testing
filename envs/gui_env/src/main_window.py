import logging
import random
import sys
from functools import partial

import numpy as np
from PySide6.QtCore import Qt, QPoint, Slot, Signal
from PySide6.QtGui import QAction, QPalette, QPaintEvent, QMouseEvent, QColor
from PySide6.QtGui import QPainter, QPen, QBrush, QColorConstants
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar

from envs.gui_env.src.backend.calculator import Calculator
from envs.gui_env.src.backend.figure_printer import FigurePrinter
from envs.gui_env.src.backend.text_printer import TextPrinter
from envs.gui_env.src.settings_dialog import SettingsDialog
from envs.gui_env.src.utils.utils import load_ui, convert_qimage_to_ndarray


class MainWindow(QMainWindow):
    screenshot_signal = Signal(np.ndarray)
    screenshot_and_coordinates_signal = Signal(np.ndarray, int, int)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.setWindowTitle("test-gui-worldmodels")
        self.setFixedSize(600, 600)

        self.main_window = load_ui("envs/gui_env/src/main_window.ui")

        # Initialize menu bar
        self.menu_bar = QMenuBar()
        self.settings_action = QAction("Settings")
        self.menu_bar.addAction(self.settings_action)
        self.setMenuBar(self.menu_bar)

        # Christmas Tree is hidden at first, must be activated in the settings
        self.main_window.figure_printer_button.setVisible(False)

        # TODO populate this, and implement changing this when appropriate widgets are clicked
        self.currently_shown_widgets = []
        self.points = []

        self._connect_buttons()

        self.text_printer = TextPrinter(self.main_window.text_printer_output)
        self.calculator = Calculator(self.main_window.calculator_output, self.main_window.first_operand_combobox,
                                     self.main_window.second_operand_combobox, self.main_window.math_operator_combobox)
        self.figure_printer = FigurePrinter(self.main_window.figure_combobox)

        self.settings_dialog = SettingsDialog(text_printer=self.text_printer, calculator=self.calculator,
                                              figure_printer=self.figure_printer, parent=self)
        self.settings_action.triggered.connect(self.settings_dialog.exec)

        self.settings_dialog.figure_printer_activated.connect(self._toogle_figure_printing)

        self.setCentralWidget(self.main_window)

    def _connect_buttons(self):
        # Text Printer
        self.main_window.text_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 0)
        )
        self.main_window.start_text_printer_button.clicked.connect(self.start_text_printing)

        # Calculator
        self.main_window.calculator_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 1)
        )
        self.main_window.start_calculation_button.clicked.connect(self.start_calculation)

        # Christmas Tree
        self.main_window.figure_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 2)
        )
        self.main_window.start_drawing_figure_button.clicked.connect(self.start_drawing_figure)

    @Slot(bool)
    def _toogle_figure_printing(self, checked: bool):
        if checked:
            self.main_window.figure_printer_button.setVisible(True)
            self.main_window.figure_printer_button.setEnabled(True)
        else:
            self.main_window.figure_printer_button.setVisible(False)
            self.main_window.figure_printer_button.setEnabled(False)

            # Could be that the stacked widget is still on the figure printer but we deactivate it, therefore simply
            # switch back to the first index
            if self.main_window.main_stacked_widget.currentIndex() == 2:
                self.main_window.main_stacked_widget.setCurrentIndex(0)

    def start_text_printing(self):
        self.text_printer.apply_settings()
        self.text_printer.generate_text()

    def start_calculation(self):
        self.calculator.calculate()

    def start_drawing_figure(self):
        if self.figure_printer.current_color == "green":
            color = QColorConstants.Green
        elif self.figure_printer.current_color == "black":
            color = QColorConstants.Black
        elif self.figure_printer.current_color == "blue":
            color = QColorConstants.Blue
        elif self.figure_printer.current_color == "brown":
            color = QColor("#8b4513")
        else:
            raise RuntimeError("Invalid color for the figure printer specified!")

        palette: QPalette = self.main_window.figure_printer_output.palette()
        palette.setColor(QPalette.Text, color)
        self.main_window.figure_printer_output.setPalette(palette)

        chosen_figure = self.main_window.figure_combobox.currentText()
        figure = self.figure_printer.get_figure_by_name(chosen_figure)

        self.main_window.figure_printer_output.setPlainText(figure)

    def take_screenshot(self) -> np.ndarray:
        screen = QApplication.primaryScreen()
        window = self.window()
        screenshot = screen.grabWindow(window.winId(), 0, 0).toImage()

        return convert_qimage_to_ndarray(screenshot)

    @Slot(int, int)
    def simulate_click(self, pos_x: int, pos_y: int):
        pos = QPoint(pos_x, pos_y)
        recv_widget = self.childAt(pos)
        local_pos = recv_widget.mapFrom(self.main_window, pos)

        QTest.mouseClick(recv_widget, Qt.LeftButton, Qt.NoModifier, local_pos)

        logging.debug(f"Received pos {pos_x, pos_y}, clicked widget '{recv_widget}' at local pos '{local_pos}'")

        screenshot = self.take_screenshot()
        self.screenshot_signal.emit(screenshot)

    @Slot()
    def simulate_click_on_random_widget(self):
        randomly_selected_widget = random.choice(self.currently_shown_widgets)

        height = randomly_selected_widget.height()
        width = randomly_selected_widget.width()

        x = random.randint(0, width)
        y = random.randint(0, height)

        click_point = QPoint(x, y)

        QTest.mouseClick(randomly_selected_widget, Qt.LeftButton, Qt.NoModifier, click_point)

        # TODO this will probably fail in the settings dialog as this prints out a coordinate system in the dialog's
        #  coordinate system but we want the coordinate of the window
        # Convert coordinates in the widget's coordinate system to the coordinate system of the parent
        global_position: QPoint = randomly_selected_widget.mapToParent(click_point)

        logging.debug(
            f"Randomly selected '{randomly_selected_widget.text()}', local pos '{click_point}', and global pos " +
            f"'{global_position}'!")

        screenshot = self.take_screenshot()
        self.screenshot_and_coordinates_signal.emit(screenshot, global_position.x(), global_position.y())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.points.append(event.position())
        self.update()

        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        # Draws a red point for each click in the MainWindow (not on widgets!)
        qp = QPainter(self)
        pen = QPen(Qt.red, 5)
        brush = QBrush(Qt.red)
        qp.setPen(pen)
        qp.setBrush(brush)

        if self.points is not None:
            qp.drawPoints(self.points)

        qp.end()
        super().paintEvent(event)


def main():
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
