import logging
import os
import sys
import time
from functools import partial

import coverage.exceptions
import numpy as np
from PySide6.QtCore import Qt, QPoint, Slot, Signal
from PySide6.QtGui import QAction, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QWidget, QComboBox
from coverage import Coverage

from envs.gui_env.src.backend.calculator import Calculator
from envs.gui_env.src.backend.figure_printer import FigurePrinter
from envs.gui_env.src.backend.text_printer import TextPrinter
from envs.gui_env.src.settings_dialog import SettingsDialog
from envs.gui_env.src.utils.utils import load_ui, convert_qimage_to_ndarray

WINDOW_SIZE = (448, 448)  # pragma: no cover


class MainWindow(QMainWindow):
    observation_signal = Signal(float, np.ndarray)  # pragma: no cover
    observation_and_coordinates_signal = Signal(float, np.ndarray, int, int)  # pragma: no cover

    def __init__(self, random_click_probability: float = None, random_seed: int = None, **kwargs):  # pragma: no cover
        super().__init__(**kwargs)

        self.setWindowTitle("test-gui-worldmodels")
        self.setFixedSize(*WINDOW_SIZE)
        self.setWindowFlag(Qt.FramelessWindowHint, True)

        self.main_window = load_ui("envs/gui_env/src/main_window.ui")

        self._initialize()

        # TODO populate this, and implement changing this when appropriate widgets are clicked
        self.currently_shown_widgets_main_window = []
        # Initially we start with these widgets
        self._set_currently_shown_widgets_text_printer()

        self._connect_buttons()

        self.text_printer = TextPrinter(self.main_window.text_printer_output)
        self.calculator = Calculator(self.main_window.calculator_output, self.main_window.first_operand_combobox,
                                     self.main_window.second_operand_combobox, self.main_window.math_operator_combobox)
        self.figure_printer = FigurePrinter(self.main_window.figure_printer_output, self.main_window.figure_combobox)

        self.settings_dialog = SettingsDialog(text_printer=self.text_printer, calculator=self.calculator,
                                              figure_printer=self.figure_printer, parent=self)

        self.settings_action.triggered.connect(self.settings_dialog.show)
        self.settings_dialog.figure_printer_activated.connect(self._toogle_figure_printing)

        self.setCentralWidget(self.main_window)

        self.current_coverage = Coverage(config_file="envs/gui_env/.coveragerc")
        self.old_coverage_percentage = self.get_current_coverage_percentage()

        self.random_click_probability = random_click_probability
        self.random_state = np.random.RandomState(random_seed)

        # self.main_window.frame_5: QWidget
        # size_pol = self.main_window.frame_5.sizePolicy()
        # size_pol.setRetainSizeWhenHidden(True)
        # self.main_window.frame_5.setSizePolicy(size_pol)
        # self.main_window.frame_5.setVisible(False)

        # QTimer.singleShot(0, self._test)

    def _initialize(self):
        # Initialize menu bar
        self.menu_bar = QMenuBar(parent=self)
        self.settings_action = QAction("Settings")
        self.menu_bar.addAction(self.settings_action)
        self.setMenuBar(self.menu_bar)

        # Car Configurator

        # Set top label to bold, to function as a headline
        font = self.main_window.car_configurator_headline_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.main_window.car_configurator_headline_label.setFont(font)

        # Make the configurations invisible
        default_size_policy = self.main_window.tire_selection_frame.sizePolicy()
        # This keeps the space in the layout, even when the widget is not visible
        default_size_policy.setRetainSizeWhenHidden(True)
        self.main_window.tire_selection_frame.setSizePolicy(default_size_policy)
        self.main_window.tire_selection_frame.setVisible(False)

        self.main_window.interior_design_frame.setSizePolicy(default_size_policy)
        self.main_window.interior_design_frame.setVisible(False)

        self.main_window.propulsion_system_frame.setSizePolicy(default_size_policy)
        self.main_window.propulsion_system_frame.setVisible(False)

        # Use the size policy from the button because I do not know if it is different to the one from the QFrame
        button_size_policy = self.main_window.show_configuration_button.sizePolicy()
        button_size_policy.setRetainSizeWhenHidden(True)
        self.main_window.show_configuration_button.setSizePolicy(button_size_policy)
        self.main_window.show_configuration_button.setVisible(False)

        # Figure Printer

        # Figure Printer is hidden at first, must be activated in the settings
        self.main_window.figure_printer_button.setVisible(False)

        # Need a monospace font to display the ASCII art correctly
        document = self.main_window.figure_printer_output.document()
        font = QFontDatabase.font("Bitstream Vera Sans Mono", "Normal", 10)
        document.setDefaultFont(font)

    def _connect_buttons(self):
        # Text Printer
        self.main_window.text_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 0)
        )
        self.main_window.text_printer_button.clicked.connect(self._set_currently_shown_widgets_text_printer)
        self.main_window.start_text_printer_button.clicked.connect(self.start_text_printing)

        # Calculator
        self.main_window.calculator_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 1)
        )
        self.main_window.calculator_button.clicked.connect(self._set_currently_shown_widgets_calculator)
        self.main_window.start_calculation_button.clicked.connect(self.start_calculation)

        # Car Configurator
        self.main_window.car_configurator_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 2)
        )

        # Figure Printer
        self.main_window.figure_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 3)
        )
        self.main_window.figure_printer_button.clicked.connect(self._set_currently_shown_widgets_figure_printer)
        self.main_window.start_drawing_figure_button.clicked.connect(self.start_drawing_figure)

    @Slot(bool)
    def _toogle_figure_printing(self, checked: bool):
        if checked:
            self.main_window.figure_printer_button.setVisible(True)
            self.main_window.figure_printer_button.setEnabled(True)
            if not self.main_window.figure_printer_button in self.currently_shown_widgets_main_window:
                self.currently_shown_widgets_main_window.append(self.main_window.figure_printer_button)
        else:
            self.main_window.figure_printer_button.setVisible(False)
            self.main_window.figure_printer_button.setEnabled(False)
            try:
                self.currently_shown_widgets_main_window.remove(self.main_window.figure_printer_button)
            except ValueError:
                pass

            # Could be that the stacked widget is still on the figure printer but we deactivate it, therefore simply
            # switch back to the first index
            if self.main_window.main_stacked_widget.currentIndex() == 2:
                self.main_window.main_stacked_widget.setCurrentIndex(0)

    def _get_main_widgets_main_window(self):
        # TODO settings qaction
        currently_shown_widgets_main_window = [
            self.main_window.text_printer_button,
            self.main_window.calculator_button
        ]

        if self.main_window.figure_printer_button.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.figure_printer_button)

        return currently_shown_widgets_main_window

    def _set_currently_shown_widgets_text_printer(self):
        currently_show_widgets_main_window = self._get_main_widgets_main_window()
        currently_show_widgets_main_window.append(self.main_window.start_text_printer_button)

        self.currently_shown_widgets_main_window = currently_show_widgets_main_window

    def _set_currently_shown_widgets_calculator(self):
        currently_show_widgets_main_window = self._get_main_widgets_main_window()
        currently_show_widgets_main_window.extend([
            self.main_window.first_operand_combobox,
            self.main_window.math_operator_combobox,
            self.main_window.second_operand_combobox,
            self.main_window.start_calculation_button
        ])

        self.currently_shown_widgets_main_window = currently_show_widgets_main_window

    def _set_currently_shown_widgets_figure_printer(self):
        currently_show_widgets_main_window = self._get_main_widgets_main_window()
        currently_show_widgets_main_window.extend([
            self.main_window.figure_combobox,
            self.main_window.start_drawing_figure_button
        ])

        self.currently_shown_widgets_main_window = currently_show_widgets_main_window

    def start_text_printing(self):
        self.text_printer.generate_text()

    def start_calculation(self):
        self.calculator.calculate()

    def start_drawing_figure(self):
        self.figure_printer.draw_figure()

    def take_screenshot(self) -> np.ndarray:
        screen = QApplication.primaryScreen()
        window = self.window()
        screenshot = screen.grabWindow(window.winId(), 0, 0).toImage()

        return convert_qimage_to_ndarray(screenshot)

    def get_current_coverage_percentage(self):
        with open(os.devnull, "w") as f:
            try:
                coverage_percentage = self.current_coverage.report(file=f)
            except coverage.exceptions.CoverageException:
                # Is thrown when nothing was ever recorded by the coverage object
                coverage_percentage = 0
        return coverage_percentage

    def calculate_coverage_increase(self):
        new_coverage_percentage = self.get_current_coverage_percentage()
        reward = new_coverage_percentage - self.old_coverage_percentage
        self.old_coverage_percentage = new_coverage_percentage
        return reward

    def execute_mouse_click(self, recv_widget: QWidget, local_pos: QPoint) -> float:
        self.current_coverage.start()
        QTest.mouseClick(recv_widget, Qt.LeftButton, Qt.NoModifier, local_pos)
        self.current_coverage.stop()

        if isinstance(recv_widget, QComboBox):
            logging.debug("Sleeping after click because of combo box")
            time.sleep(0.5)

        reward = self.calculate_coverage_increase()

        return reward

    @Slot(int, int)
    def simulate_click(self, pos_x: int, pos_y: int):
        pos = QPoint(pos_x, pos_y)
        global_pos = self.mapToGlobal(pos)
        logging.debug(f"Received position {pos}, mapped to global position {global_pos}")

        recv_widget = QApplication.widgetAt(global_pos)
        local_pos = recv_widget.mapFromGlobal(global_pos)

        logging.debug(f"Found widget {recv_widget}, mapped to local position {local_pos}")

        reward = self.execute_mouse_click(recv_widget, local_pos)

        screenshot = self.take_screenshot()
        self.observation_signal.emit(reward, screenshot)

    @Slot()
    def simulate_click_on_random_widget(self):
        if self.settings_dialog.isVisible():
            random_widget_list = self.settings_dialog.currently_shown_widgets
        else:
            random_widget_list = self.currently_shown_widgets_main_window

        randomly_selected_widget = self.random_state.choice(random_widget_list)

        height = randomly_selected_widget.height()
        width = randomly_selected_widget.width()

        x = self.random_state.randint(0, width)
        y = self.random_state.randint(0, height)

        local_pos = QPoint(x, y)
        reward = self.execute_mouse_click(randomly_selected_widget, local_pos)

        global_pos = randomly_selected_widget.mapToGlobal(local_pos)
        main_window_pos = self.mapFromGlobal(global_pos)

        logging.debug(f"Randomly selected widget '{randomly_selected_widget}' with local position '{local_pos}', " +
                      f"global position '{global_pos}' and main window position '{main_window_pos}'")

        screenshot = self.take_screenshot()
        self.observation_and_coordinates_signal.emit(reward, screenshot, main_window_pos.x(), main_window_pos.y())

    def generate_html_report(self, directory: str):
        self.current_coverage.html_report(directory=directory)


def main():  # pragma: no cover
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    main()
