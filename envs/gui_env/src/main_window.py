import logging
import os
import sys
from functools import partial
from typing import List, Union, Tuple

import coverage.exceptions
import numpy as np
from PySide6.QtCore import Qt, QPoint, Slot, Signal
from PySide6.QtGui import QAction, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QWidget, QComboBox
from coverage import Coverage

from envs.gui_env.src.backend.calculator import Calculator
from envs.gui_env.src.backend.car_configurator import (CarConfigurator, show_disabled_cars_error_dialog,
                                                       show_car_configuration_dialog)
from envs.gui_env.src.backend.figure_printer import FigurePrinter, toggle_figure_printer_widgets
from envs.gui_env.src.backend.text_printer import TextPrinter
from envs.gui_env.src.settings_dialog import SettingsDialog
from envs.gui_env.src.utils.paint_event_filter import PaintEventFilter
from envs.gui_env.src.utils.utils import load_ui, do_nothing_function
from envs.gui_env.window_configuration import WINDOW_SIZE


class MainWindow(QMainWindow):
    observation_and_coordinates_signal = Signal(float, np.ndarray, int, int)  # pragma: no cover

    def __init__(self, coverage_measurer: Coverage, paint_event_filter: PaintEventFilter,
                 random_click_probability: float = None, random_seed: int = None, **kwargs):  # pragma: no cover
        super().__init__(**kwargs)

        self.setWindowTitle("test-gui-worldmodels")
        self.setFixedSize(*WINDOW_SIZE)
        self.setWindowFlag(Qt.FramelessWindowHint, True)

        self.main_window = load_ui("envs/gui_env/src/main_window.ui")

        self._initialize()

        self.currently_shown_widgets_main_window = []
        # Initially we start with these widgets
        self._set_currently_shown_widgets_text_printer()

        self.text_printer = TextPrinter(self.main_window.text_printer_output)
        self.calculator = Calculator(self.main_window.calculator_output, self.main_window.first_operand_combobox,
                                     self.main_window.second_operand_combobox, self.main_window.math_operator_combobox)
        self.figure_printer = FigurePrinter(self.main_window.figure_printer_output, self.main_window.figure_combobox)
        self.car_configurator = CarConfigurator(
            self.main_window.car_model_selection_frame, self.main_window.car_model_selection_combobox,
            self.main_window.tire_selection_frame, self.main_window.tire_selection_combobox,
            self.main_window.interior_design_frame, self.main_window.interior_design_combobox,
            self.main_window.propulsion_system_frame, self.main_window.propulsion_system_combobox,
            self.main_window.show_configuration_button
        )

        self.settings_dialog = SettingsDialog(text_printer=self.text_printer, calculator=self.calculator,
                                              car_configurator=self.car_configurator,
                                              figure_printer=self.figure_printer, parent=self.main_window)

        self._connect_buttons()

        self.setCentralWidget(self.main_window)

        self.coverage_measurer = coverage_measurer
        self.old_coverage_percentage = 0

        self.paint_event_filter = paint_event_filter

        self.random_click_probability = random_click_probability
        self.random_state = np.random.RandomState(random_seed)

        # Keep track of an open combo box to close it when clicked somewhere else
        self.open_combobox = None

        self.i = 0

    def _initialize(self):
        # Initialize menu bar
        self.menu_bar = QMenuBar(parent=self)
        self.settings_action = QAction("Settings")
        self.menu_bar.addAction(self.settings_action)
        self.setMenuBar(self.menu_bar)

        # Text Printer

        # The inner widget of the text printer output is called 'qt_scrollarea_viewport', but this will interfere with
        # some hacks that are done later when clicking on combo boxes. Therefore we need to change the name here
        self.main_window.text_printer_output.findChild(QWidget, "qt_scrollarea_viewport").setObjectName(
            "scrollarea_text_printer"
        )

        # Car Configurator

        # Set top label to bold, to function as a headline
        font = self.main_window.car_configurator_headline_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.main_window.car_configurator_headline_label.setFont(font)

        # Figure Printer

        # Same reason to do this, as with the text printer output a few lines above
        self.main_window.figure_printer_output.findChild(QWidget, "qt_scrollarea_viewport").setObjectName(
            "scrollarea_figure_printer"
        )

        # Figure Printer is hidden at first, must be activated in the settings
        self.main_window.figure_printer_button.setVisible(False)

        # Need a monospace font to display the ASCII art correctly
        document = self.main_window.figure_printer_output.document()
        font = QFontDatabase.font("Bitstream Vera Sans Mono", "Normal", 6)
        document.setDefaultFont(font)

    def _connect_buttons(self):
        # Text Printer
        self.main_window.text_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 0)
        )
        self.main_window.text_printer_button.clicked.connect(self._set_currently_shown_widgets_text_printer)
        self.main_window.start_text_printer_button.clicked.connect(self.text_printer.generate_text)

        # Calculator
        self.main_window.calculator_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 1)
        )
        self.main_window.calculator_button.clicked.connect(self._set_currently_shown_widgets_calculator)
        self.main_window.start_calculation_button.clicked.connect(self.calculator.calculate)

        # Car Configurator
        self.main_window.car_configurator_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 2)
        )
        self.main_window.car_configurator_button.clicked.connect(self._set_currently_shown_widgets_car_configurator)

        self.car_configurator.signal_handler.disabled_cars.connect(partial(show_disabled_cars_error_dialog, self))
        self.car_configurator.signal_handler.car_configured.connect(partial(show_car_configuration_dialog, self))
        self.car_configurator.signal_handler.changed_active_car_configurator_widgets.connect(
            self._change_currently_shown_widgets_car_configurator
        )

        # Figure Printer
        self.main_window.figure_printer_button.clicked.connect(
            partial(self.main_window.main_stacked_widget.setCurrentIndex, 3)
        )
        self.main_window.figure_printer_button.clicked.connect(self._set_currently_shown_widgets_figure_printer)
        self.main_window.start_drawing_figure_button.clicked.connect(self.figure_printer.draw_figure)

        # Settings action and dialog
        self.settings_action.triggered.connect(self.settings_dialog.show)
        self.settings_dialog.figure_printer_activated.connect(partial(toggle_figure_printer_widgets, self))

    def _get_main_widgets_main_window(self):
        currently_shown_widgets_main_window = [
            self.main_window.text_printer_button,
            self.main_window.calculator_button,
            self.main_window.car_configurator_button,
            self.settings_action  # This is technically not a QWidget but a QAction, but the click function handles it
        ]

        if self.main_window.figure_printer_button.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.figure_printer_button)

        return currently_shown_widgets_main_window

    def _set_currently_shown_widgets_text_printer(self):
        currently_shown_widgets_main_window = self._get_main_widgets_main_window()
        currently_shown_widgets_main_window.append(self.main_window.start_text_printer_button)

        self.currently_shown_widgets_main_window = currently_shown_widgets_main_window

    def _set_currently_shown_widgets_calculator(self):
        currently_shown_widgets_main_window = self._get_main_widgets_main_window()
        currently_shown_widgets_main_window.extend([
            self.main_window.first_operand_combobox,
            self.main_window.math_operator_combobox,
            self.main_window.second_operand_combobox,
            self.main_window.start_calculation_button
        ])

        self.currently_shown_widgets_main_window = currently_shown_widgets_main_window

    def _set_currently_shown_widgets_car_configurator(self):
        currently_shown_widgets_main_window = self._get_main_widgets_main_window()
        currently_shown_widgets_main_window.append(self.main_window.car_model_selection_combobox)

        if self.main_window.tire_selection_frame.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.tire_selection_combobox)

        if self.main_window.interior_design_frame.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.interior_design_combobox)

        if self.main_window.propulsion_system_frame.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.propulsion_system_combobox)

        if self.main_window.show_configuration_button.isVisible():
            currently_shown_widgets_main_window.append(self.main_window.show_configuration_button)

        self.currently_shown_widgets_main_window = currently_shown_widgets_main_window

    @Slot(object)
    def _change_currently_shown_widgets_car_configurator(self, widget_list: List[Tuple[bool, QWidget]]):
        for is_visible, widget in widget_list:
            if is_visible and widget not in self.currently_shown_widgets_main_window:
                self.currently_shown_widgets_main_window.append(widget)
                continue

            if not is_visible and widget in self.currently_shown_widgets_main_window:
                self.currently_shown_widgets_main_window.remove(widget)

        # If the current main widget is the car configurator change the current active widgets here, since we maybe
        # return to the initial state where only the car can be chosen
        if self.main_window.main_stacked_widget.currentIndex() == 2:
            self._set_currently_shown_widgets_car_configurator()

    def _set_currently_shown_widgets_figure_printer(self):
        currently_shown_widgets_main_window = self._get_main_widgets_main_window()
        currently_shown_widgets_main_window.extend([
            self.main_window.figure_combobox,
            self.main_window.start_drawing_figure_button
        ])

        self.currently_shown_widgets_main_window = currently_shown_widgets_main_window

    def get_current_coverage_percentage(self):
        with open(os.devnull, "w") as f:
            try:
                coverage_percentage = self.coverage_measurer.report(file=f)
            except coverage.exceptions.CoverageException:
                # Is thrown when nothing was ever recorded by the coverage object
                coverage_percentage = 0
        return coverage_percentage

    def calculate_coverage_increase(self):
        new_coverage_percentage = self.get_current_coverage_percentage()
        reward = new_coverage_percentage - self.old_coverage_percentage
        self.old_coverage_percentage = new_coverage_percentage
        return reward

    def execute_mouse_click(self, recv_widget: Union[QWidget, QAction], local_pos: QPoint) -> Tuple[float, bool]:
        """
        QTest.mouseClick has some problems and works only for QWidgets, therefore we test for some conditions before
        executing the click.

        1. If a combo box is open, we want to close it, therefore we take either the click in the combo box, which will
        select an item and closes it or hidePopup which simulates a click outside the combo box and closes it
        2. If no combo box is open we check if a modal window is active. If this is the case we can only click inside
        it, because other windows of the application cannot be clicked (because of the modality). If the click is
        outside the modal window (this is checked with the isAncestorOf function), then we set the click function to
        an essential "None function" which does nothing. Otherwise we use the QTest.mouseClick function or
         QAction.trigger
        """

        if isinstance(recv_widget, QAction):
            # QAction do not work with QTest.mouseClick, as this function only works with QWidgets
            click_function = recv_widget.trigger
            logging.debug(f"{self.i}: Set click function to QAction.trigger")
        else:
            click_function = partial(QTest.mouseClick, recv_widget, Qt.LeftButton, Qt.NoModifier, local_pos)
            logging.debug(f"{self.i}: Set click function to QTest.mouseClick")

        closed_combobox = False
        # Use this to check which click function should be used and later if additional delay is needed
        current_active_modal_widget = QApplication.activeModalWidget()

        # If we have an open combo box and click somewhere else in the window, the combo box must be closed.
        # QTest.mouseClick() ignores this unfortunately, therefore we have to manually close it
        if self.open_combobox is not None:
            # If this next condition is true, a click in the combo box is planned. For some reason testing for a widget
            # in an open combo box does not return the combo box but a QWidget with the name "qt_scrollarea_viewport".
            # This is probably because the open QComboBox uses a viewport.
            # Only if this is not the case we know that we clicked outside the combo box and therefore the combo box
            # should be closed, which we do with the hidePopup() function
            if recv_widget.objectName() != "qt_scrollarea_viewport":
                logging.debug(f"{self.i}: Set click function to hidePopup")
                click_function = self.open_combobox.hidePopup

            self.open_combobox = None
            closed_combobox = True
        else:
            # First test if a modal window is active, if this is the case only clicks in the modal window are allowed
            # Unfortunately QTest.mouseClick() would ignore this
            if current_active_modal_widget is not None:
                # This checks if the recv_widget is part of the modal window, if this is the case it is allowed to be
                # clicked
                is_ancestor = current_active_modal_widget.isAncestorOf(recv_widget)

                if not is_ancestor and recv_widget.objectName() != "qt_scrollarea_viewport":
                    click_function = do_nothing_function
                    logging.debug(f"{self.i}: Is not an ancestor, set click function to do nothing function")

        self.coverage_measurer.start()
        click_function()
        self.coverage_measurer.stop()

        if isinstance(recv_widget, QComboBox) and not closed_combobox:
            # If recv_widget is a QComboBox and we did not close a previously opened combo box, then we know that this
            # widget has to be a combo box that has just been opened.
            self.open_combobox = recv_widget

        reward = self.calculate_coverage_increase()

        increased_delay: bool = (
                closed_combobox
                or self.open_combobox is not None
                or isinstance(recv_widget, QAction)
                or isinstance(recv_widget, QMenuBar)
                or current_active_modal_widget != QApplication.activeModalWidget()  # Indicates a newly opened or closed
                                                                                    # modal widget
        )

        return reward, increased_delay

    def simulate_click(self, pos_x: int, pos_y: int) -> Tuple[float, bool]:
        pos = QPoint(pos_x, pos_y)
        global_pos = self.mapToGlobal(pos)
        logging.debug(f"{self.i}: Received position {pos}, mapped to global position {global_pos}")

        recv_widget = QApplication.widgetAt(global_pos)
        local_pos = recv_widget.mapFromGlobal(global_pos)

        logging.debug(f"{self.i}: Found widget {recv_widget}, mapped to local position {local_pos}")

        reward, increased_delay = self.execute_mouse_click(recv_widget, local_pos)

        self.i += 1

        return reward, increased_delay

    def simulate_click_on_random_widget(self) -> Tuple[float, int, int, bool]:
        current_active_modal_widget = QApplication.activeModalWidget()

        if current_active_modal_widget is not None:
            random_widget_list = current_active_modal_widget.currently_shown_widgets
        else:
            random_widget_list = self.currently_shown_widgets_main_window

        randomly_selected_widget = self.random_state.choice(random_widget_list)

        if isinstance(randomly_selected_widget, QComboBox) and self.open_combobox is not None:
            if randomly_selected_widget == self.open_combobox:
                # Actual clicks on an opened QComboBox happen in its view(), specifically in a child widget that has the
                # object name "qt_scrollarea_viewport". Therefore, replace the found widget with its child widget, so a
                # click can happen in an opened QComboBox
                randomly_selected_widget = self.open_combobox.view().viewport()

        if isinstance(randomly_selected_widget, QAction):
            # QAction does not have the width() and height() functions as a QWidget does, therefore use workarounds
            action_rectangle = self.menu_bar.actionGeometry(randomly_selected_widget)
            width = action_rectangle.width()
            height = action_rectangle.height()
        else:
            width = randomly_selected_widget.width()
            height = randomly_selected_widget.height()

        x = self.random_state.randint(0, width)
        y = self.random_state.randint(0, height)

        local_pos = QPoint(x, y)

        if isinstance(randomly_selected_widget, QAction):
            global_pos = self.menu_bar.mapToGlobal(local_pos)
        else:
            global_pos = randomly_selected_widget.mapToGlobal(local_pos)

        found_widget_at_point = QApplication.widgetAt(global_pos)

        if not found_widget_at_point == randomly_selected_widget and not isinstance(randomly_selected_widget, QAction):
            new_local_pos = found_widget_at_point.mapFromGlobal(global_pos)
            logging.debug(
                f"NOT EQUAL: Found widget {found_widget_at_point} at click point where randomly selected widget " +
                f"{randomly_selected_widget} was set to be clicked. Compare old local_pos {local_pos} with " +
                f"new {new_local_pos}"
            )
            randomly_selected_widget = found_widget_at_point
            local_pos = new_local_pos

        reward, increased_delay = self.execute_mouse_click(randomly_selected_widget, local_pos)

        main_window_pos = self.mapFromGlobal(global_pos)
        logging.debug(f"{self.i}: Randomly selected widget '{randomly_selected_widget}' with local " +
                      f"position '{local_pos}', global position '{global_pos}' and main window " +
                      f"position '{main_window_pos}'")

        self.i += 1

        return reward, main_window_pos.x(), main_window_pos.y(), increased_delay

    def generate_html_report(self, directory: str):
        try:
            self.coverage_measurer.html_report(directory=directory, precision=4)
        except coverage.exceptions.CoverageException:
            logging.debug("Did not create an HTML report because nothing was measured")


def main():  # pragma: no cover
    from coverage import Coverage

    app = QApplication([])
    coverage_measurer = Coverage()
    widget = MainWindow(coverage_measurer, None)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    main()
