import logging
from functools import partial

from PySide6.QtCore import Slot, Signal, Qt
from PySide6.QtWidgets import QDialog, QApplication, QGridLayout

from envs.gui_env.src.backend.calculator import (NUMERAL_SYSTEMS, Calculator, show_missing_operators_error,
                                                 show_division_by_zero_error)
from envs.gui_env.src.backend.car_configurator import CarConfigurator
from envs.gui_env.src.backend.figure_printer import (FigurePrinter, show_missing_figures_error,
                                                     toggle_figure_printer_settings)
from envs.gui_env.src.backend.text_printer import TextPrinter, WORD_COUNTS, FONT_SIZES, FONTS, GreenColorEventFilter
from envs.gui_env.src.utils.utils import load_ui


class SettingsDialog(QDialog):

    figure_printer_activated = Signal(bool)  # pragma: no cover

    def __init__(self, text_printer: TextPrinter, calculator: Calculator, car_configurator: CarConfigurator,
                 figure_printer: FigurePrinter,
                 **kwargs):  # pragma: no cover
        super().__init__(**kwargs)

        self.setWindowFlag(Qt.FramelessWindowHint, True)

        self.settings_dialog = load_ui("envs/gui_env/src/settings_dialog.ui")
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.settings_dialog, 1, 1)
        self.setLayout(self.layout)

        # Disables possible clicks outside the dialog, and keeps the dialog always on top until it is closed
        self.setModal(True)

        self.text_printer = text_printer
        self.calculator = calculator
        self.car_configurator = car_configurator
        self.figure_printer = figure_printer

        self.currently_shown_widgets = []
        self._set_currently_shown_widgets_text_printer_settings()

        self._initialize()
        self._connect()

    def _initialize(self):
        # Text Printer
        self.settings_dialog.number_of_words_combobox.clear()
        self.settings_dialog.number_of_words_combobox.addItems(str(wc) for wc in WORD_COUNTS)
        self.settings_dialog.font_size_combobox.clear()
        self.settings_dialog.font_size_combobox.addItems(str(fs) for fs in FONT_SIZES)
        self.settings_dialog.font_combobox.clear()
        self.settings_dialog.font_combobox.addItems(f for f in FONTS)

        self.settings_dialog.black_text_color_button.toggle()

        # Calculator
        self.settings_dialog.numeral_system_combobox.addItems(numeral_system for numeral_system in NUMERAL_SYSTEMS)

        # Car Configurator
        # Used for layout purposes
        size_policy = self.settings_dialog.dummy_checkbox_electric_motor.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.settings_dialog.dummy_checkbox_electric_motor.setSizePolicy(size_policy)
        self.settings_dialog.dummy_checkbox_electric_motor.setVisible(False)
        self.settings_dialog.dummy_checkbox_interior.setSizePolicy(size_policy)
        self.settings_dialog.dummy_checkbox_interior.setVisible(False)

    def _connect(self):
        self.settings_dialog.close_settings_dialog.clicked.connect(self.close)

        self._connect_text_printer()
        self._connect_calculator()
        self._connect_car_configurator()
        self._connect_figure_printer()

        self.settings_dialog.settings_tab.currentChanged.connect(self._tab_changed)

    def _connect_text_printer(self):
        # Number of Words
        self.settings_dialog.number_of_words_combobox.currentTextChanged.connect(self.text_printer.change_word_count)

        # Font
        self.settings_dialog.font_size_combobox.currentTextChanged.connect(self.text_printer.change_font_size)
        self.settings_dialog.font_combobox.currentTextChanged.connect(self.text_printer.change_font)
        # If any of the buttons in the text color button group is clicked, this button is sent to the connected function
        self.settings_dialog.text_color_button_group.buttonClicked.connect(self.text_printer.change_font_color)

        # Font formats
        self.settings_dialog.italic_font_checkbox.stateChanged.connect(self.text_printer.change_font_italic)
        self.settings_dialog.bold_font_checkbox.stateChanged.connect(self.text_printer.change_font_bold)
        self.settings_dialog.underline_font_checkbox.stateChanged.connect(self.text_printer.change_font_underline)

        green_click_filter = GreenColorEventFilter(self.settings_dialog.green_text_color_button,
                                                   parent=self.settings_dialog)
        self.settings_dialog.green_text_color_button.installEventFilter(green_click_filter)

    def _connect_calculator(self):
        self.settings_dialog.addition_checkbox.stateChanged.connect(self.calculator.change_addition_operator)
        self.settings_dialog.subtraction_checkbox.stateChanged.connect(self.calculator.change_subtraction_operator)
        self.settings_dialog.multiplication_checkbox.stateChanged.connect(
            self.calculator.change_multiplication_operator
        )
        self.settings_dialog.division_checkbox.stateChanged.connect(self.calculator.change_division_operator)

        self.settings_dialog.numeral_system_combobox.currentTextChanged.connect(self.calculator.change_numeral_system)

        self.calculator.signal_handler.division_by_zero_occured.connect(partial(show_division_by_zero_error, self))
        self.calculator.signal_handler.all_operators_deselected.connect(partial(show_missing_operators_error, self))

    def _connect_car_configurator(self):
        self.settings_dialog.tire_18_inch_checkbox.stateChanged.connect(self.car_configurator.change_18_inch_tire)
        self.settings_dialog.tire_19_inch_checkbox.stateChanged.connect(self.car_configurator.change_19_inch_tire)
        self.settings_dialog.tire_20_inch_checkbox.stateChanged.connect(self.car_configurator.change_20_inch_tire)
        self.settings_dialog.tire_22_inch_checkbox.stateChanged.connect(self.car_configurator.change_22_inch_tire)

        self.settings_dialog.modern_interior_checkbox.stateChanged.connect(self.car_configurator.change_modern_interior)
        self.settings_dialog.vintage_interior_checkbox.stateChanged.connect(
            self.car_configurator.change_vintage_interior
        )
        self.settings_dialog.sport_interior_checkbox.stateChanged.connect(self.car_configurator.change_sport_interior)

        self.settings_dialog.combustion_engine_a_checkbox.stateChanged.connect(
            self.car_configurator.change_combustion_engine_a
        )
        self.settings_dialog.combustion_engine_b_checkbox.stateChanged.connect(
            self.car_configurator.change_combustion_engine_b
        )
        self.settings_dialog.combustion_engine_c_checkbox.stateChanged.connect(
            self.car_configurator.change_combustion_engine_c
        )
        self.settings_dialog.electric_motor_a_checkbox.stateChanged.connect(
            self.car_configurator.change_electric_motor_a
        )
        self.settings_dialog.electric_motor_b_checkbox.stateChanged.connect(
            self.car_configurator.change_electric_motor_b
        )

    def _connect_figure_printer(self):
        # Activate or deactivate the settings and the main buttons
        self.settings_dialog.activate_figure_printer_checkbox.stateChanged.connect(
            partial(toggle_figure_printer_settings, self)
        )

        self.settings_dialog.christmas_tree_checkbox.stateChanged.connect(self.figure_printer.change_christmas_tree)
        self.settings_dialog.guitar_checkbox.stateChanged.connect(self.figure_printer.change_guitar)
        self.settings_dialog.space_ship_checkbox.stateChanged.connect(self.figure_printer.change_space_ship)
        self.settings_dialog.house_checkbox.stateChanged.connect(self.figure_printer.change_house)

        self.settings_dialog.tree_color_button_group.buttonClicked.connect(self.figure_printer.change_color)

        self.figure_printer.signal_handler.all_figures_deselected.connect(partial(show_missing_figures_error, self))

    @Slot(int)
    def _tab_changed(self, tab: int):
        logging.debug(f"Settings tab changed to '{tab}'")
        if tab == 0:
            self._set_currently_shown_widgets_text_printer_settings()
        elif tab == 1:
            self._set_currently_shown_widgets_calculator_settings()
        elif tab == 2:
            self._set_currently_shown_widgets_car_configurator_settings()
        elif tab == 3:
            self.set_currently_shown_widgets_figure_printer_settings()

    def _get_always_shown_widgets(self):
        currently_shown_widgets = [
            self.settings_dialog.close_settings_dialog,
            self.settings_dialog.settings_tab.tabBar()
        ]
        return currently_shown_widgets

    def _set_currently_shown_widgets_text_printer_settings(self):
        currently_shown_widgets = self._get_always_shown_widgets()

        currently_shown_widgets.extend([
            self.settings_dialog.number_of_words_combobox,
            self.settings_dialog.font_size_combobox,
            self.settings_dialog.font_combobox,
            self.settings_dialog.red_text_color_button,
            self.settings_dialog.green_text_color_button,
            self.settings_dialog.blue_text_color_button,
            self.settings_dialog.black_text_color_button,
            self.settings_dialog.bold_font_checkbox,
            self.settings_dialog.italic_font_checkbox,
            self.settings_dialog.underline_font_checkbox
        ])

        self.currently_shown_widgets = currently_shown_widgets

    def _set_currently_shown_widgets_calculator_settings(self):
        currently_shown_widgets = self._get_always_shown_widgets()

        currently_shown_widgets.extend([
            self.settings_dialog.addition_checkbox,
            self.settings_dialog.multiplication_checkbox,
            self.settings_dialog.subtraction_checkbox,
            self.settings_dialog.division_checkbox,
            self.settings_dialog.numeral_system_combobox
        ])

        self.currently_shown_widgets = currently_shown_widgets

    def _set_currently_shown_widgets_car_configurator_settings(self):
        currently_shown_widgets = self._get_always_shown_widgets()

        currently_shown_widgets.extend([
            self.settings_dialog.tire_20_inch_checkbox,
            self.settings_dialog.tire_22_inch_checkbox,
            self.settings_dialog.tire_18_inch_checkbox,
            self.settings_dialog.tire_19_inch_checkbox,
            self.settings_dialog.modern_interior_checkbox,
            self.settings_dialog.vintage_interior_checkbox,
            self.settings_dialog.sport_interior_checkbox,
            self.settings_dialog.combustion_engine_a_checkbox,
            self.settings_dialog.combustion_engine_b_checkbox,
            self.settings_dialog.combustion_engine_c_checkbox,
            self.settings_dialog.electric_motor_a_checkbox,
            self.settings_dialog.electric_motor_b_checkbox,
        ])

        self.currently_shown_widgets = currently_shown_widgets

    def set_currently_shown_widgets_figure_printer_settings(self):
        currently_shown_widgets = self._get_always_shown_widgets()

        currently_shown_widgets.append(self.settings_dialog.activate_figure_printer_checkbox)

        if self.settings_dialog.activate_figure_printer_checkbox.isChecked():
            currently_shown_widgets.extend([
                self.settings_dialog.christmas_tree_checkbox,
                self.settings_dialog.guitar_checkbox,
                self.settings_dialog.space_ship_checkbox,
                self.settings_dialog.house_checkbox,
                self.settings_dialog.green_figure_color_button,
                self.settings_dialog.blue_figure_color_button,
                self.settings_dialog.black_figure_color_button,
                self.settings_dialog.brown_figure_color_button
            ])

        self.currently_shown_widgets = currently_shown_widgets


def main():  # pragma: no cover
    from PySide6.QtWidgets import QPlainTextEdit, QLCDNumber, QComboBox, QFrame, QPushButton
    app = QApplication()
    text_printer = TextPrinter(QPlainTextEdit())
    calculator = Calculator(QLCDNumber(), QComboBox(), QComboBox(), QComboBox())
    car_configurator = CarConfigurator(QFrame(), QComboBox(), QFrame(), QComboBox(), QFrame(), QComboBox(), QFrame(),
                                       QComboBox(), QPushButton())
    figure_printer = FigurePrinter(QPlainTextEdit(), QComboBox())
    dialog = SettingsDialog(text_printer=text_printer, calculator=calculator, car_configurator=car_configurator,
                            figure_printer=figure_printer)
    dialog.show()
    app.exec()


if __name__ == '__main__':  # pragma: no cover
    main()
