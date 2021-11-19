from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QDialog, QApplication, QGridLayout, QPlainTextEdit, QAbstractButton, QComboBox, QCheckBox

from envs.gui_env.src.backend.calculator import NUMERAL_SYSTEMS, Calculator
from envs.gui_env.src.backend.figure_printer import FigurePrinter
from envs.gui_env.src.backend.text_printer import TextPrinter, FONT_SIZES, WORD_COUNT_MAP
from envs.gui_env.src.utils.utils import load_ui


class SettingsDialog(QDialog):

    figure_printer_activated = Signal(bool)

    def __init__(self, text_printer: TextPrinter, calculator: Calculator, figure_printer: FigurePrinter, **kwargs):
        super().__init__(**kwargs)
        self.settings_dialog = load_ui("envs/gui_env/src/settings_dialog.ui")
        self.layout = QGridLayout()
        self.layout.addWidget(self.settings_dialog, 1, 1)
        self.setLayout(self.layout)

        self.text_printer = text_printer
        self.calculator = calculator
        self.figure_printer = figure_printer

        self._initialize()
        self._connect()

    def _initialize(self):
        # Text Printer
        self.settings_dialog.font_size_combobox.clear()
        self.settings_dialog.font_size_combobox.addItems(str(fs) for fs in FONT_SIZES)
        self.settings_dialog.number_of_words_combobox.clear()
        self.settings_dialog.number_of_words_combobox.addItems(str(wc) for wc in list(WORD_COUNT_MAP.keys()))
        self.settings_dialog.black_text_color_button.toggle()

        # Calculator
        self.settings_dialog.numeral_system_combobox.addItems(numeral_system for numeral_system in NUMERAL_SYSTEMS)

    @Slot(bool)
    def _toggle_figure_printer_settings(self, checked: bool):
        # Activate or deactivate the settings and the main button in the MainWindow
        if checked:
            self.settings_dialog.christmas_tree_checkbox.setEnabled(True)
            self.settings_dialog.guitar_checkbox.setEnabled(True)
            self.settings_dialog.space_ship_checkbox.setEnabled(True)
            self.settings_dialog.house_checkbox.setEnabled(True)

            self.settings_dialog.blue_figure_color_button.setEnabled(True)
            self.settings_dialog.green_figure_color_button.setEnabled(True)
            self.settings_dialog.black_figure_color_button.setEnabled(True)
            self.settings_dialog.brown_figure_color_button.setEnabled(True)
        else:
            self.settings_dialog.christmas_tree_checkbox.setEnabled(False)
            self.settings_dialog.guitar_checkbox.setEnabled(False)
            self.settings_dialog.space_ship_checkbox.setEnabled(False)
            self.settings_dialog.house_checkbox.setEnabled(False)

            self.settings_dialog.blue_figure_color_button.setEnabled(False)
            self.settings_dialog.green_figure_color_button.setEnabled(False)
            self.settings_dialog.black_figure_color_button.setEnabled(False)
            self.settings_dialog.brown_figure_color_button.setEnabled(False)

        self.figure_printer_activated.emit(checked)

    def _connect(self):
        self.settings_dialog.close_settings_dialog.clicked.connect(self.close)

        self._connect_text_printer()
        self._connect_calculator()
        self._connect_figure_printer()

    def _connect_text_printer(self):
        # Text Printer
        self.settings_dialog.bold_font_checkbox.stateChanged.connect(self.text_printer.change_font_bold)
        self.settings_dialog.italic_font_checkbox.stateChanged.connect(self.text_printer.change_font_italic)
        self.settings_dialog.underline_font_checkbox.stateChanged.connect(self.text_printer.change_font_underline)

        # If any of the buttons in the text color button group is clicked, this button is sent to the connected function
        self.settings_dialog.text_color_button_group.buttonClicked.connect(self.change_font_color)

        self.settings_dialog.font_combobox.currentFontChanged.connect(self.text_printer.change_font)
        self.settings_dialog.font_size_combobox.currentTextChanged.connect(self.text_printer.change_font_size)

        self.settings_dialog.number_of_words_combobox.currentTextChanged.connect(self.text_printer.change_word_count)

    def _connect_calculator(self):
        self.settings_dialog.addition_checkbox.stateChanged.connect(self.calculator.change_addition_operator)
        self.settings_dialog.subtraction_checkbox.stateChanged.connect(self.calculator.change_subtraction_operator)
        self.settings_dialog.multiplication_checkbox.stateChanged.connect(
            self.calculator.change_multiplication_operator
        )
        self.settings_dialog.division_checkbox.stateChanged.connect(self.calculator.change_division_operator)

        self.settings_dialog.numeral_system_combobox.currentTextChanged.connect(self.calculator.change_numeral_system)

    def _connect_figure_printer(self):
        # Activate or deactivate the settings and the main buttons
        self.settings_dialog.activate_figure_printer_checkbox.stateChanged.connect(
            self._toggle_figure_printer_settings
        )

        self.settings_dialog.christmas_tree_checkbox.stateChanged.connect(self.figure_printer.change_christmas_tree)
        self.settings_dialog.guitar_checkbox.stateChanged.connect(self.figure_printer.change_guitar)
        self.settings_dialog.space_ship_checkbox.stateChanged.connect(self.figure_printer.change_space_ship)
        self.settings_dialog.house_checkbox.stateChanged.connect(self.figure_printer.change_house)

        self.settings_dialog.green_figure_color_button.clicked.connect(
            self.figure_printer.change_current_color_to_green
        )
        self.settings_dialog.black_figure_color_button.clicked.connect(
            self.figure_printer.change_current_color_to_black
        )
        self.settings_dialog.blue_figure_color_button.clicked.connect(
            self.figure_printer.change_current_color_to_blue
        )
        self.settings_dialog.brown_figure_color_button.clicked.connect(
            self.figure_printer.change_current_color_to_brown
        )

    @Slot(QAbstractButton)
    def change_font_color(self, clicked_button: QAbstractButton):
        if clicked_button == self.settings_dialog.red_text_color_button:
            self.text_printer.change_font_color("red")
        elif clicked_button == self.settings_dialog.green_text_color_button:
            self.text_printer.change_font_color("green")
        elif clicked_button == self.settings_dialog.blue_text_color_button:
            self.text_printer.change_font_color("blue")
        elif clicked_button == self.settings_dialog.black_text_color_button:
            self.text_printer.change_font_color("black")


def main():
    app = QApplication()
    text_printer = TextPrinter(QPlainTextEdit())
    dialog = SettingsDialog(text_printer=text_printer)
    dialog.show()
    app.exec()


if __name__ == '__main__':
    main()
