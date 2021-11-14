from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDialog, QApplication, QGridLayout, QPlainTextEdit, QAbstractButton, QComboBox

from envs.gui_env.src.backend.calculator import NUMERAL_SYSTEMS, Calculator
from envs.gui_env.src.backend.text_printer import TextPrinter, FONT_SIZES, WORD_COUNT_MAP
from envs.gui_env.src.utils.utils import load_ui


class SettingsDialog(QDialog):

    def __init__(self, text_printer: TextPrinter, calculator: Calculator, **kwargs):
        super().__init__(**kwargs)
        self.settings_dialog = load_ui("envs/gui_env/src/settings_dialog.ui")
        self.layout = QGridLayout()
        self.layout.addWidget(self.settings_dialog, 1, 1)
        self.setLayout(self.layout)

        self.text_printer = text_printer
        self.calculator = calculator

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

    def _connect(self):
        self.settings_dialog.close_settings_dialog.clicked.connect(self.close)

        self._connect_text_printer()
        self._connect_calculator()

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
