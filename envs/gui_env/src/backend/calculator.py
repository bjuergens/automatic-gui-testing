import logging

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLCDNumber, QComboBox

POSSIBLE_OPERANDS_BASE_10 = [i for i in range(1, 5)]
POSSIBLE_OPERANDS_BASE_2 = [bin(i) for i in range(1, 5)]
POSSIBLE_OPERANDS_BASE_16 = [hex(i) for i in range(1, 5)]

NUMERAL_SYSTEMS = ["Base 10", "Base 2", "Base 16"]


class Calculator:

    def __init__(self, calculator_output: QLCDNumber, first_operand_combobox: QComboBox,
                 second_operand_combobox: QComboBox, math_operator_combobox: QComboBox):
        self.calculator_output = calculator_output
        self.first_operand_combobox = first_operand_combobox
        self.second_operand_combobox = second_operand_combobox
        self.math_operator_combobox = math_operator_combobox
        self.numeral_system = "Base 10"

        self.addition_operator = True
        self.subtraction_operator = True
        self.multiplication_operator = False
        self.division_operator = False

        self._initialize()

    def _initialize(self):
        self._initialize_operands()
        self._initialize_operators()

    def _initialize_operands(self):
        self.first_operand_combobox.clear()
        self.second_operand_combobox.clear()

        if self.numeral_system == "Base 10":
            self.first_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_10)
            self.second_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_10)
        elif self.numeral_system == "Base 2":
            self.first_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_2)
            self.second_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_2)
        elif self.numeral_system == "Base 16":
            self.first_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_16)
            self.second_operand_combobox.addItems(str(i) for i in POSSIBLE_OPERANDS_BASE_16)

    def _initialize_operators(self):
        self.math_operator_combobox.clear()

        operators = []
        if self.addition_operator:
            operators.append("+")
        if self.subtraction_operator:
            operators.append("-")
        if self.multiplication_operator:
            operators.append("*")
        if self.division_operator:
            operators.append("/")

        self.math_operator_combobox.addItems(operators)

    @Slot(str)
    def change_numeral_system(self, numeral_system: str):
        self.numeral_system = numeral_system
        self._initialize_operands()

    @Slot(bool)
    def change_addition_operator(self, checked: bool):
        self.addition_operator = checked
        self._initialize_operators()

    @Slot(bool)
    def change_subtraction_operator(self, checked: bool):
        self.subtraction_operator = checked
        self._initialize_operators()

    @Slot(bool)
    def change_multiplication_operator(self, checked: bool):
        self.multiplication_operator = checked
        self._initialize_operators()

    @Slot(bool)
    def change_division_operator(self, checked: bool):
        self.division_operator = checked
        self._initialize_operators()

    def _convert(self, number, output=False):
        if self.numeral_system == "Base 10":
            return int(number)
        elif self.numeral_system == "Base 2":
            if not output:
                return int(number, 2)
            else:
                return bin(int(number))
        elif self.numeral_system == "Base 16":
            if not output:
                return int(number, 16)
            else:
                return hex(int(number))

    def calculate(self):
        a = self._convert(self.first_operand_combobox.currentText())
        b = self._convert(self.second_operand_combobox.currentText())
        operator = self.math_operator_combobox.currentText()

        if operator == "+":
            output = a + b
        elif operator == "-":
            output = a - b
        elif operator == "*":
            output = a * b
        elif operator == "/":
            output = a / b
        else:
            logging.warning("Calculator received unknown operator, displaying a 0")
            output = 0

        self.calculator_output.display(self._convert(output, True))
