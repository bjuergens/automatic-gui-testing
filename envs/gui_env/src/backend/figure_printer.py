from PySide6.QtCore import Slot
from PySide6.QtGui import QColorConstants, QColor, QPalette
from PySide6.QtWidgets import QComboBox, QAbstractButton, QPlainTextEdit

from envs.gui_env.src.backend.ascii_art import CHRISTMAS_TREE, GUITAR, SPACE_SHIP, HOUSE
from envs.gui_env.src.utils.utils import SignalHandler


class FigurePrinter:
    def __init__(self, figure_output_field: QPlainTextEdit, figure_combobox: QComboBox):
        self.figure_output_field = figure_output_field
        self.figure_combobox = figure_combobox

        self.signal_handler = SignalHandler()

        self.christmas_tree = True
        self.guitar = False
        self.space_ship = False
        self.house = False
        self.color = "black"

        self.update_available_figures()

    @Slot(bool)
    def change_christmas_tree(self, checked: bool):
        if checked:
            self.christmas_tree = True
        else:
            self.christmas_tree = False
        self.update_available_figures()

    @Slot(bool)
    def change_guitar(self, checked: bool):
        if checked:
            self.guitar = True
        else:
            self.guitar = False
        self.update_available_figures()

    @Slot(bool)
    def change_space_ship(self, checked: bool):
        if checked:
            self.space_ship = True
        else:
            self.space_ship = False
        self.update_available_figures()

    @Slot(bool)
    def change_house(self, checked: bool):
        if checked:
            self.house = True
        else:
            self.house = False
        self.update_available_figures()

    @Slot(QAbstractButton)
    def change_color(self, color_button: QAbstractButton):
        button_text = color_button.text()

        if button_text == "Green":
            self.color = "green"
        elif button_text == "Blue":
            self.color = "blue"
        elif button_text == "Black":
            self.color = "black"
        elif button_text == "Brown":
            self.color = "brown"

        assert self.color in ["green", "blue", "black", "brown"]

    def update_available_figures(self):
        self.figure_combobox.clear()

        available_figures = []
        if self.christmas_tree:
            available_figures.append("Christmas Tree")

        if self.guitar:
            available_figures.append("Guitar")

        if self.space_ship:
            available_figures.append("Space Ship")

        if self.house:
            available_figures.append("House")

        if not available_figures:
            # TODO implement this dialog
            self.signal_handler.all_figures_deselected.emit()

        self.figure_combobox.addItems(available_figures)

    def draw_figure(self):
        color = None
        if self.color == "green":
            color = QColorConstants.Green
        elif self.color == "blue":
            color = QColorConstants.Blue
        elif self.color == "black":
            color = QColorConstants.Black
        elif self.color == "brown":
            color = QColor("#8b4513")
        assert color is not None

        palette: QPalette = self.figure_output_field.palette()
        palette.setColor(QPalette.Text, color)
        self.figure_output_field.setPalette(palette)

        figure = None
        figure_txt = self.figure_combobox.currentText()
        if figure_txt == "Christmas Tree":
            figure = CHRISTMAS_TREE
        elif figure_txt == "Guitar":
            figure = GUITAR
        elif figure_txt == "Space Ship":
            figure = SPACE_SHIP
        elif figure_txt == "House":
            figure = HOUSE
        assert figure is not None

        self.figure_output_field.setPlainText(figure)
