from typing import List

from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QComboBox, QCheckBox

from envs.gui_env.src.backend.ascii_art import CHRISTMAS_TREE, GUITAR, SPACE_SHIP, HOUSE


class FigurePrinter:
    def __init__(self, figure_combobox: QComboBox):
        self.figure_combobox = figure_combobox

        self.current_color = "black"

        self.christmas_tree = True
        self.guitar = False
        self.space_ship = False
        self.house = False

        self.update_available_figures()

    def at_least_one_figure(self) -> bool:
        return self.christmas_tree or self.guitar or self.space_ship or self.house

    @Slot(bool)
    def change_christmas_tree(self, checked: bool):
        self.christmas_tree = checked
        self.update_available_figures()

    @Slot(bool)
    def change_guitar(self, checked: bool):
        self.guitar = checked
        self.update_available_figures()

    @Slot(bool)
    def change_space_ship(self, checked: bool):
        self.space_ship = checked
        self.update_available_figures()

    @Slot(bool)
    def change_house(self, checked: bool):
        self.house = checked
        self.update_available_figures()

    @Slot()
    def change_current_color_to_green(self):
        self.current_color = "green"

    @Slot()
    def change_current_color_to_black(self):
        self.current_color = "black"

    @Slot()
    def change_current_color_to_blue(self):
        self.current_color = "blue"

    @Slot()
    def change_current_color_to_brown(self):
        self.current_color = "brown"

    def get_available_figures(self) -> List[str]:
        available_figures = []

        if self.christmas_tree:
            available_figures.append("Christmas Tree")

        if self.guitar:
            available_figures.append("Guitar")

        if self.space_ship:
            available_figures.append("Space Ship")

        if self.house:
            available_figures.append("House")

        return available_figures

    def update_available_figures(self):
        available_figures = self.get_available_figures()
        self.figure_combobox.clear()
        self.figure_combobox.addItems(available_figures)

    @staticmethod
    def get_figure_by_name(figure: str):
        if figure == "Christmas Tree":
            return CHRISTMAS_TREE
        elif figure == "Guitar":
            return GUITAR
        elif figure == "Space Ship":
            return SPACE_SHIP
        elif figure == "House":
            return HOUSE
