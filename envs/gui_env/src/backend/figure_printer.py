from PySide6.QtCore import Slot
from PySide6.QtGui import QColorConstants, QColor, QPalette
from PySide6.QtWidgets import QComboBox, QAbstractButton, QPlainTextEdit

from envs.gui_env.src.backend.ascii_art import CHRISTMAS_TREE, GUITAR, SPACE_SHIP, HOUSE
from envs.gui_env.src.utils.alert_dialogs import MissingContentDialog
from envs.gui_env.src.utils.utils import SignalHandler


class FigurePrinter:
    def __init__(self, figure_output_field: QPlainTextEdit, figure_combobox: QComboBox):  # pragma: no cover
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


@Slot(bool)
def toggle_figure_printer_widgets(main_window, checked: bool):
    if checked:
        main_window.main_window.figure_printer_button.setVisible(True)
        main_window.main_window.figure_printer_button.setEnabled(True)
        if main_window.main_window.figure_printer_button not in main_window.currently_shown_widgets_main_window:
            main_window.currently_shown_widgets_main_window.append(main_window.main_window.figure_printer_button)
    else:
        main_window.main_window.figure_printer_button.setVisible(False)
        main_window.main_window.figure_printer_button.setEnabled(False)
        try:
            main_window.currently_shown_widgets_main_window.remove(main_window.main_window.figure_printer_button)
        except ValueError:  # pragma: no cover
            pass

        # Could be that the stacked widget is still on the figure printer but we deactivate it, therefore simply
        # switch back to the first index
        if main_window.main_window.main_stacked_widget.currentIndex() == 3:
            main_window.main_window.main_stacked_widget.setCurrentIndex(0)


@Slot(bool)
def toggle_figure_printer_settings(settings_dialog, checked: bool):
    # Activate or deactivate the settings and the main button in the MainWindow
    if checked:
        settings_dialog.settings_dialog.christmas_tree_checkbox.setEnabled(True)
        settings_dialog.settings_dialog.guitar_checkbox.setEnabled(True)
        settings_dialog.settings_dialog.space_ship_checkbox.setEnabled(True)
        settings_dialog.settings_dialog.house_checkbox.setEnabled(True)

        settings_dialog.settings_dialog.blue_figure_color_button.setEnabled(True)
        settings_dialog.settings_dialog.green_figure_color_button.setEnabled(True)
        settings_dialog.settings_dialog.black_figure_color_button.setEnabled(True)
        settings_dialog.settings_dialog.brown_figure_color_button.setEnabled(True)
    else:
        settings_dialog.settings_dialog.christmas_tree_checkbox.setEnabled(False)
        settings_dialog.settings_dialog.guitar_checkbox.setEnabled(False)
        settings_dialog.settings_dialog.space_ship_checkbox.setEnabled(False)
        settings_dialog.settings_dialog.house_checkbox.setEnabled(False)

        settings_dialog.settings_dialog.blue_figure_color_button.setEnabled(False)
        settings_dialog.settings_dialog.green_figure_color_button.setEnabled(False)
        settings_dialog.settings_dialog.black_figure_color_button.setEnabled(False)
        settings_dialog.settings_dialog.brown_figure_color_button.setEnabled(False)

    settings_dialog.set_currently_shown_widgets_figure_printer_settings()
    settings_dialog.figure_printer_activated.emit(checked)


@Slot()
def show_missing_figures_error(settings_dialog):
    missing_figures_dialog = MissingContentDialog(
        warning_text="All Figures have been deselected, please choose at least one:",
        content=["Christmas Tree", "Guitar", "Space Ship", "House"],
        parent=settings_dialog.settings_dialog
    )

    @Slot()
    def set_figure():
        chosen_figure = missing_figures_dialog.dialog.content_combobox.currentText()
        checkbox = None
        if chosen_figure == "Christmas Tree":
            checkbox = settings_dialog.settings_dialog.christmas_tree_checkbox
        elif chosen_figure == "Guitar":
            checkbox = settings_dialog.settings_dialog.guitar_checkbox
        elif chosen_figure == "Space Ship":
            checkbox = settings_dialog.settings_dialog.space_ship_checkbox
        elif chosen_figure == "House":
            checkbox = settings_dialog.settings_dialog.house_checkbox
        assert checkbox is not None

        checkbox.setChecked(True)

    missing_figures_dialog.dialog.close_button.clicked.connect(set_figure)
    missing_figures_dialog.show()
