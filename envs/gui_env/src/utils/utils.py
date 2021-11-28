import numpy as np
from PySide6.QtCore import QFile, QObject, Signal
from PySide6.QtGui import QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget


class SignalHandler(QObject):  # pragma: no cover
    division_by_zero_occured = Signal()
    all_operators_deselected = Signal()
    all_figures_deselected = Signal()
    green_color_text_printer_selected = Signal()


def load_ui(ui_file: str, parent_widget: QWidget = None) -> QWidget:
    loader = QUiLoader()
    ui_file = QFile(ui_file)
    ui_file.open(QFile.ReadOnly)
    loaded_widget = loader.load(ui_file, parent_widget)
    ui_file.close()

    return loaded_widget


def convert_qimage_to_ndarray(image: QImage):
    width = image.width()
    height = image.height()

    data = image.constBits()

    # Discard Alpha channel at the end
    array = np.array(data).reshape((height, width, 4))[:, :, :3]
    return array
