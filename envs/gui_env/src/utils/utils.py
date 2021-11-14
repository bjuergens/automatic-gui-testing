import numpy as np
from PySide6.QtCore import QFile
from PySide6.QtGui import QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget


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
    array = np.array(data).reshape((height, width, 4))
    return array
