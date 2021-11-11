import sys

import PySide6.QtGui
import numpy as np
from PySide6.QtCore import QFile, Qt, QPoint, Slot, Signal
from PySide6.QtGui import QAction
from PySide6.QtGui import QPainter, QPen, QBrush, QImage
from PySide6.QtTest import QTest
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar


def convert_qimage_to_ndarray(image: QImage):
    width = image.width()
    height = image.height()

    data = image.constBits()
    array = np.array(data).reshape((height, width, 4))
    return array


class MainWindow(QMainWindow):
    screenshot_signal = Signal(np.ndarray)
    screenshot_and_coordinates_signal = Signal(np.ndarray, int, int)

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("test-gui-worldmodels")
        self.setFixedSize(600, 600)

        self.main_widget = self.load_ui()
        self.setCentralWidget(self.main_widget)

        self.main_widget.pushButton_3.setVisible(False)

        self.points = []

        self.menu_bar = QMenuBar()
        self.settings_action = QAction("Settings")
        self.menu_bar.addAction(self.settings_action)
        self.setMenuBar(self.menu_bar)

    def take_screenshot(self) -> np.ndarray:
        print("Taking Screenshot!")
        screen = QApplication.primaryScreen()
        window = self.window()
        screenshot = screen.grabWindow(window.winId(), 0, 0).toImage()

        return convert_qimage_to_ndarray(screenshot)

        # screen.grabWindow(window.winId(), 0, 0).save("screenshot_by_screen.png")
        # window.grab().save("screenshot_by_window.png")

        # self.grab().save("screenshot_test.png")

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile("envs/gui_env/src/main_window.ui")
        ui_file.open(QFile.ReadOnly)
        widget = loader.load(ui_file)
        ui_file.close()

        return widget

    @Slot(int, int)
    def simulate_click(self, pos_x: int, pos_y: int):
        print(f"Received emitted signal with pos '{pos_x, pos_y}!")

        pos = QPoint(pos_x, pos_y)
        recv_widget = self.childAt(pos)
        local_pos = recv_widget.mapFrom(self.main_widget, pos)

        QTest.mouseClick(recv_widget, Qt.LeftButton, Qt.NoModifier, local_pos)

        # mouse_event_press = QMouseEvent(QEvent.MouseButtonPress, local_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        # mouse_event_release = QMouseEvent(QEvent.MouseButtonRelease, local_pos, Qt.LeftButton, Qt.LeftButton,
        #                                   Qt.NoModifier)

        # QApplication.sendEvent(recv_widget, mouse_event_press)
        # QApplication.sendEvent(recv_widget, mouse_event_release)
        #
        # print(f"Clicked widget '{recv_widget}' at given pos '{pos}' with calculated local pos '{local_pos}'")
        print(
            f"Received pos {pos_x, pos_y}, clicked widget '{recv_widget}' at local pos '{local_pos}'"
        )

        screenshot = self.take_screenshot()
        self.screenshot_signal.emit(screenshot)

    @Slot()
    def simulate_click_on_random_widget(self):
        print("Received random widget signal")
        screenshot = self.take_screenshot()
        self.screenshot_and_coordinates_signal.emit(screenshot, 99, 123)

    def mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
        self.points.append(event.position())
        print("Added point")
        self.update()

        super().mouseReleaseEvent(event)

    def paintEvent(self, event: PySide6.QtGui.QPaintEvent) -> None:
        # Draws a red point for each click in the MainWindow (not on widgets!)
        qp = QPainter(self)
        pen = QPen(Qt.red, 5)
        brush = QBrush(Qt.red)
        qp.setPen(pen)
        qp.setBrush(brush)

        if self.points is not None:
            print("Drawing current point!")
            qp.drawPoints(self.points)
        qp.end()
        super().paintEvent(event)


if __name__ == "__main__":
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
