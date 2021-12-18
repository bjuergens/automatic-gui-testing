import logging
import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Tuple, Union

import gym
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, QTimer, Qt, QElapsedTimer
from PySide6.QtWidgets import QApplication
from coverage import Coverage

from envs.gui_env.src.utils.paint_event_filter import PaintEventFilter
from envs.gui_env.src.utils.utils import take_screenshot
from envs.gui_env.window_configuration import WINDOW_SIZE


LAST_STEP_TIMEOUT = 100
LAST_STEP_TIMEOUT_ADDITIONAL_DELAY = 300
LAST_PAINT_EVENT_TIMEOUT = 100
LAST_PAINT_EVENT_TIMEOUT_ADDITIONAL_DELAY = 300


class RegisterClickThread(QThread):
    position_signal = Signal(int, int)
    random_widget_signal = Signal()
    generate_html_report_signal = Signal()

    def __init__(self, paint_event_filter: PaintEventFilter, window_id, click_connection_child: Connection,
                 terminate_connection_child: Connection, screenshot_connection_child: Connection,
                 generate_html_report: bool = False):
        super().__init__()
        self.paint_event_filter = paint_event_filter
        self.window_id = window_id

        self.click_connection_child = click_connection_child
        self.terminate_connection_child = terminate_connection_child
        self.screenshot_connection_child = screenshot_connection_child

        self.connections = [self.click_connection_child, self.terminate_connection_child,
                            self.screenshot_connection_child]

        self.generate_html_report = generate_html_report

        self.last_step_timer = QElapsedTimer()
        self.last_step_timer.start()

        self.current_last_step_timeout = LAST_STEP_TIMEOUT_ADDITIONAL_DELAY

    def run(self) -> None:
        logging.debug("Clicking Thread: Starting thread")
        while True:
            for conn in mp.connection.wait(self.connections):
                if conn == self.terminate_connection_child:
                    assert conn.recv()
                    if self.generate_html_report:
                        # Signal is connected to block here, so the html report can be created fully and then the thread
                        # continues
                        self.generate_html_report_signal.emit()

                    logging.debug("Clicking Thread: Stopping thread gracefully")
                    self.terminate_connection_child.send(True)
                    return
                elif conn == self.click_connection_child:
                    while not self.last_step_timer.hasExpired(self.current_last_step_timeout):
                        QThread.msleep(25)

                    try:
                        received_data = conn.recv()
                    except EOFError:
                        logging.debug("Clicking Thread: Pipe was destroyed, exiting!")
                        return

                    if isinstance(received_data, Tuple):
                        self.position_signal.emit(received_data[0], received_data[1])
                    else:
                        self.random_widget_signal.emit()
                elif conn == self.screenshot_connection_child:
                    increased_delay = conn.recv()

                    if increased_delay:
                        self.current_last_step_timeout = LAST_STEP_TIMEOUT_ADDITIONAL_DELAY
                        last_paint_event_timeout = LAST_PAINT_EVENT_TIMEOUT_ADDITIONAL_DELAY
                    else:
                        self.current_last_step_timeout = LAST_STEP_TIMEOUT
                        last_paint_event_timeout = LAST_PAINT_EVENT_TIMEOUT

                    last_paint_event_timer = self.paint_event_filter.last_paint_event_timer

                    while not last_paint_event_timer.hasExpired(last_paint_event_timeout):
                        QThread.msleep(25)
                    QThread.msleep(last_paint_event_timeout)

                    screenshot = take_screenshot(self.window_id)
                    self.screenshot_connection_child.send(screenshot)

                    self.last_step_timer.restart()


class GUIEnv(gym.Env):

    def __init__(self, generate_html_report: bool = False, html_report_directory: str = None):
        self.generate_html_report = generate_html_report
        self.html_report_directory = html_report_directory

        self.click_connection_parent, self.click_connection_child = None, None
        self.terminate_connection_parent, self.terminate_connection_child = None, None
        self.screenshot_connection_parent, self.screenshot_connection_child = None, None

        self.application_process: Process

        self.random_state = np.random.RandomState()

    def _initialize(self):
        self.click_connection_parent, self.click_connection_child = Pipe(duplex=True)
        self.terminate_connection_parent, self.terminate_connection_child = Pipe(duplex=True)
        self.screenshot_connection_parent, self.screenshot_connection_child = Pipe(duplex=True)

        if self.generate_html_report:
            logging.info("Enabled HTML report generation")

        self.application_process = Process(
            target=self._start_application,
            args=(self.click_connection_child, self.terminate_connection_child, self.screenshot_connection_child,
                  self.generate_html_report)
        )

        self.application_process.start()

    def _on_timeout(self):
        # Initial observation trigger
        screenshot = take_screenshot(self.main_window.window().winId())
        self.screenshot_connection_child.send(screenshot)

    def _start_application(self, click_connection_child: Connection, terminate_connection_child: Connection,
                           screenshot_connection_child: Connection, generate_html_report: bool):

        # data_suffix appends process id to the database file which is needed when this environment is run in parallel
        coverage_measurer = Coverage(data_suffix=True, config_file="envs/gui_env/.coveragerc")
        coverage_measurer.start()
        from envs.gui_env.src.main_window import MainWindow
        coverage_measurer.stop()

        self.paint_event_filter = PaintEventFilter()
        app = QApplication()
        app.installEventFilter(self.paint_event_filter)

        self.main_window = MainWindow(coverage_measurer, self.paint_event_filter)
        self.main_window.show()

        self.register_click_thread = RegisterClickThread(self.paint_event_filter, self.main_window.window().winId(),
                                                         click_connection_child, terminate_connection_child,
                                                         screenshot_connection_child, generate_html_report)

        # Connect click thread signals to main window
        self.register_click_thread.position_signal.connect(self._simulate_click,
                                                           type=Qt.BlockingQueuedConnection)
        self.register_click_thread.random_widget_signal.connect(self._simulate_click_on_random_widget,
                                                                type=Qt.BlockingQueuedConnection)
        self.register_click_thread.generate_html_report_signal.connect(self._generate_html_report,
                                                                       type=Qt.BlockingQueuedConnection)

        # Connect main window observation signals to this process
        self.register_click_thread.start()

        # Send initial observation, but this has to happen after startup, i.e. after app.exec() runs
        QTimer.singleShot(2000, self._on_timeout)

        app.exec()

    def _stop_application_process(self):
        logging.debug("Sending close indication to clicking thread")
        self.terminate_connection_parent.send(True)
        self.terminate_connection_parent.recv()

        self.application_process.terminate()
        self.application_process.join()
        self.application_process.close()

    @Slot(int, int)
    def _simulate_click(self, pos_x: int, pos_y: int):
        reward, increased_delay = self.main_window.simulate_click(pos_x, pos_y)
        self.click_connection_child.send((reward, increased_delay))

    @Slot()
    def _simulate_click_on_random_widget(self):
        reward, pos_x, pos_y, increased_delay = self.main_window.simulate_click_on_random_widget()
        self.click_connection_child.send((reward, pos_x, pos_y, increased_delay))

    @Slot()
    def _generate_html_report(self):
        if self.html_report_directory is not None:
            directory = self.html_report_directory
        else:
            clicker_type = self.get_clicker_type()
            directory = os.path.join("coverage-reports", clicker_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.main_window.generate_html_report(directory=directory)

    @staticmethod
    def get_clicker_type():
        return "gui-env"

    def sample_random_coordinates(self) -> Tuple[int, int]:
        x = self.random_state.randint(0, WINDOW_SIZE[0])
        y = self.random_state.randint(0, WINDOW_SIZE[1])

        return x, y

    def internal_step(self, action: Union[Tuple[int, int], bool]) -> Tuple[float, np.ndarray, bool, dict]:
        self.click_connection_parent.send(action)

        if isinstance(action, bool):
            reward, x, y, increased_delay = self.click_connection_parent.recv()
        else:
            reward, increased_delay = self.click_connection_parent.recv()
            x = action[0]
            y = action[1]

        info = {"x": x, "y": y}

        self.screenshot_connection_parent.send(increased_delay)
        observation = self.screenshot_connection_parent.recv()

        return reward, observation, False, info

    def step(self, action: Tuple[int, int]) -> Tuple[float, np.ndarray, bool, dict]:
        reward, observation, done, info = self.internal_step(action)

        return reward, observation, done, info

    def reset(self):
        if self.terminate_connection_parent is not None:
            self._stop_application_process()

        self._initialize()

        initial_observation = self.screenshot_connection_parent.recv()

        return initial_observation

    def render(self, mode="human"):
        pass

    def close(self):
        self._stop_application_process()

        logging.debug("Closed application process, closing environment now")

        super().close()

    def seed(self, seed=None):
        self.random_state = np.random.RandomState(seed)
        return super().seed(seed)


class GUIEnvRandomClick(GUIEnv):

    def step(self, action: bool = None) -> Tuple[float, np.ndarray, bool, dict]:
        x, y = self.sample_random_coordinates()

        reward, observation, done, info = self.internal_step((x, y))

        return reward, observation, done, info

    @staticmethod
    def get_clicker_type():
        return "random-clicks"


class GUIEnvRandomWidget(GUIEnv):

    def __init__(self, random_click_probability: float = 0.125, **kwargs):
        super().__init__(**kwargs)
        self.random_click_probability = random_click_probability

    def step(self, action: bool = None) -> Tuple[float, np.ndarray, bool, dict]:
        if self.random_state.rand() < self.random_click_probability:
            # Random click
            logging.debug("Selecting random click")
            x, y = self.sample_random_coordinates()

            reward, observation, done, info = self.internal_step((x, y))

            return reward, observation, done, info
        else:
            # Random widget
            logging.debug("Selecting random widget")

            # Info contains the selected x and y coordinates
            reward, observation, done, info = self.internal_step(True)

            return reward, observation, done, info

    @staticmethod
    def get_clicker_type():
        return "random-widgets"
