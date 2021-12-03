import logging
import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Tuple

import gym
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication

from envs.gui_env.src.main_window import MainWindow, WINDOW_SIZE


class RegisterClickThread(QThread):
    position_signal = Signal(int, int)
    random_widget_signal = Signal()
    generate_html_report_signal = Signal()

    def __init__(self, click_connection_child: Connection, terminate_connection_child: Connection,
                 html_report_connection_child: Connection = None):
        super().__init__()
        self.click_connection_child = click_connection_child
        self.terminate_connection_child = terminate_connection_child
        self.html_report_connection_child = html_report_connection_child
        self.connections = [self.click_connection_child, self.terminate_connection_child]

        self.generate_html_report = False
        if self.html_report_connection_child is not None:
            self.generate_html_report = True

    def run(self) -> None:
        logging.debug("Clicking Thread: Starting thread")
        while True:

            for conn in mp.connection.wait(self.connections):
                if conn == self.terminate_connection_child:
                    if self.generate_html_report:
                        self.generate_html_report_signal.emit()
                        # Wait for the report to be generated
                        self.html_report_connection_child.recv()

                    logging.debug("Clicking Thread: Stopping thread gracefully")
                    self.terminate_connection_child.send(True)
                    return

                try:
                    received_data = self.click_connection_child.recv()
                except EOFError:
                    logging.debug("Clicking Thread: Pipe was destroyed, exiting!")
                    return
                finally:
                    if isinstance(received_data, Tuple):
                        self.position_signal.emit(received_data[0], received_data[1])
                    elif isinstance(received_data, bool):
                        self.random_widget_signal.emit()


class GUIEnv(gym.Env):

    def __init__(self, generate_html_report: bool = False):
        self.generate_html_report = generate_html_report

        self.click_connection_parent: Connection
        self.click_connection_child: Connection

        self.terminate_connection_parent: Connection
        self.terminate_connection_child: Connection

        self.application_process: Process

        self.main_window: MainWindow

        self._initialize()

        self.random_state = np.random.RandomState()

    def _initialize(self):
        self.click_connection_parent, self.click_connection_child = Pipe(duplex=True)
        self.terminate_connection_parent, self.terminate_connection_child = Pipe(duplex=True)
        if self.generate_html_report:
            logging.info("Enabled HTML report generation")
            self.html_report_connection_parent, self.html_report_connection_child = Pipe(duplex=True)
        else:
            self.html_report_connection_parent, self.html_report_connection_child = None, None

        self.application_process = Process(
            target=self._start_application,
            args=(self.click_connection_child, self.terminate_connection_child, self.html_report_connection_child)
        )

    def _on_timeout(self):
        # Initial observation trigger
        screenshot = self.main_window.take_screenshot()
        self.main_window.observation_signal.emit(0, screenshot)

    def _start_application(self, click_connection_child: Connection, terminate_connection_child: Connection,
                           html_report_connection_child: Connection = None):
        app = QApplication()

        self.main_window = MainWindow()
        self.main_window.show()

        self.register_click_thread = RegisterClickThread(click_connection_child, terminate_connection_child,
                                                         html_report_connection_child)

        # Connect click thread signals to main window
        self.register_click_thread.position_signal.connect(self.main_window.simulate_click)
        self.register_click_thread.random_widget_signal.connect(self.main_window.simulate_click_on_random_widget)
        self.register_click_thread.generate_html_report_signal.connect(self._generate_html_report)

        # Connect main window observation signals to this process
        self.main_window.observation_signal.connect(self._get_observation)
        self.main_window.observation_and_coordinates_signal.connect(self._get_observation_random_widget)
        self.register_click_thread.start()

        # Send initial observation, but this has to happen after startup, i.e. after app.exec() runs
        QTimer.singleShot(2000, self._on_timeout)

        app.exec()

    @Slot(float, np.ndarray)
    def _get_observation(self, reward: float, observation: np.ndarray):
        self.click_connection_child.send((reward, observation))

    @Slot(float, np.ndarray, int, int)
    def _get_observation_random_widget(self, reward: float, observation: np.ndarray, pos_x: int, pos_y: int):
        self.click_connection_child.send((reward, observation, pos_x, pos_y))

    @Slot()
    def _generate_html_report(self):
        clicker_type = self.get_clicker_type()

        directory = os.path.join("coverage-reports", clicker_type, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        self.main_window.generate_html_report(directory=directory)
        self.html_report_connection_parent.send(True)

    @staticmethod
    def get_clicker_type():
        return "gui-env"

    def step(self, action: Tuple[int, int]) -> Tuple[float, np.ndarray, bool, dict]:
        self.click_connection_parent.send(action)
        reward, observation = self.click_connection_parent.recv()
        return reward, observation, False, {}

    def reset(self):
        # TODO Cleanup needed to restart an env that previously ran; similar to the code in close(), but calling close
        #  would cleanup internal gym.Env stuff I think so that won't work
        self._initialize()
        self.application_process.start()

        _, initial_observation = self.click_connection_parent.recv()

        return initial_observation

    def render(self, mode="human"):
        pass

    def close(self):
        logging.debug("Sending close indication to clicking thread")
        self.terminate_connection_parent.send(True)
        self.terminate_connection_parent.recv()

        self.application_process.terminate()
        self.application_process.join()
        self.application_process.close()

        logging.debug("Closed application process, closing environment now")

        super().close()

    def seed(self, seed=None):
        self.random_state = np.random.RandomState(seed)
        return super().seed(seed)


class GUIEnvRandomClick(GUIEnv):

    def step(self, action: bool = None) -> Tuple[float, np.ndarray, bool, dict]:
        x = self.random_state.randint(0, WINDOW_SIZE[0])
        y = self.random_state.randint(0, WINDOW_SIZE[1])

        self.click_connection_parent.send((x, y))
        reward, observation = self.click_connection_parent.recv()

        return reward, observation, False, {"x": x, "y": y}

    @staticmethod
    def get_clicker_type():
        return "random-clicks"


class GUIEnvRandomWidget(GUIEnv):

    def __init__(self, random_click_probability: float, **kwargs):
        super().__init__(**kwargs)
        self.random_click_probability = random_click_probability

    def step(self, action: bool = None) -> Tuple[float, np.ndarray, bool, dict]:
        if self.random_state.rand() < self.random_click_probability:
            logging.debug("Selecting random click")
            # Random click
            x = self.random_state.randint(0, WINDOW_SIZE[0])
            y = self.random_state.randint(0, WINDOW_SIZE[1])

            self.click_connection_parent.send((x, y))
            reward, observation = self.click_connection_parent.recv()

            return reward, observation, False, {"x": x, "y": y}
        else:
            logging.debug("Selecting random widget")
            # Random widget
            self.click_connection_parent.send(True)
            reward, observation, x, y = self.click_connection_parent.recv()

            return reward, observation, False, {"x": x, "y": y}

    @staticmethod
    def get_clicker_type():
        return "random-widgets"
