import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Tuple, Union

import gym
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication

from envs.gui_env.src.main_window import MainWindow


class RegisterClickThread(QThread):
    position_signal = Signal(int, int)
    random_widget_signal = Signal()

    def __init__(self, click_connection_child: Connection, terminate_connection_child: Connection):
        super().__init__()

        self.click_connection_child = click_connection_child
        self.terminate_connection_child = terminate_connection_child
        self.connections = [self.click_connection_child, self.terminate_connection_child]

    def run(self) -> None:
        print("Running clicking thread!")

        while True:

            for conn in mp.connection.wait(self.connections):
                if conn == self.terminate_connection_child:
                    print("Terminating clicking!")
                    self.click_connection_child.close()
                    self.terminate_connection_child.close()
                    return

                try:
                    received_data = self.click_connection_child.recv()
                except EOFError:
                    print("Pipe destroyed, exiting!")
                    return
                finally:
                    if isinstance(received_data, Tuple):
                        print("Emitting position!")
                        self.position_signal.emit(received_data[0], received_data[1])
                    elif isinstance(received_data, bool):
                        print("Choosing random widget!")
                        self.random_widget_signal.emit()


class GUIEnv(gym.Env):

    def __init__(self):
        self.click_connection_parent: Connection
        self.click_connection_child: Connection

        self.terminate_connection_parent: Connection
        self.terminate_connection_child: Connection

        self.application_process: Process

        self.main_window: MainWindow

        self._initialize()

    def _initialize(self):
        self.click_connection_parent, self.click_connection_child = Pipe(duplex=True)
        self.terminate_connection_parent, self.terminate_connection_child = Pipe(duplex=True)

        self.application_process = Process(
            target=self._start_application,
            args=(self.click_connection_child, self.terminate_connection_child)
        )

    def _on_timeout(self):
        # Initial observation trigger
        self.main_window.take_screenshot()

    def _start_application(self, click_connection_child: Connection, terminate_connection_child: Connection):
        app = QApplication()

        self.main_window = MainWindow()
        self.main_window.show()

        self.register_click_thread = RegisterClickThread(click_connection_child, terminate_connection_child)
        self.register_click_thread.position_signal.connect(self.main_window.simulate_click)
        self.register_click_thread.random_widget_signal.connect(self.main_window.simulate_click_on_random_widget)
        self.main_window.current_screenshot.connect(self._get_observation)
        self.register_click_thread.start()

        # Send initial observation, but this has to happen after startup, i.e. after app.exec() runs
        QTimer.singleShot(2000, self._on_timeout)

        app.exec()

    @Slot(np.ndarray)
    def _get_observation(self, observation: np.ndarray):
        print("Got observation")
        self.click_connection_child.send(observation)

    def step(self, action: Union[Tuple[int, int], bool]) -> np.ndarray:
        self.click_connection_parent.send(action)
        observation: np.ndarray = self.click_connection_parent.recv()

        return observation

    def reset(self):
        # TODO Cleanup needed to restart an env that previously ran; similar to the code in close(), but calling close
        #  would cleanup internal gym.Env stuff I think so that won't work
        self._initialize()
        self.application_process.start()

        initial_observation = self.click_connection_parent.recv()

        return initial_observation

    def render(self, mode="human"):
        pass

    def close(self):
        self.terminate_connection_parent.send(True)

        # Child connections are closed in child process, i.e. self.application_process
        self.click_connection_parent.close()
        self.terminate_connection_parent.close()

        self.application_process.terminate()
        self.application_process.join()
        self.application_process.close()

        super().close()
