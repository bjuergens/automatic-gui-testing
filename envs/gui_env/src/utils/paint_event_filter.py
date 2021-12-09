from PySide6.QtCore import QObject, QEvent, QElapsedTimer


class PaintEventFilter(QObject):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.last_paint_event_timer = QElapsedTimer()

    def eventFilter(self, obj: QObject, event: QEvent):
        if event.type() == QEvent.Paint:
            # Paint event occurred, restart the timer
            self.last_paint_event_timer.restart()

        # Always return false, indicating that we did not handle the event. We only want to know when the last paint
        # event occurred
        return False
