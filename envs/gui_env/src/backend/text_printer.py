from PySide6.QtCore import Slot, QObject, QEvent
from PySide6.QtGui import QTextDocument, QPalette, QColorConstants, QFontDatabase
from PySide6.QtWidgets import QPlainTextEdit, QAbstractButton, QRadioButton

from envs.gui_env.src.utils.alert_dialogs import ConfirmationDialog

TEXT_50_WORDS = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut 
labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet 
clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
"""  # pragma: no cover

TEXT_100_WORDS = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut 
labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet 
clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur 
sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At 
vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem 
ipsum dolor sit amet."""  # pragma: no cover


TEXT_200_WORDS = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut 
labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet 
clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur 
sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At 
vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem 
ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt 
ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   

Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat 
nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis 
dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet,"""  # pragma: no cover


TEXT_400_WORDS = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut
labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet 
clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur 
sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. 
At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem 
ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt 
ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   

Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat 
nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis 
dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh 
euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   

Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo 
consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu 
feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit 
augue duis dolore te feugait nulla facilisi.   

Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim 
assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet 
dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit 
lobortis nisl ut aliquip ex ea commodo consequat.   

Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat 
nulla facilisis.   

At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem 
ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur"""  # pragma: no cover


WORD_COUNTS = [50, 100, 200, 400]  # pragma: no cover

FONT_SIZES = [12, 14, 16, 18, 20]  # pragma: no cover

FONTS = ["DejaVu Sans", "Liberation Mono", "Noto Sans", "Terminus"]  # pragma: no cover


class TextPrinter:

    def __init__(self, output_text_field: QPlainTextEdit):  # pragma: no cover
        self.output_text_field = output_text_field
        self.output_document: QTextDocument = self.output_text_field.document()
        self.output_palette: QPalette = self.output_text_field.palette()

        self.word_count = 50
        self.font_size = 12
        self.font_name = "DejaVu Sans"
        self.font_color = "black"
        self.bold_font = False
        self.italic_font = False
        self.underline_font = False

    @Slot(str)
    def change_word_count(self, word_count: str):
        if word_count == "50":
            self.word_count = 50
        elif word_count == "100":
            self.word_count = 100
        elif word_count == "200":
            self.word_count = 200
        elif word_count == "400":
            self.word_count = 400

        assert self.word_count in WORD_COUNTS

    @Slot(str)
    def change_font_size(self, font_size: str):
        if font_size == "12":
            self.font_size = 12
        elif font_size == "14":
            self.font_size = 14
        elif font_size == "16":
            self.font_size = 16
        elif font_size == "18":
            self.font_size = 18
        elif font_size == "20":
            self.font_size = 20

        assert self.font_size in FONT_SIZES

    @Slot(str)
    def change_font(self, font: str):
        if font == "DejaVu Sans":
            self.font_name = "DejaVu Sans"
        elif font == "Liberation Mono":
            self.font_name = "Liberation Mono"
        elif font == "Noto Sans":
            self.font_name = "Noto Sans"
        elif font == "Terminus":
            self.font_name = "Terminus"

        assert self.font_name in FONTS

    @Slot(QAbstractButton)
    def change_font_color(self, color_button: QAbstractButton):
        button_text = color_button.text()

        if button_text == "Red":
            self.font_color = "red"
        elif button_text == "Green":
            self.font_color = "green"
        elif button_text == "Blue":
            self.font_color = "blue"
        elif button_text == "Black":
            self.font_color = "black"

        assert self.font_color in ["red", "green", "blue", "black"]

    @Slot(bool)
    def change_font_italic(self, checked: bool):
        if checked:
            self.italic_font = True
        else:
            self.italic_font = False

    @Slot(bool)
    def change_font_bold(self, checked: bool):
        if checked:
            self.bold_font = True
        else:
            self.bold_font = False

    @Slot(bool)
    def change_font_underline(self, checked: bool):
        if checked:
            self.underline_font = True
        else:
            self.underline_font = False

    def generate_text(self):
        text = None
        if self.word_count == 50:
            text = TEXT_50_WORDS
        elif self.word_count == 100:
            text = TEXT_100_WORDS
        elif self.word_count == 200:
            text = TEXT_200_WORDS
        elif self.word_count == 400:
            text = TEXT_400_WORDS
        assert text

        self.output_text_field.setPlainText(text)

        font_size = None
        if self.font_size == 12:
            font_size = 12
        elif self.font_size == 14:
            font_size = 14
        elif self.font_size == 16:
            font_size = 16
        elif self.font_size == 18:
            font_size = 18
        elif self.font_size == 20:
            font_size = 20
        assert font_size

        font_family = None
        if self.font_name == "DejaVu Sans":
            font_family = "DejaVu Sans"
        elif self.font_name == "Liberation Mono":
            font_family = "Liberation Mono"
        elif self.font_name == "Noto Sans":
            font_family = "Noto Sans"
        elif self.font_name == "Terminus":
            font_family = "Terminus"
        assert font_family

        color = None
        if self.font_color == "red":
            color = QColorConstants.Red
        elif self.font_color == "green":
            color = QColorConstants.Green
        elif self.font_color == "blue":
            color = QColorConstants.Blue
        elif self.font_color == "black":
            color = QColorConstants.Black
        assert color

        self.output_palette.setColor(QPalette.Text, color)
        self.output_text_field.setPalette(self.output_palette)

        font = QFontDatabase.font(font_family, "", font_size)

        if self.italic_font:
            font.setItalic(True)

        if self.bold_font:
            font.setBold(True)

        if self.underline_font:
            font.setUnderline(True)

        self.output_document.setDefaultFont(font)


class GreenColorEventFilter(QObject):

    def __init__(self, green_button: QRadioButton, **kwargs):  # pragma: no cover
        super().__init__(**kwargs)
        self.green_button = green_button

    def eventFilter(self, obj: QObject, event: QEvent):
        if event.type() == QEvent.MouseButtonPress and not self.green_button.isChecked():
            confirmation_dialog = ConfirmationDialog(
                "Do you really want to set the text color to green?",
                parent=self.parent()
            )

            def accept():
                self.green_button.click()

            def decline():
                assert True  # Technically nothing has to be done, but we want to increase code coverage

            confirmation_dialog.dialog.accept_button.clicked.connect(accept)
            confirmation_dialog.dialog.decline_button.clicked.connect(decline)

            # Dialog is modal so an accept or decline needs to happen before the program can continue
            confirmation_dialog.show()

            # Returning True indicates that the event has been handled and shall not be passed on
            return True

        # Ignore all other events
        return False
