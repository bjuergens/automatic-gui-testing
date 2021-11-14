import logging

from PySide6.QtCore import Slot
from PySide6.QtGui import QFont, QTextDocument, QPalette, QColorConstants
from PySide6.QtWidgets import QPlainTextEdit

TEXT_50_WORDS = """
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
"""

TEXT_100_WORDS = """
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing 
elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos 
et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor 
sit amet."""


TEXT_200_WORDS = """
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing 
elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos 
et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor 
sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et 
dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   

Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat 
nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis 
dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet,"""


TEXT_400_WORDS = """
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing 
elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos 
et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor 
sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et 
dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   

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
ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur"""

WORD_COUNT_MAP = {
    50: TEXT_50_WORDS,
    100: TEXT_100_WORDS,
    200: TEXT_200_WORDS,
    400: TEXT_400_WORDS
}

FONT_COLORS = {
    "red": QColorConstants.Red,
    "green": QColorConstants.Green,
    "blue": QColorConstants.Blue,
    "black": QColorConstants.Black
}

FONT_SIZES = [
    12, 14, 16, 18, 20
]


class TextPrinter:

    def __init__(self, output_text_field: QPlainTextEdit):
        self.output_text_field = output_text_field
        self.output_document: QTextDocument = self.output_text_field.document()
        self.output_palette: QPalette = self.output_text_field.palette()
        self.font: QFont = self.output_document.defaultFont()
        self.word_count = list(WORD_COUNT_MAP.keys())[0]

    def apply_settings(self):
        self.output_document.setDefaultFont(self.font)
        self.output_text_field.setPalette(self.output_palette)

    @Slot(str)
    def change_font_size(self, font_size: str):
        font_size = int(font_size)

        self.font.setPointSize(font_size)

    @Slot(QFont)
    def change_font(self, font: QFont):
        is_italic = self.font.italic()
        is_bold = self.font.bold()
        is_underlined = self.font.underline()
        font_size = self.font.pointSize()

        self.font = font

        # Restore old values
        self.change_font_italic(is_italic)
        self.change_font_bold(is_bold)
        self.change_font_underline(is_underlined)
        self.change_font_size(font_size)

    def change_font_color(self, font_color: str):
        try:
            color = FONT_COLORS[font_color]
        except KeyError:
            logging.warning("Invalid color selected in Text Printer, using black as a default")
            color = FONT_COLORS["black"]

        self.output_palette.setColor(QPalette.Text, color)

    @Slot(str)
    def change_word_count(self, word_count: str):
        word_count = int(word_count)
        assert word_count in WORD_COUNT_MAP.keys()
        self.word_count = word_count

    @Slot(bool)
    def change_font_italic(self, checked: bool):
        self.font.setItalic(checked)

    @Slot(bool)
    def change_font_bold(self, checked: bool):
        self.font.setBold(checked)

    @Slot(bool)
    def change_font_underline(self, checked: bool):
        self.font.setUnderline(checked)

    def generate_text(self):
        self.output_text_field.setPlainText(WORD_COUNT_MAP[self.word_count])


