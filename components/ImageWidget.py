from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import Qt, QSize, QPoint, QEvent

import os


class ImageWidget(QWidget):
    """
    图片自适应 QWidget (通过QLabel显式)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.installEventFilter(self)
        self.image_rate = None
        self.last_pos = None

        self.painter = QPainter()

    def set_image(self, file_name):
        try:
            pix_map = QPixmap(file_name)
            self.image_rate = pix_map.width() / pix_map.height()
            self.image_label.setPixmap(pix_map)
            self.compute_size()
        except:
            pass

    def compute_size(self):
        if self.image_rate is not None:
            w = self.size().width()
            h = self.size().height()
            scale_w = int(h * self.image_rate)

            if scale_w <= w:
                self.image_label.resize(QSize(scale_w, h))
                self.image_label.setProperty(
                    "pos", QPoint(int((w - scale_w) / 2), 0))
            else:
                scale_h = int(w / self.image_rate)
                self.image_label.resize(QSize(w, scale_h))
                self.image_label.setProperty(
                    "pos", QPoint(0, int((h - scale_h) / 2)))
        else:
            self.image_label.resize(self.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.compute_size()


if __name__ == "__main__":
    app = QApplication([])
    # widget = QWidget()
    # widget.show()
    image = ImageWidget()
    image.set_image(os.path.realpath("./components/dog.jpg"))
    image.show()
    app.exec()
