from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import Qt, QSize, QPoint

import os
from pprint import pprint


class ImageLabel(QLabel):
    """
    直接继承自QLabel, 尝试显示图片并绘制点
    """

    def __init__(self, file_name=None):
        super().__init__()

        self.setScaledContents(True)

        if file_name:
            self.set_image(file_name)

        self.painter = QPainter()
        self.pen = QPen()
        self.pen.setWidth(3)
        self.pen.setColor(QColor(0, 0, 255))  # 设置画笔颜色为蓝色
        self.last_pos = None

    def set_image(self, file_name):
        try:
            print(file_name)
            self.setPixmap(QPixmap(file_name))
        except Exception as e:
            QMessageBox.critical(self, "Error", "Image loading failed!")
            print(f"Error: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        pass


if __name__ == "__main__":
    app = QApplication([])
    # widget = QWidget()
    # widget.show()
    image = ImageLabel(os.path.realpath("./components/dog.jpg"))
    image.show()
    app.exec()
