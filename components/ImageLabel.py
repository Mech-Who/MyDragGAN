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
        self.points = []

    def set_image(self, file_name):
        try:
            print(file_name)
            self.setPixmap(QPixmap(file_name))
        except Exception as e:
            QMessageBox.critical(self, "Error", "Image loading failed!")
            print(f"Error: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(event.position())
            self.update()

    def get_points(self):
        return self.points
    
    def clear_points(self):
        self.points = []

    def mouseReleaseEvent(self, event):
        pass

    def paintEvent(self, event):
        res = super().paintEvent(event)
        if self.points:
            self.painter.begin(self)
            for index, pos in enumerate(self.points):
                if index % 2 != 0:
                    self.pen.setColor(QColor(255, 0, 0))  # 设置画笔颜色为红色
                else:
                    self.pen.setColor(QColor(0, 0, 255))  # 设置画笔颜色为蓝色
                self.painter.setPen(self.pen)
                self.painter.drawPoint(pos)
            self.painter.end()
        return res


if __name__ == "__main__":
    app = QApplication([])
    # widget = QWidget()
    # widget.show()
    image = ImageLabel(os.path.realpath("./components/dog.jpg"))
    image.show()
    app.exec()
