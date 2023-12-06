from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtCore import QSize, QPoint
from components.ImageLabel import ImageLabel
from PIL import Image, ImageQt
import os

from pprint import pprint

class ImageWidget(QWidget):
    """
    图片自适应 QWidget (通过QLabel显式)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_label = ImageLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.installEventFilter(self)
        self.image_rate = None
        self.last_pos = None

        self.painter = QPainter()

    def set_image(self, file_name):
        try:
            self.pixmap = QPixmap(file_name)
            pix_map = self.pixmap
            self.image_rate = pix_map.width() / pix_map.height()
            self.image_label.setPixmap(pix_map)
            self.compute_size()
        except Exception as e:
            QMessageBox.critical(self, "Error", "Load image failed!")
            print(e)

    def set_image_from_array(self, image):
        # try:
            # q_image = QImage(   image.data, 
            #                             image.shape[1], 
            #                             image.shape[0], 
            #                             image.shape[1]*3, 
            #                             QImage.Format_RGB888
            #                             )
        # pprint(image)
        # pprint(len(image))

        # height, width, channel = image.shape
        # bytes_per_line = 3 * width

        # qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        img = image.cpu().numpy()
        img = Image.fromarray(img)
        # pix_map = QPixmap.fromImage(qimage)
        pix_map = img.toqpixmap()
        self.image_rate = pix_map.width() / pix_map.height()
        self.image_label.setPixmap(pix_map)
        self.compute_size()
        # except Exception as e:
        #     QMessageBox.critical(self, "Error", "Load image failed!")
        #     print(e)

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

    def save_image(self, filename, format, quality):
        """
        pixmap.save()方法是用于将QPixmap对象保存为文件的方法，它可以写入多种不同的文件格式，具体支持的格式取决于Qt库和操作系统的支持情况。
        在大多数情况下，Qt库支持的格式包括：
        1. BMP格式：使用"bmp"或"wbmp"文件扩展名
        2. GIF格式：使用"gif"文件扩展名
        3. JPEG格式：使用"jpg"、"jpeg"或"jpe"文件扩展名
        4. PNG格式：使用"png"文件扩展名
        5. PBM格式：使用"pbm"文件扩展名
        6. PGM格式：使用"pgm"文件扩展名
        7. PPM格式：使用"ppm"文件扩展名
        8. XBM格式：使用"xbm"文件扩展名
        9. XPM格式：使用"xpm"文件扩展名
        """
        self.pixmap.save(filename, format, quality)

    def set_status(self, new_status):
        self.image_label.set_status(new_status)

    def get_points(self):
        return self.image_label.get_points()

    def add_points(self, points):
        self.image_label.add_points(points)

    def clear_points(self):
        self.image_label.clear_points()

if __name__ == "__main__":
    app = QApplication([])
    # widget = QWidget()
    # widget.show()
    image = ImageWidget()
    image.set_image(os.path.realpath("./components/dog.jpg"))
    image.show()
    app.exec()
