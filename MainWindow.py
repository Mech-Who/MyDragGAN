import datetime
import os
import sys
import random
import json
import threading
from pprint import pprint
import time

from PySide6.QtCore import Signal, Slot, QPoint
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QFileDialog
from PySide6.QtGui import QPainter, QImage

sys.path.append('stylegan2_ada')

from ui.Ui_MainWindow import Ui_DragGAN
from components.LabelStatus import LabelStatus
from components.ConfigMainWindow import ConfigMainWindow
# from model import StyleGAN
from stylegan2_ada.model import StyleGAN
import utils as utils
from metrics.md_metrics import mean_distance

import torch.nn.functional as torch_F
import torch
import copy
import numpy as np
from DragGAN import DragGAN, DragThread


class MainWindow(ConfigMainWindow):

    def __init__(self):
        super().__init__(os.path.join(os.getcwd(), "config.json"))
        self.ui = Ui_DragGAN()
        self.ui.setupUi(self)
        self.setWindowTitle(self.tr("DragGAN"))

        self.DragGAN = DragGAN()

        #### UI初始化 ####
        self.ui.Device_ComboBox.addItem(self.tr("cpu"))
        self.ui.Device_ComboBox.addItem(self.tr("cuda"))
        self.ui.Device_ComboBox.addItem(self.tr("mps"))
        self.ui.Device_ComboBox.setCurrentText(self.device)
        self.ui.Pickle_Label.setText(self.DragGAN.pickle_path)
        self.ui.Seed_LineEdit.setText(str(self.DragGAN.seed))
        self.ui.Seed_LineEdit.setPlaceholderText(f"{self.DragGAN.min_seed} - {self.DragGAN.max_seed}")
        self.ui.RandomSeed_CheckBox.setChecked(self.DragGAN.random_seed)
        self.ui.Wp_CheckBox.setChecked(self.DragGAN.w_plus)
        self.ui.W_CheckBox.setChecked(not self.DragGAN.w_plus)
        self.ui.Radius_LineEdit.setText(str(self.DragGAN.radius))
        self.ui.Lambda_LineEdit.setText(str(self.DragGAN.lambda_))
        self.ui.StepSize_LineEdit.setText(str(self.DragGAN.step_size))
        self.ui.R1_LineEdit.setText(str(self.DragGAN.r1))
        self.ui.R2_LineEdit.setText(str(self.DragGAN.r2))
        self.ui.StepNumber_Label.setText(str(self.DragGAN.steps))
        self.ui.TestTimes_LineEdit.setText(str(self.DragGAN.test_times))
        self.ui.DragTimes_LineEdit.setText(str(self.DragGAN.drag_times))


################### model ##################

    @Slot()
    def on_Device_ComboBox_currentIndexChanged(self):
        device = self.ui.Device_ComboBox.currentText()
        self.DragGAN.setDevice(device)
        print(f"current device: {device}")

    @Slot()
    def on_Recent_PushButton_clicked(self):
        self.DragGAN.pickle_path = self.getConfig("last_pickle")
        self.ui.Pickle_LineEdit.setText(os.path.basename(self.DragGAN.pickle_path))

    @Slot()
    def on_Browse_PushButton_clicked(self):
        file = QFileDialog.getOpenFileName(
            self, "Select Pickle Files", os.path.realpath("./checkpoints"), "Pickle Files (*.pkl)")
        pickle_path = file[0]
        if not os.path.isfile(pickle_path):
            return
        self.DragGAN.pickle_path = pickle_path
        self.ui.Pickle_LineEdit.setText(os.path.basename(pickle_path))
        self.addConfig("last_pickle", pickle_path)

    @Slot()
    def on_Seed_LineEdit_textChanged(self):
        try:
            new_seed = int(self.ui.Seed_LineEdit.text())
            if new_seed <= self.DragGAN.max_seed and new_seed >= self.DragGAN.min_seed:
                self.DragGAN.seed = new_seed
            else:
                self.DragGAN.ui.Seed_LineEdit.setText(str(self.DragGAN.seed))
        except ValueError as e:
            print("invalid seed")

    @Slot()
    def on_Minus4Seed_PushButton_clicked(self):
        self.DragGAN.seed = int(self.ui.Seed_LineEdit.text())
        if self.DragGAN.seed > self.DragGAN.min_seed:
            self.DragGAN.seed -= 1
            self.ui.Seed_LineEdit.setText(str(self.DragGAN.seed))

    @Slot()
    def on_Plus4Seed_PushButton_clicked(self):
        self.DragGAN.seed = int(self.ui.Seed_LineEdit.text())
        if self.DragGAN.seed < self.DragGAN.max_seed:
            self.DragGAN.seed += 1
            self.ui.Seed_LineEdit.setText(str(self.DragGAN.seed))

    @Slot()
    def on_RandomSeed_CheckBox_stateChanged(self):
        if self.ui.RandomSeed_CheckBox.isChecked():
            self.DragGAN.random_seed = True
            self.ui.Plus4Seed_PushButton.setDisabled(True)
            self.ui.Minus4Seed_PushButton.setDisabled(True)
        else:
            self.DragGAN.random_seed = False
            self.ui.Plus4Seed_PushButton.setEnabled(True)
            self.ui.Minus4Seed_PushButton.setEnabled(True)

    @Slot()
    def on_W_CheckBox_stateChanged(self):
        if self.ui.W_CheckBox.isChecked():
            self.DragGAN.w_plus = False
        else:
            self.DragGAN.w_plus = True
        print(f"w current w_plus: {self.DragGAN.w_plus}")

    @Slot()
    def on_Wp_CheckBox_stateChanged(self):
        if self.ui.Wp_CheckBox.isChecked():
            self.DragGAN.w_plus = True
        else:
            self.DragGAN.w_plus = False
        print(f"wp current w_plus: {self.DragGAN.w_plus}")

    @Slot()
    def on_Generate_PushButton_clicked(self):
        print("start generate")
        self.DragGAN.loadCpkt(self.DragGAN.pickle_path)

        if self.DragGAN.random_seed:
            self.DragGAN.seed = random.randint(self.DragGAN.min_seed, self.DragGAN.max_seed)
            self.ui.Seed_LineEdit.setText(str(self.DragGAN.seed))
        image = self.DragGAN.generateImage(self.DragGAN.seed, self.DragGAN.w_plus) # 3 * 512 * 512
        if image is not None:
            self.ui.Image_Widget.set_image_from_array(image)

    @Slot()
    def on_SaveReal_PushButton_clicked(self):
        pickle = os.path.basename(self.DragGAN.pickle_path).split(os.extsep)[0]
        image_format = "png"
        filename = f"{pickle}_{self.DragGAN.seed}.{image_format}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, "save_images", "generated_images")
        filename = os.path.join(image_dir, filename)
        self.ui.Image_Widget.save_image(filename, image_format, 100)
        print(f"save image to {filename}")

################## drag ##################

    @Slot()
    def on_AddPoint_PushButton_clicked(self):
        self.ui.Image_Widget.set_status(LabelStatus.Draw)

    @Slot()
    def on_ResetPoint_PushButton_clicked(self):
        print("reset points")
        self.ui.Image_Widget.clear_points()

    @Slot()
    def on_Start_PushButton_clicked(self):
        print("start drag")
        if self.DragGAN.isDragging:
            QMessageBox.warning(self, "Warning", "Dragging is running!", QMessageBox.Ok)
            return
        self.DragGAN.isDragging = True
        drag_thread = DragThread(self.DragGAN, self.ui.Image_Widget.get_points())
        drag_thread.start()

    @Slot(torch.Tensor, list, int, int)
    def on_drag_finished(self, image, points, loss, steps):
        self.ui.Image_Widget.clear_points()
        self.ui.Image_Widget.add_points(points)
        self.ui.Image_Widget.set_image_from_array(image)
        self.ui.StepNumber_Label.setText(str(steps))
        print(f"step: {steps}, loss: {loss}")

    @Slot()
    def on_Stop_PushButton_clicked(self):
        print("stop drag")
        self.DragGAN.isDragging = False

    @Slot()
    def on_StepSize_LineEdit_editingFinished(self):
        self.DragGAN.step_size = float(self.ui.StepSize_LineEdit.text())
        print(f"current step_size: {self.DragGAN.step_size}")

    @Slot()
    def on_Reset4StepSize_PushButton_clicked(self):
        self.DragGAN.step_size = self.DragGAN.DEFAULT_STEP_SIZE
        self.ui.StepSize_LineEdit.setText(str(self.DragGAN.step_size))

    @Slot()
    def on_R1_LineEdit_editingFinished(self):
        self.DragGAN.r1 = float(self.ui.R1_LineEdit.text())
        print(f"current r1: {self.DragGAN.r1}")

    @Slot()
    def on_Reset4R1_PushButton_clicked(self):
        self.DragGAN.r1 = self.DragGAN.DEFAULT_R1
        self.ui.R1_LineEdit.setText(str(self.DragGAN.r1))

    @Slot()
    def on_R2_LineEdit_editingFinished(self):
        self.DragGAN.r2 = float(self.ui.R2_LineEdit.text())
        print(f"current r2: {self.DragGAN.r2}")

    @Slot()
    def on_Reset4R2_PushButton_clicked(self):
        self.DragGAN.r2 = self.DragGAN.DEFAULT_R2
        self.ui.R2_LineEdit.setText(str(self.DragGAN.r2))

    @Slot()
    def on_SaveGenerate_PushButton_clicked(self):
        pickle = os.path.basename(self.DragGAN.pickle_path).split(os.extsep)[0]
        image_format = "png"
        filename = f"{pickle}_{self.DragGAN.seed}.{image_format}"
        image_dir = os.path.join(os.path.abspath(__file__), "save_images", "edited_images")
        # self.save_image(image_dir+filename, image_format, 100)
        filename = os.path.join(image_dir, filename)
        self.ui.Image_Widget.save_image(filename, image_format, 100)

    @Slot()
    def on_Minus4Radius_PushButton_clicked(self):
        self.DragGAN.radius = float(self.ui.Radius_LineEdit.text())
        self.DragGAN.radius -= 1
        self.ui.Radius_LineEdit.setText(str(self.DragGAN.radius))

    @Slot()
    def on_Plus4Radius_PushButton_clicked(self):
        self.DragGAN.radius = float(self.ui.Radius_LineEdit.text())
        if self.DragGAN.radius < self.DragGAN.max_radius:
            self.DragGAN.radius += 1
            self.ui.Radius_LineEdit.setText(str(self.DragGAN.radius))

    @Slot()
    def on_Minus4Lambda_PushButton_clicked(self):
        self.DragGAN.lambda_ = float(self.ui.Lambda_LineEdit.text())
        if self.DragGAN.lambda_ > self.DragGAN.min_lambda:
            self.DragGAN.lambda_ -= 1
            self.ui.Lambda_LineEdit.setText(str(self.DragGAN.lambda_))

    @Slot()
    def on_Plus4Lambda_PushButton_clicked(self):
        self.DragGAN.lambda_ = float(self.ui.Lambda_LineEdit.text())
        if self.DragGAN.lambda_ < self.DragGAN.max_lambda:
            self.DragGAN.lambda_ += 1
            self.ui.Lambda_LineEdit.setText(str(self.DragGAN.lambda_))

    @Slot()
    def on_FlexibleArea_PushButton_clicked(self):
        print("flexible area")

    @Slot()
    def on_FixedArea_PushButton_clicked(self):
        print("fixed area")

    @Slot()
    def on_ResetMask_PushButton_clicked(self):
        print("reset mask")

################### image ##################


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
