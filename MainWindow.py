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
from DragGAN import DragGAN, DragThread, ExperienceThread


class MainWindow(ConfigMainWindow):

    def __init__(self):
        super().__init__(os.path.join(os.getcwd(), "config.json"))
        self.ui = Ui_DragGAN()
        self.ui.setupUi(self)
        self.setWindowTitle(self.tr("DragGAN"))

        self.DragGAN = DragGAN()
        self.run_thread = None

        #### UI初始化 ####
        self.ui.Device_ComboBox.addItem(self.tr("cpu"))
        self.ui.Device_ComboBox.addItem(self.tr("cuda"))
        self.ui.Device_ComboBox.addItem(self.tr("mps"))
        self.ui.Device_ComboBox.setCurrentText(self.DragGAN.device)
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
        self.ui.R3_LineEdit.setText(str(self.DragGAN.r3))
        self.ui.StepNumber_Label.setText(str(self.DragGAN.steps))
        self.ui.TestTimes_LineEdit.setText(str(self.DragGAN.test_times))
        self.ui.DragTimes_LineEdit.setText(str(self.DragGAN.drag_times))
        self.ui.Optimize_CheckBox.setChecked(self.DragGAN.is_optimize)


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
    def on_EmbeddingBrowse_PushButton_clicked(self):
        file = QFileDialog.getOpenFileName(
            self, "Select Embedding Files", os.path.realpath("./checkpoints/PTI"), "Embedding Files (*.pt)")
        embedding_path = file[0]
        if not os.path.isfile(embedding_path):
            return
        self.DragGAN.w_load = torch.load(embedding_path)
        self.DragGAN.embedding_path = embedding_path
        self.ui.Embedding_LineEdit.setText(os.path.basename(embedding_path))

    @Slot()
    def on_EmbeddingReset_PushButton_clicked(self):
        self.DragGAN.w_load = None
        self.DragGAN.embedding_path = None
        self.ui.Embedding_LineEdit.setText("")

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
        if self.DragGAN.w_load is not None:
            image = self.DragGAN.generateImage(self.DragGAN.seed, self.DragGAN.w_plus, self.DragGAN.w_load) # 3 * 512 * 512
        else:
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
        self.run_thread = DragThread(self.DragGAN, self.ui.Image_Widget)
        self.run_thread.once_finished.connect(self.on_once_finished)
        self.run_thread.drag_finished.connect(lambda: print("********** Drag Finished **********"))
        self.run_thread.start()

    @Slot(torch.Tensor, list, float, int)
    def on_once_finished(self, image, points, loss, steps):
        self.ui.Image_Widget.clear_points()
        self.ui.Image_Widget.add_points(points)
        # print(f"pointfs: {self.ui.Image_Widget.get_points()}")
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
    def on_R3_LineEdit_editingFinished(self):
        self.DragGAN.r3 = float(self.ui.R3_LineEdit.text())
        print(f"current r3: {self.DragGAN.r3}")

    @Slot()
    def on_Reset4R3_PushButton_clicked(self):
        self.DragGAN.r3 = self.DragGAN.DEFAULT_R3
        self.ui.R3_LineEdit.setText(str(self.DragGAN.r3))

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
    def on_Optimize_CheckBox_stateChanged(self):
        if self.ui.Optimize_CheckBox.isChecked():
            self.DragGAN.is_optimize = True
            print(f"optimize: {self.DragGAN.is_optimize}")
        else:
            self.DragGAN.is_optimize = False
            print(f"optimize: {self.DragGAN.is_optimize}")
################## model ##################

################## experiment ##################

    @Slot()
    def on_Test_PushButton_clicked(self):
        print("test")
        import dlib
        import cv2

        # 保存图片
        pickle = os.path.basename(self.DragGAN.pickle_path).split(os.extsep)[0]
        image_format = "png"
        filename = f"{pickle}_{self.DragGAN.seed}.{image_format}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, "save_images", "edited_images")
        filename = os.path.join(image_dir, filename)
        self.ui.Image_Widget.save_image(filename, image_format, 100)
        print(f"save image as {filename}")
        ###################################################################################################################
        # 参数设置

        dat_68 = "./landmarks/shape_predictor_68_face_landmarks.dat"
        dat_5 = "./landmarks/shape_predictor_5_face_landmarks.dat"

        shape_predictor = ""
        if self.DragGAN.only_one_point:
            shape_predictor = dat_68
        if self.DragGAN.five_points:
            shape_predictor = dat_5
        if self.DragGAN.sixty_eight_points:
            shape_predictor = dat_68
        origin_file = filename
        target_file = self.ui.TargetImage_LineEdit.text()

        ###################################################################################################################
        # （1）先检测人脸，然后定位脸部的关键点。优点: 与直接在图像中定位关键点相比，准确度更高。
        detector = dlib.get_frontal_face_detector()			# 1.1、基于dlib的人脸检测器
        predictor = dlib.shape_predictor(shape_predictor)	# 1.2、基于dlib的关键点定位（68个关键点）

        # （2）图像预处理
        # 2.1、读取图像
        origin_image = cv2.imread(origin_file)
        target_image = cv2.imread(target_file)

        q_size = self.ui.Image_Widget.image_label.size()
        # image_rate = self.ui.Image_Widget.image_rate

        width = q_size.width() 		        # 指定宽度

        (o_h, o_w) = origin_image.shape[:2]	# 获取图像的宽和高
        o_r = width / float(o_w)			# 计算比例
        o_dim = (width, int(o_h * o_r))		# 按比例缩放高度: (宽, 高)

        (t_h, t_w) = target_image.shape[:2]	# 获取图像的宽和高
        t_r = width / float(t_w)			# 计算比例
        t_dim = (width, int(t_h * t_r))		# 按比例缩放高度: (宽, 高)

        # self.ui.Image_Widget.set_image_scale(o_r)
        # 2.2、图像缩放
        origin_image = cv2.resize(origin_image, o_dim, interpolation=cv2.INTER_AREA)
        target_image = cv2.resize(target_image, t_dim, interpolation=cv2.INTER_AREA)
        # 2.3、灰度图
        origin_gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        # （3）人脸检测
        origin_rects = detector(origin_gray, 1)				# 若有多个目标，则返回多个人脸框
        target_rects = detector(target_gray, 1)				# 若有多个目标，则返回多个人脸框

        # （4）遍历检测得到的【人脸框 + 关键点】
        # rect: 人脸框
        for o_rect, t_rect in zip(origin_rects, target_rects):		
            # 4.1、定位脸部的关键点（返回的是一个结构体信息，需要遍历提取坐标）
            o_shape = predictor(origin_gray, o_rect)
            t_shape = predictor(target_gray, t_rect)
            # 4.2、遍历shape提取坐标并进行格式转换: ndarray
            o_shape = utils.shape_to_np(o_shape)
            t_shape = utils.shape_to_np(t_shape)
            # 4.3、根据脸部位置获得点（每个脸部由多个关键点组成）
            points = []
            if self.DragGAN.only_one_point:
                o_x, o_y = o_shape[33]
                t_x, t_y = t_shape[33]
                points.append(QPoint(int(o_x/o_r), int(o_y/o_r)))
                points.append(QPoint(int(t_x/t_r), int(t_y/t_r)))
            else:
                for (o_x, o_y), (t_x, t_y) in zip(o_shape, t_shape):
                    points.append(QPoint(int(o_x/o_r), int(o_y/o_r)))
                    points.append(QPoint(int(t_x/t_r), int(t_y/t_r)))
                
            self.ui.Image_Widget.add_points(points)


    @Slot()
    def on_SaveExperiment_PushButton_clicked(self):
        target_file = self.ui.TargetImage_LineEdit.text()
        target_file = os.path.basename(target_file)
        target_seed = target_file.split(os.extsep)[0]
        print(target_seed)
        target_seed = target_seed.split("_")[1]
        pickle = os.path.basename(self.DragGAN.pickle_path).split(os.extsep)[0]
        image_format = "png"
        filename = f"{pickle}_{self.DragGAN.seed}_{target_seed}.{image_format}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, "save_images", "experiment_images")
        filename = os.path.join(image_dir, filename)
        self.save_image(filename, image_format, 100)
        print(f"save image to {filename}")

    @Slot()
    def on_TestTimes_LineEdit_editingFinished(self):
        test_times = int(self.ui.TestTimes_LineEdit.text())
        if 0 < test_times:
            self.DragGAN.test_times = test_times
        print(self.DragGAN.test_times)

    @Slot()
    def on_DragTimes_LineEdit_editingFinished(self):
        drag_times = int(self.ui.DragTimes_LineEdit.text())
        if 0 < drag_times:
            self.DragGAN.drag_times = drag_times
        print(self.DragGAN.drag_times)

    @Slot()
    def on_Experience_PushButton_clicked(self):
        print("experience")
        self.run_thread = ExperienceThread(self.DragGAN, self.ui.Image_Widget)
        self.run_thread.experience_start.connect(self.on_experience_start)
        self.run_thread.random_seed.connect(self.on_random_seed)
        self.run_thread.once_finished.connect(self.on_once_finished)
        self.run_thread.start()

    @Slot()
    def on_experience_start(self):
        if not self.DragGAN.random_seed:
            self.DragGAN.random_seed = True
            self.ui.RandomSeed_CheckBox.setChecked(True)

    @Slot(int)
    def on_random_seed(self, new_seed):
        if self.DragGAN.random_seed:
            self.DragGAN.seed = new_seed
            self.ui.Seed_LineEdit.setText(str(new_seed))

    @Slot()
    def on_TargetImage_LineEdit_editingFinished(self):
        print(f"target image line edit: {self.ui.TargetImage_LineEdit.text()}")

    @Slot()
    def on_TargetImage_ToolButton_clicked(self):
        print("target image tool button")
        file = QFileDialog.getOpenFileName(
            self, "Select Target Files", os.path.realpath("./save_images/generated_images"), "Image files (*.png)")
        target_image_path = file[0]
        if not os.path.isfile(target_image_path):
            return
        # self.ui.Target_LineEdit.setText(os.path.basename(target_image_path))
        self.ui.TargetImage_LineEdit.setText(target_image_path)

    @Slot()
    def on_OnePoint_CheckBox_stateChanged(self):
        print("one point")
        if self.ui.OnePoint_CheckBox.isChecked():
            self.DragGAN.only_one_point = True
        else:
            self.DragGAN.only_one_point = False

    @Slot()
    def on_FivePoints_CheckBox_stateChanged(self):
        print("five points")
        if self.ui.FivePoints_CheckBox.isChecked():
            self.DragGAN.five_points = True
        else:
            self.DragGAN.five_points = False

    @Slot()
    def on_SixtyEightPoints_CheckBox_stateChanged(self):
        print("sixty eight points")
        if self.ui.SixtyEightPoints_CheckBox.isChecked():
            self.DragGAN.sixty_eight_points = True
        else:
            self.DragGAN.sixty_eight_points = False

    @Slot()
    def on_FourBlock_CheckBox_stateChanged(self):
        print("four block")
        if self.ui.FourBlock_CheckBox.isChecked():
            self.DragGAN.fourth_block = True
        else:
            self.DragGAN.fourth_block = False
    
    @Slot()
    def on_FiveBlock_CheckBox_stateChanged(self):
        print("five block")
        if self.ui.FiveBlock_CheckBox.isChecked():
            self.DragGAN.fifth_block = True
        else:
            self.DragGAN.fifth_block = False

    @Slot()
    def on_SixBlock_CheckBox_stateChanged(self):
        print("six block")
        if self.ui.SixBlock_CheckBox.isChecked():
            self.DragGAN.sixth_block = True
        else:
            self.DragGAN.sixth_block = False

    @Slot()
    def on_SevenBlock_CheckBox_stateChanged(self):
        print("seven block")
        if self.ui.SevenBlock_CheckBox.isChecked():
            self.DragGAN.seventh_block = True
        else:
            self.DragGAN.seventh_block = False

################## mask ##################

    @Slot()
    def on_FlexibleArea_PushButton_clicked(self):
        print("flexible area")

    @Slot()
    def on_FixedArea_PushButton_clicked(self):
        print("fixed area")

    @Slot()
    def on_ResetMask_PushButton_clicked(self):
        print("reset mask")

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

################### image ##################


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
