import os
import sys
import random
import json
import threading

from PySide6.QtCore import Signal, Slot, QPoint
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QFileDialog
from PySide6.QtGui import QPainter, QImage

sys.path.append('stylegan2_ada')

from ui.Ui_MainWindow import Ui_DragGAN
from components.LabelStatus import LabelStatus
# from model import StyleGAN
from stylegan2_ada.model import StyleGAN
import utils as utils

import torch.nn.functional as torch_F
import torch
import copy
import cv2
import numpy as np
from array import array



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DragGAN()
        self.ui.setupUi(self)
        self.setWindowTitle(self.tr("DragGAN"))

        # 配置文件
        self.config_path = os.path.join(os.getcwd(), "config.json")

        #### model部分初始化 ####
        # 设置device可选值并设默认值
        self.device = "cpu"
        # 设置模型并设置默认值
        self.model = StyleGAN(device=self.device)
        
        # 设置下拉框值
        self.ui.Device_ComboBox.addItem(self.tr("cpu"))
        self.ui.Device_ComboBox.addItem(self.tr("cuda"))
        self.ui.Device_ComboBox.addItem(self.tr("mps"))
        self.ui.Device_ComboBox.setCurrentText(self.device)

        # 设置模型参数文件路径默认值
        self.pickle_path = ""
        self.ui.Pickle_Label.setText(self.pickle_path)
        # 设置seed默认值与阈值
        self.seed = 0
        self.ui.Seed_LineEdit.setText(str(self.seed))
        self.min_seed = 0
        self.max_seed = 20000

        # 设置默认步长
        self.step_size = 0.5
        self.ui.StepSize_LineEdit.setText(str(self.step_size))

        #### drag部分初始化 ####
        # 设置Radius默认值与阈值
        self.radius = 1
        self.ui.Radius_LineEdit.setText(str(self.radius))
        self.min_radius = 0.1
        self.max_radius = 10
        # 设置Lambda默认值与阈值
        self.lambda_ = 0.5
        self.ui.Lambda_LineEdit.setText(str(self.lambda_))
        self.min_lambda = 0
        self.max_lambda = 10

        self.isDragging = False

        self.steps = 0
        self.ui.StepNumber_Label.setText(str(self.steps))

        #### 初始化ImageLabel ####
        # 设置ImageLabel的自适应大小比率
        self.image_rate = None
        self.image_width, self.image_height= 512, 512
        self.rgb_channel, self.rgba_channel = 3, 4
        self.image_pixels = self.image_height * self.image_width
        self.raw_data_size = self.image_width * self.image_height * self.rgba_channel
        self.raw_data = array('f', [1] * self.raw_data_size)

    def getConfig(self, key=None):
        config = None
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if key:
            try:
                value = config[key]
            except KeyError:
                raise KeyError(f"key: {key} not found in config.json")
            return value
        return config
    
    def setConfig(self, config):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    def addConfig(self, key, value):
        config = self.getConfig()
        config[key] = value
        self.setConfig(config)
    
    def delConfig(self, key):
        config = self.getConfig()
        del config[key]
        self.setConfig(config)

    def change_device(self, new_device):
        if new_device != self.device:
            self.device = new_device
            self.model.change_device(new_device)

    def loadCpkt(self, pickle_path):
        self.pickle_path = pickle_path
        self.model.load_ckpt(pickle_path)

    def generateImage(self, seed):
        if self.pickle_path:
            # 将opt设置为None, 表示开始一次新的优化
            self.optimizer = None

            # seed -> w -> image(torch.Tensor)
            self.W = self.model.gen_w(seed)
            img, self.init_F = self.model.gen_img(self.W)
            # print(f"type of img: {type(img)}")
            # print(f"shape of img: {img.shape}")

            # 将生成的图片转换成可ui支持的raw data
            t_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]   # numpy.ndarray (512, 512, 3)
            # print(f"type of t_img 1: {type(t_img)}")
            # print(f"shape of t_img 1: {t_img.shape}")
            # print(f"t_img 1: {t_img}")
            t_img = cv2.resize(t_img, (512, 512))
            # print(f"type of t_img 2: {type(t_img)}")
            # print(f"shape of t_img 2: {t_img.shape}")
            # raw_img = (t_img / 2 + 0.5).clip(0, 1).reshape(-1)
            # raw_img = (t_img / 2 + 0.5).clip(0, 1)
            # raw_img = ((t_img / 2 + 0.5)*255)
            # raw_img = t_img
            raw_img = (t_img / 2 + 0.5).clip(0, 1)
            raw_img = (t_img * 255).astype(np.uint8)
            # print(f"type of raw_img: {type(raw_img)}")
            # print(f"shape of img: {raw_img.shape}")

            return raw_img
        else:
            return None


    def update_image(self, new_image):
        # # Convert image data (rgb) to raw_data (rgba)
        # for i in range(0, self.image_pixels):
        #     rd_base, im_base = i * self.rgba_channel, i * self.rgb_channel
        #     self.raw_data[rd_base:rd_base + self.rgb_channel] = array(
        #         'f', new_image[im_base:im_base + self.rgb_channel]
        #     )
        # self.ui.Image_Widget.set_image_from_array(self.raw_data)
        self.ui.Image_Widget.set_image_from_array(new_image)

    def prepare2Drag(self, init_pts, lr=2e-3):
        # 1. 备份初始图像的特征图 -> motion supervision和point tracking都需要用到
        self.F0_resized = torch_F.interpolate(  self.init_F,
                                                size=(512, 512),
                                                mode="bilinear",
                                                align_corners=True).detach().clone()
        # print("set F0_resized")

        # 2. 备份初始点坐标 -> point tracking
        temp_init_pts_0 = copy.deepcopy(init_pts)
        self.init_pts_0 = torch.from_numpy(temp_init_pts_0).float().to(self.device)
        # print("backup for points")

        # 3. 将w向量的部分特征设置为可训练
        temp_W = self.W.cpu().numpy().copy()
        self.W = torch.from_numpy(temp_W).to(self.device).float()
        self.W.requires_grad_(False)

        self.W_layers_to_optimize = self.W[:, :6, :].detach().clone().requires_grad_(True)
        self.W_layers_to_fixed = self.W[:, 6:, :].detach().clone().requires_grad_(False)
        # print("set part of W as trainable")

        # 4. 初始化优化器
        self.optimizer = torch.optim.Adam([self.W_layers_to_optimize], lr=lr)
        # print("init optimizer")

# 计算motion supervision loss, 用来更新w，使图像中目标点邻域的特征与起始点领域的特征靠近
    def motion_supervision( self, 
                            F,
                            init_pts, 
                            tar_pts, 
                            r1=3):
        
        n = init_pts.shape[0]
        loss = 0.0
        for i in range(n):
            # 计算方向向量
            dir_vec = tar_pts[i] - init_pts[i]
            d_i = dir_vec / (torch.norm(dir_vec) + 1e-7)
            if torch.norm(d_i) > torch.norm(dir_vec):
                d_i = dir_vec
            # 创建一个圆形mask区域，以起始点为中心，r为半径，也就是原文的red circle(Fig.3)
            mask = utils.create_circular_mask(
                F.shape[2], F.shape[3], center=init_pts[i].tolist(), radius=r1
            ).to(self.device)
            # 将mask的index找到
            coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]
            # 根据之前计算的运动向量，移动mask的index, 也就是得到目标点的mask对应的index
            shifted_coordinates = coordinates + d_i[None]
            # 从特征图中，拿出mask区域的特征
            F_qi = F[:, :, mask] # [1, C, num_points]

            # 下面这一坨代码都是为了得到mask平移后新的位置的特征, 并且需要这个过程可微
            # 1. 将coordinates坐标系的原点平移到图像中心，为了grid_sample函数准备
            h, w = F.shape[2], F.shape[3]
            norm_shifted_coordinates = shifted_coordinates.clone()
            norm_shifted_coordinates[:, 0] = (2.0 * shifted_coordinates[:, 0] / (h - 1)) - 1
            norm_shifted_coordinates[:, 1] = (2.0 * shifted_coordinates[:, 1] / (w - 1)) - 1
            # 将norm_shifted_coordinates尺寸从[num_points, 2] -> [1, 1, num_points, 2] 
            norm_shifted_coordinates = norm_shifted_coordinates.unsqueeze(0).unsqueeze(0)
            norm_shifted_coordinates = norm_shifted_coordinates.clamp(-1, 1)
            # 2. 将[x, y] -> [y, x]，为了grid_sample函数准备
            norm_shifted_coordinates = norm_shifted_coordinates.flip(-1)
            # 3. 执行grid_sample，拿到平移后mask对应的特征
            F_qi_plus_di = torch_F.grid_sample(F, norm_shifted_coordinates, mode="bilinear", align_corners=True)
            F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

            # 监督移动前后的特征要一致
            loss += torch_F.l1_loss(F_qi.detach(), F_qi_plus_di)
            # 同时监督特征图上, mask外的特征要一致 暂不支持
        
        return loss

    # 目的是更新初始点,因为图像通过motion_supervision已经发生了变化
    # init_pts -> new init_pts -> ... -> tar_pts
    def point_tracking( self, 
                        F,
                        init_pts, 
                        r2):
        n = init_pts.shape[0]
        new_init_pts = torch.zeros_like(init_pts)
        '''
        为什么要有这一步: 
            motion_supervision更新了图像, 之前的初始点不知道具体移动到哪里了
        如何找到最新的初始点:  
            在老的初始点附近划定一个区域, 在这个区域内找到与老的初始点对应特征最近的特征, 那么这个特征对应的位置就是新的初始点
        '''
        for i in range(n):
            # 以初始点为中心生成一个正方形mask,
            patch = utils.create_square_mask(   F.shape[2], 
                                                F.shape[3], 
                                                center=init_pts[i].tolist(), 
                                                radius=r2).to(self.device)
            patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]
            # 拿到mask区域的特征
            F_qi = F[..., patch_coordinates[:, 0], patch_coordinates[:, 1]] # [N, C, num_points] torch.Size([1, 128, 729])
            # 旧初始点的特征
            f_i = self.F0_resized[..., self.init_pts_0[i][0].long(), self.init_pts_0[i][1].long()] # [N, C, 1]
            # 计算mask内每个特征与老的初始点对应特征的距离
            distances = (F_qi - f_i[:, :, None]).abs().mean(1) # [N, num_points] torch.Size([1, 729])
            # 找到距离最小的，也就是老的初始点对应特征最像的特征
            min_index = torch.argmin(distances)
            new_init_pts[i] = patch_coordinates[min_index] # [row, col] 
            
        return new_init_pts

    def drag(self, _init_pts, _tar_pts, allow_error_px=2, r1=3, r2=13):
        init_pts = torch.from_numpy(_init_pts).float().to(self.device)
        tar_pts = torch.from_numpy(_tar_pts).float().to(self.device)
        # print(f"get init_pts: {_init_pts} and tar_pts: {_tar_pts}")

        # 如果起始点和目标点之间的像素误差足够小，则停止
        if torch.allclose(init_pts, tar_pts, atol=allow_error_px):
            return False, (None, None)
        self.optimizer.zero_grad()
        # print("check if init_pts and tar_pts are close enough")

        # 将latent的0:6设置成可训练,6:设置成不可训练 See Sec3.2
        W_combined = torch.cat([self.W_layers_to_optimize, self.W_layers_to_fixed], dim=1)
        # print("set 0:6 as trainable and 6: as fixed")
        # 前向推理
        new_img, _F = self.model.gen_img(W_combined)
        # print("forward inference")
        # See, Sec 3.2 in paper, 计算motion supervision loss
        F_resized = torch_F.interpolate(_F, size=(512, 512), mode="bilinear", align_corners=True)
        loss = self.motion_supervision(
            F_resized,
            init_pts, tar_pts,
            r1=r1)

        loss.backward()
        self.optimizer.step()
        # print("calculate Motion Supervision loss")

        # 更新初始点 see Sec3.3 Point Tracking
        with torch.no_grad():
            # 以上过程会优化一次latent, 直接用新的latent生成图像，用于中间过程的显示
            new_img, F_for_point_tracking = self.model.gen_img(W_combined)
            new_img = new_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            new_img = cv2.resize(new_img, (512, 512))
            # new_raw_img = (new_img / 2 + 0.5).clip(0, 1).reshape(-1)
            new_raw_img = (new_img / 2 + 0.5).clip(0, 1)
            new_raw_img = (new_img * 255).astype(np.uint8)

            F_for_point_tracking_resized = torch_F.interpolate(F_for_point_tracking, size=(512, 512),
                                                               mode="bilinear", align_corners=True).detach()
            new_init_pts = self.point_tracking(F_for_point_tracking_resized, init_pts, r2=r2)
        print(f"Loss: {loss.item():0.4f}, tar pts: {tar_pts.cpu().numpy()}, new init pts: {new_init_pts.cpu().numpy()}")
        print('\n')
        # print("update init_pts as Point Tracking")

        return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_raw_img)
        # return True, (np.array([0, 0]), np.array([1, 1]), np.array([0, 1]))

    def drag_thread(self):
        points = self.ui.Image_Widget.get_points()
        if len(points) % 2 == 1:
            points = points[:-1]
        init_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 0])
        tar_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 1])
        init_pts = np.vstack(init_pts)[:, ::-1].copy()
        tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
        self.prepare2Drag(init_pts)
        
        self.steps = 0
        while(self.isDragging):
            # 迭代一次
            status, ret = self.drag(init_pts, tar_pts)
            if status:
                init_pts, _, image = ret
            else:
                self.isDragging = False
                return
            # 显示最新的图像  
            points = []
            for i in range(init_pts.shape[0]):
                points.append(QPoint(int(init_pts[i][1]), int(init_pts[i][0])))
            for i in range(tar_pts.shape[0]):
                points.append(QPoint(int(tar_pts[i][1]), int(tar_pts[i][0])))
            self.ui.Image_Widget.clear_points()
            self.ui.Image_Widget.add_points(points)
            self.update_image(image)

            self.steps += 1
            self.ui.StepNumber_Label.setText(str(self.steps))

################### model ##################

    @Slot()
    def on_Device_ComboBox_currentIndexChanged(self):
        self.device = self.ui.Device_ComboBox.currentText()
        self.model.change_device(self.device)
        print(f"current device: {self.device}")

    @Slot()
    def on_Recent_PushButton_clicked(self):
        self.pickle_path = self.getConfig("last_pickle")
        self.ui.Pickle_LineEdit.setText(os.path.basename(self.pickle_path))

    @Slot()
    def on_Browse_PushButton_clicked(self):
        file = QFileDialog.getOpenFileName(
            self, "Select Pickle Files", os.path.realpath("./checkpoints"), "Pickle Files (*.pkl)")
        self.pickle_path = file[0]
        if not os.path.isfile(self.pickle_path):
            return
        self.ui.Pickle_LineEdit.setText(os.path.basename(self.pickle_path))
        self.addConfig("last_pickle", self.pickle_path)
        

    @Slot()
    def on_Minus4Seed_PushButton_clicked(self):
        self.seed = int(self.ui.Seed_LineEdit.text())
        if self.seed > self.min_seed:
            self.seed -= 1
            self.ui.Seed_LineEdit.setText(str(self.seed))

    @Slot()
    def on_Plus4Seed_PushButton_clicked(self):
        self.seed = int(self.ui.Seed_LineEdit.text())
        if self.seed < self.max_seed:
            self.seed += 1
            self.ui.Seed_LineEdit.setText(str(self.seed))

    @Slot()
    def on_StepSize_LineEdit_editingFinished(self):
        self.step_size = float(self.ui.StepSize_LineEdit.text())
        print(f"current step_size: {self.step_size}")

    @Slot()
    def on_Reset_PushButton_clicked(self):
        self.step_size = 0.5
        self.ui.StepSize_LineEdit.setText(str(self.step_size))

    @Slot()
    def on_Generate_PushButton_clicked(self):
        print("start generate")
        self.model.load_ckpt(self.pickle_path)
        # file = os.path.realpath("./components/dog.jpg")
        # print(file)
        # self.ui.Image_Widget.set_image(file)
        image = self.generateImage(self.seed) # 3 * 512 * 512
        if image is not None:
            # self.update_image(image)
            self.ui.Image_Widget.set_image_from_array(image)
        # self.model = StyleGAN(self.pickle_path, self.device, self.seed)

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
        if self.isDragging:
            QMessageBox.warning(self, "Warning", "Dragging is running!", QMessageBox.Ok)
            return
        # TODO: 可能需要使用QT内置的多线程机制重新实现
        self.isDragging = True
        threading.Thread(target=self.drag_thread, daemon=True).start()


    @Slot()
    def on_Stop_PushButton_clicked(self):
        print("stop drag")
        self.isDragging = False

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
        self.radius = int(self.ui.Radius_LineEdit.text())
        self.radius -= 0.1
        self.ui.Radius_LineEdit.setText(str(self.radius))

    @Slot()
    def on_Plus4Radius_PushButton_clicked(self):
        self.radius = float(self.ui.Radius_LineEdit.text())
        if self.radius < self.max_radius:
            self.radius += 0.1
            self.ui.Radius_LineEdit.setText(str(self.radius))

    @Slot()
    def on_Minus4Lambda_PushButton_clicked(self):
        self.lambda_ = float(self.ui.Lambda_LineEdit.text())
        if self.lambda_ > self.min_lambda:
            self.lambda_ -= 0.1
            self.ui.Lambda_LineEdit.setText(str(self.lambda_))

    @Slot()
    def on_Plus4Lambda_PushButton_clicked(self):
        self.lambda_ = float(self.ui.Lambda_LineEdit.text())
        if self.lambda_ < self.max_lambda:
            self.lambda_ += 0.1
            self.ui.Lambda_LineEdit.setText(str(self.lambda_))

################### image ##################


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
