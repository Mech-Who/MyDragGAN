import os
import sys
import random

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QFileDialog
from PySide6.QtGui import QPainter, QImage
from ui.Ui_MainWindow import Ui_DragGAN
import cv2
import numpy as np

from model import StyleGAN


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DragGAN()
        self.ui.setupUi(self)
        self.setWindowTitle(self.tr("DragGAN"))

        self.setConnect()

        #### model部分初始化 ####
        # 设置device可选值并设默认值
        self.ui.Device_ComboBox.addItem(self.tr("cpu"))
        self.ui.Device_ComboBox.addItem(self.tr("cuda"))
        self.ui.Device_ComboBox.addItem(self.tr("mps"))
        self.device = "cpu"
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

    def setConnect(self):
        pass

    def loadCpkt(self, pickle_path):
        self.pickle_path = pickle_path
        self.model.load_ckpt(pickle_path)

    def generateImage(self, seed):
        if self.pickle_path:
            # 将opt设置为None, 表示开始一次新的优化
            self.optimizer = None

            # seed -> w -> image
            self.W = self.model.gen_w(seed)
            img, self.init_F = self.model.gen_img(self.W)

            # 将生成的图片转换成可ui支持的raw data
            # t_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            # t_img = cv2.resize(t_img, (512, 512))
            # raw_img = (t_img / 2 + 0.5).clip(0, 1).reshape(-1)

            raw_img = img

            return raw_img
        else:
            return None

    def prepare4Drag(self, init_pts, lr=2e-3):
        # 1. 备份初始图像的特征图 -> motion supervision和point tracking都需要用到
        # self.F0_resized = torch_F.interpolate(self.init_F,
        #                                       size=(512, 512),
        #                                       mode="bilinear",
        #                                       align_corners=True).detach().clone()
        print("set F0_resized")

        # 2. 备份初始点坐标 -> point tracking
        # temp_init_pts_0 = copy.deepcopy(init_pts)
        # self.init_pts_0 = torch.from_numpy(temp_init_pts_0).float().to(self.device)
        print("backup for points")

        # 3. 将w向量的部分特征设置为可训练
        # temp_W = self.W.cpu().numpy().copy()
        # self.W = torch.from_numpy(temp_W).to(self.device).float()
        # self.W.requires_grad_(False)

        # self.W_layers_to_optimize = self.W[:, :6, :].detach().clone().requires_grad_(True)
        # self.W_layers_to_fixed = self.W[:, 6:, :].detach().clone().requires_grad_(False)
        print("set part of W as trainable")

        # 4. 初始化优化器
        # self.optimizer = torch.optim.Adam([self.W_layers_to_optimize], lr=lr)
        print("init optimizer")

    def drag(self, _init_pts, _tar_pts, allow_error_px=2, r1=3, r2=13):
        # init_pts = torch.from_numpy(_init_pts).float().to(self.device)
        # tar_pts = torch.from_numpy(_tar_pts).float().to(self.device)
        print(f"get init_pts: {_init_pts} and tar_pts: {_tar_pts}")

        # 如果起始点和目标点之间的像素误差足够小，则停止
        # if torch.allclose(init_pts, tar_pts, atol=atol):
        #     return False, (None, None)
        # self.optimizer.zero_grad()
        print("check if init_pts and tar_pts are close enough")

        # 将latent的0:6设置成可训练,6:设置成不可训练 See Sec3.2
        # W_combined = torch.cat([self.W_layers_to_optimize, self.W_layers_to_fixed], dim=1)
        print("set 0:6 as trainable and 6: as fixed")
        # 前向推理
        # new_img, _F = self.generator.gen_img(W_combined)
        print("forward inference")
        # See, Sec 3.2 in paper, 计算motion supervision loss
        # F_resized = torch_F.interpolate(_F, size=(512, 512), mode="bilinear", align_corners=True)
        # loss = self.motion_supervision(
        #     F_resized,
        #     init_pts, tar_pts,
        #     r1=r1)

        # loss.backward()
        # self.optimizer.step()
        print("calculate Motion Supervision loss")

        # 更新初始点 see Sec3.3 Point Tracking
        # with torch.no_grad():
        #     # 以上过程会优化一次latent, 直接用新的latent生成图像，用于中间过程的显示
        #     new_img, F_for_point_tracking = self.generator.gen_img(W_combined)
        #     new_img = new_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        #     new_img = cv2.resize(new_img, (512, 512))
        #     new_raw_img = (new_img / 2 + 0.5).clip(0, 1).reshape(-1)

        #     F_for_point_tracking_resized = torch_F.interpolate(F_for_point_tracking, size=(512, 512),
        #                                                        mode="bilinear", align_corners=True).detach()
        #     new_init_pts = self.point_tracking(F_for_point_tracking_resized, init_pts, r2=r2)
        # print(f"Loss: {loss.item():0.4f}, tar pts: {tar_pts.cpu().numpy()}, new init pts: {new_init_pts.cpu().numpy()}")
        # print('\n')
        print("update init_pts as Point Tracking")

        # return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_raw_img)
        return True, (np.array([0, 0]), np.array([1, 1]), np.array([0, 1]))

################### model ##################

    @Slot()
    def on_Device_ComboBox_currentIndexChanged(self):
        self.device = self.ui.Device_ComboBox.currentText()
        print(f"current device: {self.device}")

    @Slot()
    def on_Recent_PushButton_clicked(self):
        pass

    @Slot()
    def on_Browse_PushButton_clicked(self):
        file = QFileDialog.getOpenFileName(
            self, "Select Pickle Files", os.path.realpath("."), "Pickle Files (*.pkl)")
        self.pickle_path = file[0]
        self.ui.Pickle_LineEdit.setText(os.path.basename(self.pickle_path))

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
        self.model = StyleGAN(self.pickle_path, self.device, self.seed)

################## drag ##################

    @Slot()
    def on_AddPoint_PushButton_clicked(self):
        self.status = "drag"
        print(f"current status: {self.status}")
        print(f"current init_pts: {self.init_pts}")

    @Slot()
    def on_ResetPoint_PushButton_clicked(self):
        self.init_pts = []
        print(f"current status: {self.status}")
        print(f"current init_pts: {self.init_pts}")

    @Slot()
    def on_Start_PushButton_clicked(self):
        print("start drag")
        if self.status == "drag":
            self.init_pts = []
            self.tar_pts = []
            self.raw_img = None
            self.status = "wait"
            print(f"current status: {self.status}")

    @Slot()
    def on_Stop_PushButton_clicked(self):
        print("stop drag")

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
