import copy
import datetime

import numpy as np
import utils
import os
import torch
import torch.nn.functional as torch_F
from stylegan2_ada.model import StyleGAN

from PySide6.QtCore import QPoint, QThread, Signal


class DragGAN:
    DEFAULT_STEP_SIZE = 2e-3
    DEFAULT_R1 = 3
    DEFAULT_R2 = 13
    def __init__(self):

        #### model部分初始化 ####
        # 设置device可选值并设默认值
        self.device = "cpu"
        # 设置模型并设置默认值
        self.model = StyleGAN(device=self.device)

        self.pickle_path = ""
        # 设置seed默认值与阈值
        self.seed = 0
        self.min_seed = 0
        self.max_seed = 65535
        self.random_seed = False
        self.w_plus = True
        #### drag部分初始化 ####
        # 设置Radius默认值与阈值
        self.radius = 1
        self.min_radius = 0.1
        self.max_radius = 10
        # 设置Lambda默认值与阈值
        self.lambda_ = 10
        self.min_lambda = 5
        self.max_lambda = 20
        # 设置默认步长
        self.step_size = self.DEFAULT_STEP_SIZE
        # 初始化R1和R2
        self.r1 = self.DEFAULT_R1
        self.r2 = self.DEFAULT_R2
        self.steps = 0
        self.isDragging = False

        # 保存图像数据
        self.points = []
        self.update_image = None
        self.origin_image = None
        self.mask = None

    def setDevice(self, device):
        if device != self.device:
            self.device = device
            self.model.change_device(device)

    def loadCpkt(self, pickle_path):
        self.pickle_path = pickle_path
        self.model.load_ckpt(pickle_path)

    def generateImage(self, seed, w_plus=True):
        if self.pickle_path:
            # 将opt设置为None, 表示开始一次新的优化
            self.optimizer = None

            # seed -> w -> image(torch.Tensor)
            self.W = self.model.gen_w(seed, w_plus)
            img, self.init_F = self.model.gen_img(self.W)
            # 处理图像数据为RGB格式，每一个元素为0~255的数据
            img = img[0]
            img_scale_db = 0
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # 保存图像状态
            self.image = self.origin_image = img

            return img
        else:
            return None

    def prepare2Drag(self, init_pts, lr=2e-3):
        # 1. 备份初始图像的特征图 -> motion supervision和point tracking都需要用到
        self.F0_resized = torch_F.interpolate(  self.init_F,
                                                size=(512, 512),
                                                mode="bilinear",
                                                align_corners=True).detach().clone()

        # 2. 备份初始点坐标 -> point tracking
        temp_init_pts_0 = copy.deepcopy(init_pts)
        self.init_pts_0 = torch.from_numpy(temp_init_pts_0).float().to(self.device)

        # 3. 将w向量的部分特征设置为可训练
        temp_W = self.W.cpu().numpy().copy()
        self.W = torch.from_numpy(temp_W).to(self.device).float()
        self.W.requires_grad_(False)

        self.W_layers_to_optimize = self.W[:, :6, :].detach().clone().requires_grad_(True)
        self.W_layers_to_fixed = self.W[:, 6:, :].detach().clone().requires_grad_(False)

        # 4. 初始化优化器
        self.optimizer = torch.optim.Adam([self.W_layers_to_optimize], lr=lr)

    # 计算motion supervision loss, 用来更新w，使图像中目标点邻域的特征与起始点领域的特征靠近
    def motionSupervision( self, 
                            F,
                            init_pts, 
                            tar_pts, 
                            r1=3,
                            mask=None):
        
        n = init_pts.shape[0]
        loss = 0.0
        for i in range(n):
            # 计算方向向量d_i
            dir_vec = tar_pts[i] - init_pts[i]
            d_i = dir_vec / (torch.norm(dir_vec) + 1e-7)
            if torch.norm(d_i) > torch.norm(dir_vec):
                d_i = dir_vec
            # 创建一个圆形mask区域，以起始点为中心，r为半径，也就是原文的red circle(Fig.3)
            circle_mask = utils.create_circular_mask(
                F.shape[2], F.shape[3], center=init_pts[i].tolist(), radius=r1
            ).to(self.device)
            # 将mask的index找到
            coordinates = torch.nonzero(circle_mask).float()  # shape [num_points, 2]
            # 根据之前计算的运动向量，移动mask的index, 也就是得到目标点的mask对应的index
            shifted_coordinates = coordinates + d_i[None]
            # 从特征图中，拿出mask区域的特征
            F_qi = F[:, :, circle_mask] # [1, C, num_points]

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
            # TODO: 同时监督特征图上, mask外的特征要一致, 关键在于传入的mask
            # if mask_mask is not None:
            #     if mask_mask.min()==0 and mask_mask.max()==1:
            #         mask_mask_array = mask_mask.bool().to(self.device).unsqueeze(0).unsqueeze(0)
            #         loss_add = torch_F.l1_loss(self.init_F*mask_mask_array, self.F0_resized*mask_mask_array)
            #         loss += loss_add * self.lambda_
        
        return loss

    # 目的是更新初始点,因为图像通过motion_supervision已经发生了变化
    # init_pts -> new init_pts -> ... -> tar_pts
    def pointTracking( self, 
                        F,
                        init_pts, 
                        r2
                        ):
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
            # 拿到临近的矩形mask区域的特征
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

        # 如果起始点和目标点之间的像素误差足够小，则停止
        if torch.allclose(init_pts, tar_pts, atol=allow_error_px):
            return False, (None, None)
        self.optimizer.zero_grad()

        # 将latent的0:6设置成可训练,6:设置成不可训练 See Sec3.2
        W_combined = torch.cat([self.W_layers_to_optimize, self.W_layers_to_fixed], dim=1)
        # 前向推理
        # See, Sec 3.2 in paper, 计算motion supervision loss
        new_img, _F = self.model.gen_img(W_combined)
        F_resized = torch_F.interpolate(_F, size=(512, 512), mode="bilinear", align_corners=True)
        loss = self.motionSupervision(
            F_resized,
            init_pts, 
            tar_pts,
            r1=r1,
            mask=self.mask)

        loss.backward()
        self.optimizer.step()
        # print("calculate Motion Supervision loss")

        # 更新初始点 see Sec3.3 Point Tracking
        with torch.no_grad():
            # 以上过程会优化一次latent, 直接用新的latent生成图像，用于中间过程的显示
            new_img, F_for_point_tracking = self.model.gen_img(W_combined)
            
            new_img = new_img[0]
            img_scale_db = 0
            new_img = new_img * (10 ** (img_scale_db / 20))
            new_img = (new_img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            self.update_image = new_img

            F_for_point_tracking_resized = torch_F.interpolate(F_for_point_tracking, size=(512, 512),
                                                               mode="bilinear", align_corners=True).detach()
            new_init_pts = self.pointTracking(F_for_point_tracking_resized, init_pts, r2=r2)
        print(f"Loss: {loss.item():0.4f}, tar pts: {tar_pts.cpu().numpy()}, new init pts: {new_init_pts.cpu().numpy()}\n")
        # print("update init_pts as Point Tracking")

        return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_img, loss.item())
        # return True, (np.array([0, 0]), np.array([1, 1]), np.array([0, 1]))

    # def drag_thread(self):
    #     points = self.ui.Image_Widget.get_points()
    #     if len(points) % 2 == 1:
    #         points = points[:-1]
    #     init_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 0])
    #     tar_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 1])
    #     init_pts = np.vstack(init_pts)[:, ::-1].copy()
    #     tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
    #     self.prepare2Drag(init_pts, lr=self.step_size)
        
    #     self.steps = 0
    #     while(self.isDragging):
    #         # 迭代一次
    #         status, ret = self.drag(init_pts, tar_pts, allow_error_px=5, r1=3, r2=13)
    #         if status:
    #             init_pts, _, image, once_loss = ret
    #         else:
    #             self.isDragging = False
    #             return
    #         # 显示最新的图像  
    #         points = []
    #         for i in range(init_pts.shape[0]):
    #             points.append(QPoint(int(init_pts[i][1]), int(init_pts[i][0])))
    #         for i in range(tar_pts.shape[0]):
    #             points.append(QPoint(int(tar_pts[i][1]), int(tar_pts[i][0])))
    #         self.ui.Image_Widget.clear_points()
    #         self.ui.Image_Widget.add_points(points)
    #         self.update_image(image)

    #         self.steps += 1
    #         self.ui.StepNumber_Label.setText(str(self.steps))

class DragThread(QThread):
    drag_finished = Signal()
    once_finished = Signal(torch.Tensor, np.ndarray, np.ndarray, int)

    def __init__(self, draggan_model, points):
        super().__init__()
        self.DragGAN = draggan_model
        self.points = points

    def run(self):
        points = self.points
        if len(points) % 2 == 1:
            points = points[:-1]
        init_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 0])
        tar_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 1])
        init_pts = np.vstack(init_pts)[:, ::-1].copy()
        tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
        self.DragGAN.prepare2Drag(init_pts, lr=self.DragGAN.step_size)
        
        self.DragGAN.steps = 0
        while(self.DragGAN.isDragging):
            # 迭代一次
            status, ret = self.DragGAN.drag(init_pts, tar_pts, allow_error_px=5, r1=3, r2=13)
            if status:
                init_pts, _, image, once_loss = ret
            else:
                self.DragGAN.isDragging = False
                return
            # 显示最新的图像  
            points = []
            for i in range(init_pts.shape[0]):
                points.append(QPoint(int(init_pts[i][1]), int(init_pts[i][0])))
            for i in range(tar_pts.shape[0]):
                points.append(QPoint(int(tar_pts[i][1]), int(tar_pts[i][0])))

            self.DragGAN.steps += 1
            self.once_finished(image, init_pts, tar_pts, once_loss)
        self.drag_finished()