import copy
import utils
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_F
from stylegan2_ada.model import StyleGAN
from metrics.md_metrics import mean_distance

from PySide6.QtCore import QPoint, QThread, Signal


class DragGAN:
    DEFAULT_STEP_SIZE = 2e-3
    DEFAULT_R1 = 3
    DEFAULT_R2 = 13
    DEFAULT_R3 = 40
    def __init__(self):

        #### model部分初始化 ####
        # 设置device可选值并设默认值
        self.device = "cpu"
        # 设置模型并设置默认值
        self.model = StyleGAN(device=self.device)

        self.pickle_path = ""
        self.w_load = None
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
        self.r3 = self.DEFAULT_R3
        self.steps = 0
        self.isDragging = False
        self.showPoints = False

        self.is_optimize = False

        # experience
        self.only_one_point = False
        self.five_points = False
        self.sixty_eight_points = True

        self.fourth_block = False
        self.fifth_block = False
        self.sixth_block = True
        self.seventh_block = False

        self.test_times = 1
        self.drag_times = 200

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

    def generateImage(self, seed, w_plus=True, w_load=None):
        if self.pickle_path:
            # 将opt设置为None, 表示开始一次新的优化
            self.optimizer = None

            # seed -> w -> image(torch.Tensor)
            self.W = self.model.gen_w(seed, w_plus, w_load)
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
        temp_W = self.W.cpu().detach().numpy().copy()
        self.W = torch.from_numpy(temp_W).to(self.device).float()
        self.W.requires_grad_(False)

        layer = 6
        if self.fourth_block:
            layer = 4
        elif self.fifth_block:
            layer = 5
        elif self.sixth_block:
            layer = 6
        elif self.seventh_block:
            layer = 7

        self.W_layers_to_optimize = self.W[:, :layer, :].detach().clone().requires_grad_(True)
        self.W_layers_to_fixed = self.W[:, layer:, :].detach().clone().requires_grad_(False)

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
            return False, (0, mean_distance(init_pts.cpu().numpy(), tar_pts.cpu().numpy()))
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
        md = mean_distance(new_init_pts.cpu().numpy(), tar_pts.cpu().numpy())
        loss = loss.item()
        print(f"tar pts: {tar_pts.cpu().numpy()}, new init pts: {new_init_pts.cpu().numpy()}, Loss: {loss:0.4f}, Mean distance: {md}\n")
        # print("update init_pts as Point Tracking")

        return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_img, loss, md)
        # return True, (np.array([0, 0]), np.array([1, 1]), np.array([0, 1]))


class DragThread(QThread):
    drag_finished = Signal()
    once_finished = Signal(torch.Tensor, list, float, int)

    def __init__(self, draggan_model, image_widget):
        super().__init__()
        self.DragGAN = draggan_model
        self.image_widget = image_widget

    def drag(self):
        points = self.image_widget.get_points()
        if len(points) < 2:
            return
        if len(points) % 2 == 1:
            points = points[:-1]
        init_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 0])
        tar_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 1])

        if self.DragGAN.is_optimize:
            print("get in optimize")
            img = self.image_widget.get_image()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"img shape:{img.shape}")
            scale = self.image_widget.get_image_scale()
            new_init_pts = []
            # 获取输入点
            for point in init_pts:
                # 顶点是原图点，图片是未缩放的原图，无需缩放
                # point = (int(point[0] / scale), int(point[1] / scale))

                left_up_y = int(point[0]-self.DragGAN.r3)
                left_up_x = int(point[1]-self.DragGAN.r3)
                right_down_y = int(point[0]+self.DragGAN.r3)
                right_down_x = int(point[1]+self.DragGAN.r3)
                crop_img = img[left_up_x:right_down_x, left_up_y:right_down_y]
                # 边缘检测
                # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                gauss = cv2.GaussianBlur(crop_img,(3,3),0)
                canny = cv2.Canny(gauss, 50, 150)
                # 获得截取后的图片中心
                center = (int(crop_img.shape[0]/2), int(crop_img.shape[1]/2))
                # 获得边缘点
                contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                all_points = [point[0] for contour in contours for point in contour ]
                # 计算边缘点与中心点距离
                distance = [(point[0] - center[0])**2 + (point[1] - center[1])**2 for point in all_points]
                # 求距离最小的点的index
                min_index = distance.index(min(distance))
                # 找出最小点
                min_point = all_points[min_index]

                # 更新最小点为init点
                rate = 0.8 # 更新比率，0~1， 表示不完全更新到轮廓上

                target_point = (left_up_x + int(center[0] + (min_point[0] - center[0]) * rate), left_up_y + int(center[1] + (min_point[1] - center[1]) * rate))
                # cv2.circle(img, (point[1], point[0]), 1, (0, 0, 255), 4)
                # cv2.circle(img, (target_point[1], target_point[0]), 1, (0, 255, 0), 4)
                # target_point = (int(target_point[0] * scale), int(target_point[1] * scale))
                target_point = (target_point[1], target_point[0])
                # cv2.imshow("img", img)
                # cv2.moveWindow("img", 500, 100)
                # cv2.imshow("canny", canny)
                # cv2.moveWindow("canny", 100, 100)
                # cv2.imshow("crop", crop_img)
                # cv2.moveWindow("crop", 300, 100)
                # cv2.waitKey(0)
                new_init_pts.append(target_point)

            new_init_pts = np.array(new_init_pts)
            print(f"origin points: {init_pts}")
            print(f"new points: {new_init_pts}")
            print(f"scale: {scale}")
            init_pts = new_init_pts
            # 显示最新的图像  
            updated_points = []
            for i in range(init_pts.shape[0]):
                updated_points.append(QPoint(int(init_pts[i][0]), int(init_pts[i][1])))
                updated_points.append(QPoint(int(tar_pts[i][0]), int(tar_pts[i][1])))
            self.image_widget.set_points(updated_points)

    
        init_pts = np.vstack(init_pts)[:, ::-1].copy()
        tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
        self.DragGAN.prepare2Drag(init_pts, lr=self.DragGAN.step_size)
        
        self.DragGAN.steps = 0
        while(self.DragGAN.isDragging):
            # 迭代一次
            status, ret = self.DragGAN.drag(init_pts, tar_pts, allow_error_px=5, r1=3, r2=13)
            if status:
                init_pts, _, image, once_loss, md = ret
            else:
                self.DragGAN.isDragging = False
                return
            # 显示最新的图像  
            points = []
            for i in range(init_pts.shape[0]):
                points.append(QPoint(int(init_pts[i][1]), int(init_pts[i][0])))
                points.append(QPoint(int(tar_pts[i][1]), int(tar_pts[i][0])))

            self.DragGAN.steps += 1
            self.once_finished.emit(image, points, once_loss, self.DragGAN.steps)
        self.drag_finished.emit()

    def run(self):
        self.drag()


class ExperienceThread(QThread):
    experience_start = Signal()
    random_seed = Signal(int)
    experience_finished = Signal()
    once_finished = Signal(torch.Tensor, list, float, int)

    def __init__(self, draggan_model, image_widget):
        super().__init__()
        self.DragGAN = draggan_model
        self.image_widget = image_widget

    def generate(self):
        import random
        self.DragGAN.loadCpkt(self.DragGAN.pickle_path)

        if self.DragGAN.random_seed:
            self.DragGAN.seed = random.randint(self.DragGAN.min_seed, self.DragGAN.max_seed)
            self.random_seed.emit(self.DragGAN.seed)
        image = self.DragGAN.generateImage(self.DragGAN.seed, self.DragGAN.w_plus) # 3 * 512 * 512
        if image is not None:
            self.image_widget.set_image_from_array(image)

    def drag_once(self, index):
        points = self.image_widget.get_points()
        if len(points) < 2:
            return
        if len(points) % 2 == 1:
            points = points[:-1]
        init_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 0])
        tar_pts = np.array([[point.x(), point.y()] for index, point in enumerate(points) if index % 2 == 1])

        if self.DragGAN.is_optimize:
            try:
                print("get in experience optimize")
                img = self.image_widget.get_image()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"img shape:{img.shape}")
                scale = self.image_widget.get_image_scale()
                new_init_pts = []
                # 获取输入点
                for point in init_pts:
                    # 顶点是原图点，图片是未缩放的原图，无需缩放
                    # point = (int(point[0] / scale), int(point[1] / scale))

                    left_up_y = int(point[0]-self.DragGAN.r3)
                    left_up_x = int(point[1]-self.DragGAN.r3)
                    right_down_y = int(point[0]+self.DragGAN.r3)
                    right_down_x = int(point[1]+self.DragGAN.r3)
                    crop_img = img[left_up_x:right_down_x, left_up_y:right_down_y]
                    # 边缘检测
                    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    gauss = cv2.GaussianBlur(crop_img,(3,3),0)
                    canny = cv2.Canny(gauss, 50, 150)
                    # 获得截取后的图片中心
                    center = (int(crop_img.shape[0]/2), int(crop_img.shape[1]/2))
                    # 获得边缘点
                    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 0:
                        new_init_pts.append(point)
                        continue
                    all_points = [point[0] for contour in contours for point in contour ]
                    # 计算边缘点与中心点距离
                    distance = [(point[0] - center[0])**2 + (point[1] - center[1])**2 for point in all_points]
                    # 求距离最小的点的index
                    min_index = distance.index(min(distance))
                    # 找出最小点
                    min_point = all_points[min_index]

                    # 更新最小点为init点
                    rate = 0.8 # 更新比率，0~1， 表示不完全更新到轮廓上
                    target_point = (left_up_x + int(center[0] + (min_point[0] - center[0]) * rate), left_up_y + int(center[1] + (min_point[1] - center[1]) * rate))
                    target_point = (target_point[1], target_point[0])
                    new_init_pts.append(target_point)
                new_init_pts = np.array(new_init_pts)
                print(f"origin points: {init_pts}")
                print(f"new points: {new_init_pts}")
                print(f"scale: {scale}")
                init_pts = new_init_pts
                # 显示最新的图像  
                updated_points = []
                for i in range(init_pts.shape[0]):
                    updated_points.append(QPoint(int(init_pts[i][0]), int(init_pts[i][1])))
                    updated_points.append(QPoint(int(tar_pts[i][0]), int(tar_pts[i][1])))
                self.image_widget.set_points(updated_points)
            except Exception as e:
                print(f"Error:{e}")

        init_pts = np.vstack(init_pts)[:, ::-1].copy()
        tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
        self.DragGAN.prepare2Drag(init_pts, lr=self.DragGAN.step_size)
        
        self.DragGAN.steps = 0
        for i in range(self.DragGAN.drag_times):
            print(f"current[{index+1}/{self.DragGAN.test_times}] drag times [{i+1}/{self.DragGAN.drag_times}]")
            if not self.DragGAN.isDragging:
                break
            # 迭代一次
            try:
                status, ret = self.DragGAN.drag(init_pts, tar_pts, allow_error_px=5, r1=3, r2=13)
                if status:
                    init_pts, _, image, once_loss, once_md = ret
                else:
                    self.DragGAN.isDragging = False
                    loss, md = ret
                    return (loss, md)
            except Exception as e:
                print(f"Error:{e}")
                self.DragGAN.isDragging = False
                return None
            # 显示最新的图像  
            points = []
            for i in range(init_pts.shape[0]):
                points.append(QPoint(int(init_pts[i][1]), int(init_pts[i][0])))
                points.append(QPoint(int(tar_pts[i][1]), int(tar_pts[i][0])))

            self.DragGAN.steps += 1
            self.once_finished.emit(image, points, once_loss, self.DragGAN.steps)
        print(f"current[{index+1}/{self.DragGAN.test_times}] {self.DragGAN.drag_times} times experience: loss: {once_loss}, mean_distance: {once_md}")
        return (once_loss, once_md)

    def saveImage(self, dir_name, pickle, image_format):
        image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_images", dir_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        path = os.path.join(image_dir, f"{pickle}_{self.DragGAN.seed}.{image_format}")
        self.image_widget.save_image(path, image_format, 100, is_experience=True)
        print(f"save target image as {path}")
        return path

    def get_experience_dir(self):
        optimize = None
        if self.DragGAN.is_optimize:
            optimize = "optimize"
        else:
            optimize = "not_optimize"

        num_points = None
        if self.DragGAN.only_one_point:
            num_points = "one_point"
        elif self.DragGAN.five_points:
            num_points = "five_point"
        elif self.DragGAN.sixty_eight_points:
            num_points = "sixty_eight_point"

        r1 = str(self.DragGAN.r1)
        r2 = str(self.DragGAN.r2)
        r3 = str(self.DragGAN.r3)

        block = None
        if self.DragGAN.fourth_block:
            block = "fourth_block"
        elif self.DragGAN.fifth_block:
            block = "fifth_block"
        elif self.DragGAN.sixth_block:
            block = "sixth_block"
        elif self.DragGAN.seventh_block:
            block = "seventh_block"
        
        pickle_name = os.path.basename(self.DragGAN.pickle_path)
        if pickle_name == "stylegan2_dogs_1024_pytorch.pkl":
            pickle_name = "dogs_1024"
        elif pickle_name == "stylegan2_elephants_512_pytorch.pkl":
            pickle_name = "elephants_512"
        elif pickle_name == "stylegan2_horses_256_pytorch.pkl":
            pickle_name = "horses_256"
        elif pickle_name == "stylegan2_lions_512_pytorch.pkl":
            pickle_name = "lions_512"
        elif pickle_name == "stylegan2-afhqcat-512x512.pkl":
            pickle_name = "afhqcat_512"
        elif pickle_name == "stylegan2-ffhq-512x512.pkl":
            pickle_name = "ffhq_512"
        else:
            pickle_name = "other"

        experience_dir = os.path.join(pickle_name, optimize, num_points, r1, r2, r3, block)
        print(f"experience_dir: {experience_dir}")
        return experience_dir

    def face_detection(self, origin_file, target_file, shape_predictor):
            import dlib
            import cv2
            ###################################################################################################################
            # （1）先检测人脸，然后定位脸部的关键点。优点: 与直接在图像中定位关键点相比，准确度更高。
            detector = dlib.get_frontal_face_detector()			# 1.1、基于dlib的人脸检测器
            predictor = dlib.shape_predictor(shape_predictor)	# 1.2、基于dlib的关键点定位（68个关键点）
            # （2）图像预处理
            # 2.1、读取图像
            origin_image = cv2.imread(origin_file)
            target_image = cv2.imread(target_file)

            q_size = self.image_widget.size()
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
            
            return (origin_rects, target_rects, origin_gray, target_gray, o_r, t_r, predictor)

    def get_shape_predictor(self):
        dat_68 = "./landmarks/shape_predictor_68_face_landmarks.dat"
        dat_5 = "./landmarks/shape_predictor_5_face_landmarks.dat"

        shape_predictor = ""
        if self.DragGAN.only_one_point:
            shape_predictor = dat_5
        if self.DragGAN.five_points:
            shape_predictor = dat_5
        if self.DragGAN.sixty_eight_points:
            shape_predictor = dat_68
        
        return shape_predictor

    def get_generated_images(self, experience_dir, pickle, image_format):
        # 生成目标图像
        self.generate()
        # 保存图片
        target_filename = self.saveImage(f"experience_target/{experience_dir}", pickle, image_format)

        # 生成源图像
        self.generate()
        # 保存图片
        origin_filename = self.saveImage(f"experience_origin/{experience_dir}", pickle, image_format)

        return (origin_filename, target_filename)

    def get_input_points(self, origin_rects, target_rects, origin_gray, target_gray, o_r, t_r, predictor):
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
                o_x, o_y = o_shape[4]
                t_x, t_y = t_shape[4]
                points.append(QPoint(int(o_x/o_r), int(o_y/o_r)))
                points.append(QPoint(int(t_x/t_r), int(t_y/t_r)))
            else:
                for (o_x, o_y), (t_x, t_y) in zip(o_shape, t_shape):
                    points.append(QPoint(int(o_x/o_r), int(o_y/o_r)))
                    points.append(QPoint(int(t_x/t_r), int(t_y/t_r)))
            self.image_widget.add_points(points)

    def experience_once(self):
        
        import time

        pickle = os.path.basename(self.DragGAN.pickle_path).split(os.extsep)[0]
        image_format = "png"

        sum_results = []
        total_time = 0

        self.experience_start.emit()

        experience_dir = self.get_experience_dir()
        
        result = None

        for i in range(self.DragGAN.test_times):

            origin_filename, target_filename = self.get_generated_images(experience_dir, pickle, image_format)

            # 参数设置
            shape_predictor = self.get_shape_predictor()
            origin_file = origin_filename
            target_file = target_filename
            ret = self.face_detection(origin_file, target_file, shape_predictor)

            self.get_input_points(*ret)

            if self.DragGAN.isDragging:
                print("dragging is running!")
                break
            self.DragGAN.isDragging = True
            time_start = time.time()
            res = self.drag_once(i)
            if res is not None:
                result = res
            if result is not None:
                sum_results.append(result)
            time_end = time.time()
            total_time += (time_end - time_start)/1000000 # s
            self.DragGAN.isDragging = False

            # 保存图片
            result_filename = self.saveImage(f"experience_result/{experience_dir}", pickle, image_format)
            # 清空画布
            self.image_widget.clear_points()
        if len(sum_results) <= 0:
            print("sum_results is empty!")
            return
        avg_loss = sum([loss for loss, _ in sum_results])/len(sum_results),
        avg_md = sum([md for _, md in sum_results])/len(sum_results)
        avg_time = total_time/self.DragGAN.test_times
        print(f"Experience is finished!\n avg_result: loss: {avg_loss}, mean_distance: {avg_md}, avg_time: {avg_time}")
        self.experience_finished.emit()
    
    def run(self):
        self.experience_once()