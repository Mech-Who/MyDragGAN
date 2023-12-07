import numpy as np
import dlib
import cv2

from PySide6.QtCore import QPoint

#https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
#http://dlib.net/files/


def shape_to_np(shape, dtype="int"):
	"""获取68个关键点的坐标，并转换为ndarray格式"""
	# （1）创建68*2模板。68个关键点坐标(x,y)
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# （2）遍历每一个关键点, 得到坐标(x,y)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


if __name__ == '__main__':
    ###################################################################################################################
    # 参数设置

    shape_predictor = "./landmarks/shape_predictor_68_face_landmarks.dat"
    origin_file = "./save_images/generated_images/stylegan2-ffhq-512x512_2747.png"
    target_file = "./save_images/generated_images/stylegan2-ffhq-512x512_6197.png"

    ###################################################################################################################
    # （1）先检测人脸，然后定位脸部的关键点。优点: 与直接在图像中定位关键点相比，准确度更高。
    detector = dlib.get_frontal_face_detector()			# 1.1、基于dlib的人脸检测器
    predictor = dlib.shape_predictor(shape_predictor)	# 1.2、基于dlib的关键点定位（68个关键点）

    # （2）图像预处理
    # 2.1、读取图像
    origin_image = cv2.imread(origin_file)
    target_image = cv2.imread(target_file)

    width = 512 					    # 指定宽度

    (o_h, o_w) = origin_image.shape[:2]	# 获取图像的宽和高
    o_r = width / float(o_w)			# 计算比例
    o_dim = (width, int(o_h * o_r))		# 按比例缩放高度: (宽, 高)

    (t_h, t_w) = target_image.shape[:2]	# 获取图像的宽和高
    t_r = width / float(t_w)			# 计算比例
    t_dim = (width, int(t_h * t_r))		# 按比例缩放高度: (宽, 高)
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
        o_shape = shape_to_np(o_shape)
        t_shape = shape_to_np(t_shape)
        # 4.3、遍历当前框的所有关键点（name: 脸部位置。i,j表示脸部的坐标。）
        o_clone = origin_image.copy()
        t_clone = target_image.copy()
        # 4.4、根据脸部位置画点（每个脸部由多个关键点组成）
        for (x, y) in o_shape:
            cv2.circle(o_clone, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in t_shape:
            cv2.circle(t_clone, (x, y), 3, (0, 255, 0), -1)

        points = []
        for (o_x, o_y), (t_x, t_y) in zip(o_shape, t_shape):
            points.append(QPoint(o_x, o_y))
            points.append(QPoint(t_x, t_y))
        print(points)
        # 4.5、显示图像
        cv2.imshow("Origin Image", o_clone)		# 显示带有关键点特征的图像
        cv2.imshow("Target Image", t_clone)		# 显示带有关键点特征的图像
        cv2.waitKey(0)
