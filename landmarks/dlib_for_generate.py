import numpy as np
import argparse
import dlib
import cv2

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
    image_file = "./save_images/generated_images/stylegan2-ffhq-512x512_0.png"

    ###################################################################################################################
    # （1）先检测人脸，然后定位脸部的关键点。优点: 与直接在图像中定位关键点相比，准确度更高。
    detector = dlib.get_frontal_face_detector()			# 1.1、基于dlib的人脸检测器
    predictor = dlib.shape_predictor(shape_predictor)	# 1.2、基于dlib的关键点定位（68个关键点）

    # （2）图像预处理
    # 2.1、读取图像
    image = cv2.imread(image_file)
    (h, w) = image.shape[:2]		# 获取图像的宽和高
    width = 500						# 指定宽度
    r = width / float(w)			# 计算比例
    dim = (width, int(h * r))		# 按比例缩放高度: (宽, 高)
    # 2.2、图像缩放
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # 2.3、灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # （3）人脸检测
    rects = detector(gray, 1)				# 若有多个目标，则返回多个人脸框

    # （4）遍历检测得到的【人脸框 + 关键点】
    # rect: 人脸框
    for rect in rects:		
        # 4.1、定位脸部的关键点（返回的是一个结构体信息，需要遍历提取坐标）
        shape = predictor(gray, rect)
        # 4.2、遍历shape提取坐标并进行格式转换: ndarray
        shape = shape_to_np(shape)
        # 4.3、遍历当前框的所有关键点（name: 脸部位置。i,j表示脸部的坐标。）
        clone = image.copy()
        # 4.4、根据脸部位置画点（每个脸部由多个关键点组成）
        for (x, y) in shape:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

        # 4.5、显示图像
        cv2.imshow("Image", clone)		# 显示带有关键点特征的图像
        cv2.waitKey(0)
