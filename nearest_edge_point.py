import cv2

# 截取图片
def get_rect(img, point, r3):
    return img[point[0]-r3: point[0]+r3, point[1]-r3: point[1]+r3]

# 边缘检测
def edge_detection(img, gauss_kernel=(3, 3), gauss_sigma_x=0, canny_low_threshold=50, canny_high_threshold=150):
    gauss = cv2.GaussianBlur(img, gauss_kernel, gauss_sigma_x)
    canny = cv2.Canny(gauss, canny_low_threshold, canny_high_threshold)
    return canny

# 获得边缘点
def get_edge_points(canny):
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_points = [point[0] for contour in contours for point in contour ]
    return all_points

# 查找最近边缘点
def find_nearest_edge_point(all_points, center):
    # 计算边缘点与中心点距离
    distance = [(point[0] - center[0])**2 + (point[1] - center[1])**2 for point in all_points]
    # 求距离最小的点的index
    min_index = distance.index(min(distance))
    # 找出最小点
    min_point = all_points[min_index]
    return min_point

# 更新最小点为init点
def calculate_new_point(rate, center, min_point, origin_x=0, origin_y=1):
    target_point = (int(origin_x + center[0] + (min_point[0] - center[0]) * rate), int(origin_y + center[1] + (min_point[1] - center[1]) * rate))
    return target_point

# 绘制可视化
def tool_visualize(crop_img, canny, center, min_point, target_point):
    # 绘制用户点
    cv2.circle(crop_img, center, 1, (0, 0, 255), 4)
    cv2.circle(canny, center, 1, (0, 0, 255), 4)
    cv2.circle(crop_img, min_point, 1, (0, 0, 255), 4)
    cv2.circle(canny, min_point, 1, (0, 0, 255), 4)
    cv2.circle(crop_img, target_point, 1, (0, 255, 0), 4)
    cv2.circle(canny, target_point, 1, (0, 255, 0), 4)

if __name__ == "__main__":
    # 读取test图片
    img = cv2.imread("C:/UserData/Projects/Homework/MyDragGAN/test.jpg") # 1253(y) * 1880(x) * 3
    print(img.shape)
    # 获取坐标点数据
    user_input = (590, 1280)
    r3 = 100
    crop_img = get_rect(img, user_input, r3)
    center = (int(crop_img.shape[0]/2), int(crop_img.shape[1]/2))
    canny = edge_detection(crop_img)
    all_points = get_edge_points(canny)
    min_point = find_nearest_edge_point(all_points, center)
    rate = 0.8 # 更新比率，0~1， 表示不完全更新到轮廓上
    target_point = calculate_new_point(rate, center, min_point)

    tool_visualize(crop_img, canny, center, min_point, target_point)

    # 显示图片
    # cv2.imshow("origin", img)
    cv2.imshow("canny", canny)
    cv2.moveWindow("canny", 100, 100)
    cv2.imshow("crop", crop_img)
    cv2.moveWindow("crop", 300, 100)

    cv2.waitKey(0)