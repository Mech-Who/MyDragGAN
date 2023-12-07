import numpy as np

def mean_distance(start_pts, end_pts) -> float:
    """
    Mean Distance（MD）:
    即编辑点与目标点之间的平均距离。
    通过对编辑图像和目标图像之间的关键点位置进行比较，
    计算编辑点与目标点之间的欧氏距离，并取平均值作为MD值。
    """
    distance = 0
    for start, end in zip(start_pts, end_pts):
        x_start, y_start = start
        x_end, y_end = end
        distance += np.sqrt((x_start - x_end) ** 2 + (y_start - y_end) ** 2)
    distance = distance / len(start_pts)
    return distance


if __name__ == "__main__":
    print("mean distance test:")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 5]])
    md = mean_distance(a, b)
    print(f"mean distance: {md:.3f}")

    