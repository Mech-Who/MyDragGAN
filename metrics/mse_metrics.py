

def mean_square_error():
    """
    Mean Squared Error （MSE）：
    计算编辑点与目标点之间的平均欧氏距离。
    较低的MSE值表示编辑点更接近目标点。
    """
    pass


if __name__ == "__main__":
    print("mean square error test:")

    import torch
    import torch.nn.functional as F

    # 创建真实值和预测值的张量
    y_true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y_pred = torch.tensor([2, 3, 4, 5, 6], dtype=torch.float32)

    # 使用PyTorch的mse_loss函数计算MSE
    mse = F.mse_loss(y_pred, y_true)

    print(f"mse: {mse}")
