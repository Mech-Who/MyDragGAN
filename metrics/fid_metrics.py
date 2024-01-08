# pip install pytorch_fid

def frechet_inception_distance():
    """
    Fréchet Inception Distance （FID）：
    计算编辑图像与初始图像之间的分布差异。
    FID使用预训练的深度学习模型来度量两个图像集合之间的分布差异，
    较低的FID值表示编辑图像与初始图像的分布更相似。
    """
    pass


if __name__ == "__main__":
    print("frechet inception distance test:")

    ## 一、准备图像数据集
    import os
    import torchvision.datasets as datasets

    # TODO: 还存在问题，暂时不能用
    # datasets.CIFAR10(root='E:\\Datasets\\CIFAR10', train=True, download=True)

    # 设置数据集路径
    data_dir = '.'

    # 创建真实图像数据集加载器
    real_dataset = datasets.ImageFolder(os.path.join(data_dir, 'input_images'))

    # 创建生成图像数据集加载器
    generated_dataset = datasets.ImageFolder(os.path.join(data_dir, 'output_images'))

    ## 二、提取图像特征
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models

    device = 'cuda'

    # 创建Inception网络模型
    inception_model = models.inception_v3(pretrained=True, transform_input=True).to(device)

    # 设置图像预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(299),  # 将图像大小调整为299x299
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])

    # 定义函数来提取图像特征
    def extract_features(dataset, model):
        features = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        for images, _ in loader:
            images = images.to(device)
            features.append(model(images).detach().cpu().numpy())  # 提取特征并转换为NumPy数组
        return np.concatenate(features, axis=0)

    # 提取真实图像数据集和生成图像数据集的特征
    real_features = extract_features(real_dataset, inception_model)
    generated_features = extract_features(generated_dataset, inception_model)

    ## 三、计算特征的统计数据
    import numpy as np

    # 计算真实图像数据集和生成图像数据集的特征的均值
    real_mean = np.mean(real_features, axis=0)
    generated_mean = np.mean(generated_features, axis=0)

    # 计算真实图像数据集和生成图像数据集的特征的协方差矩阵
    real_cov = np.cov(real_features, rowvar=False)
    generated_cov = np.cov(generated_features, rowvar=False)

    ## 四、计算FID分数
    from scipy.linalg import sqrtm

    # 计算FID分数
    fid_score = np.sum((real_mean - generated_mean) ** 2) + np.trace(real_cov + generated_cov - 2 * sqrtm(np.dot(real_cov, generated_cov)))

