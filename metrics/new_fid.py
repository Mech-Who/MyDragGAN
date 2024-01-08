import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = r'C:\UserData\Projects\Homework\MyDragGAN\save_images\experience_origin\ffhq_512\optimize\one_point\3.0\13\20\sixth_block'
generated_images_folder = r'C:\UserData\Projects\Homework\MyDragGAN\save_images\experience_result\ffhq_512\optimize\one_point\3.0\13\20\sixth_block'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 batch_size=8,
                                                 device='cuda',
                                                 dims=2048,
                                                 num_workers=0)

print('FID value:', fid_value)

# scipy 1.11.1