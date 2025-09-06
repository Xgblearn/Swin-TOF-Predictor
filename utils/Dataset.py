# import os
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
#
# import os
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
#
# class MyDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         初始化数据集
#         :param data_dir: 图片所在文件夹路径
#         :param transform: 图像预处理变换
#         """
#         self.data_dir = data_dir
#         # 获取图片文件名并排序
#         self.image_names = sorted(
#             [f for f in os.listdir(data_dir) if f.startswith('GADF_Image')]
#         )
#         self.transform = transform
#
#     def __len__(self):
#         """
#         返回数据集大小
#         """
#         return len(self.image_names)
#
#     def __getitem__(self, idx):
#         """
#         获取指定索引的图片数据
#         :param idx: 索引
#         :return: 图像张量和对应标签（如果需要）
#         """
#         # 获取图片路径
#         image_path = os.path.join(self.data_dir, self.image_names[idx])
#
#         # 加载图片
#         image = Image.open(image_path).convert('RGB')  # 转换为RGB格式
#
#         # 如果有变换，应用变换
#         if self.transform:
#             image = self.transform(image)
#
#         # 示例标签（可自定义）：这里假设图片名最后几位的数字部分为标签
#         # label = int(self.image_names[idx].split('_')[-1])
#
#         # return image, label
#         return image
#
# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整到统一大小
#     transforms.ToTensor(),  # 转为Tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
# ])
#
# # 数据集路径
# data_dir = 'data/target'
#
# # 创建数据集实例
# dataset = MyDataset(data_dir=data_dir, transform=transform)
#
# # 创建数据加载器
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# # 测试加载
# for images in dataloader:
# # for images, labels in dataloader:
#     print("批次图像大小:", images.size())  # 应输出 (batch_size, 3, 224, 224)
#     # print("批次标签:", labels)
#     break
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
#
# # 遍历数据加载器
# for images in dataloader:
#     print("批次图像大小:", images.size())  # 应输出 (batch_size, 3, 224, 224)
#
#     # 将第一个图像还原为可视化格式
#     # 数据需要逆归一化以还原原始像素值
#     inv_transform = transforms.Normalize(
#         mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
#         std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
#     )
#
#     # 取第一张图片
#     image = images[2]
#     image = inv_transform(image)  # 逆归一化
#     image = image.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 格式供 matplotlib 使用
#
#     # 显示图片
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()
#     break

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, indices, transform=None):
        """
        Initialize dataset
        :param image_dir: directory of images
        :param label_dir: directory of text labels
        :param transform: optional image transforms
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        # self.image_names = sorted(
        #     [f for f in os.listdir(image_dir) if f.startswith('GADF_Image')]
        # )
        # self.label_names = sorted(
        #     [f for f in os.listdir(label_dir)]
        # )
        #
        # 根据指定的编号筛选文件名
        self.image_names = [f"GASF_Image_{idx}.jpg" for idx in indices]
        self.label_names = [f"{idx}.txt" for idx in indices]

        self.transform = transform

    def __len__(self):
        """
        Return dataset length
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Get image and label for index
        :param idx: index
        :return: image tensor and label
        """
        # 加载图片
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')  # convert to RGB
        # resize to 224x224
        image = image.resize((224, 224))
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # default to tensor
        # 加载对应的文本标签
        label_path = os.path.join(self.label_dir, self.label_names[idx])
        with open(label_path, 'r') as f:
            # read first line
            line = f.readline().strip()

            # strip commas and take first column
            line = line.replace(',', '')

            label = float(line.split()[0])
        # to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

# 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转为Tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
# ])

# # 数据集路径
# image_dir = 'data/target'
# label_dir = 'data/label'
#
# # 创建数据集实例
# dataset = MyDataset(image_dir=image_dir, label_dir=label_dir)
#
# # 创建数据加载器
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# # 测试加载
# for images, labels in dataloader:
#     print("批次图像大小:", images.size())  # 输出 (batch_size, 3, 224, 224)
#     print("批次标签:", labels)  # 输出对应标签
#     print("批次标签大小:", labels.size())  # 输出对应标签
#     break



# # 数据集路径
# image_dir = 'data/target'
# label_dir = 'data/label'
#
# # 指定编号范围（2501 到 3500）
# indices = list(range(2501, 2508))
# # 创建数据集实例
# dataset = MyDataset(image_dir=image_dir, label_dir=label_dir, indices=indices)
#
# # 创建数据加载器
# dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
# # 测试加载
# for images, labels in dataloader:
#     print("批次图像大小:", images.size())  # 输出 (batch_size, 3, 224, 224)
#     print("批次标签:", labels)  # 输出对应标签
#     break
