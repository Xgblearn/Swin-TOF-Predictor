import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from Dataset import MyDataset
from .losses import log_cosh_loss
import torchvision.transforms as transforms
from PIL import Image
class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, indices=None, transform=None):
        """
        image_dir: 图片文件夹
        label_dir: 标签文件夹
        indices: 可选，指定编号列表
        transform: 图像预处理
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 获取所有jpg文件名和txt文件名
        all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        all_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        # 提取文件编号
        def get_index(filename):
            return int(''.join(filter(str.isdigit, filename)))

        # 自动匹配编号文件
        image_dict = {get_index(f): f for f in all_images}
        label_dict = {get_index(f): f for f in all_labels}

        # 如果指定 indices，只保留对应编号
        if indices is not None:
            indices = [i for i in indices if i in image_dict and i in label_dict]
        else:
            # 否则取所有编号的交集并排序
            indices = sorted(set(image_dict.keys()) & set(label_dict.keys()))

        self.indices = indices
        self.image_names = [image_dict[i] for i in indices]
        self.label_names = [label_dict[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 读取图像
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 读取标签
        label_path = os.path.join(self.label_dir, self.label_names[idx])
        with open(label_path, 'r') as f:
            line = f.readline().strip().replace(',', '')
            label = float(line.split()[0])

        label = torch.tensor(label, dtype=torch.float32)
        return image, label





# -------------------- 训练 -------------------- #
def train_one_epoch(model, train_loader, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", ncols=100)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = log_cosh_loss(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    return running_loss / len(train_loader)

# -------------------- 验证 -------------------- #
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = log_cosh_loss(outputs.squeeze(), labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# -------------------- metrics -------------------- #
def calculate_metrics(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    non_zero_mask = all_labels != 0
    if np.any(non_zero_mask):
        relative_errors = np.abs((all_preds[non_zero_mask] - all_labels[non_zero_mask]) / all_labels[non_zero_mask])
        mare = np.mean(relative_errors)
        max_re = np.max(relative_errors)
    else:
        mare, max_re = float('nan'), float('nan')

    return mae, mse, r2, mare, max_re

# -------------------- dataloader -------------------- #
def get_dataloaders(image_dir='data/target1',
                    label_dir='data/label',
                    indices=range(1, 1200),
                    batch_size=32,
                    generator=None,
                    save_test_data=False,
                    save_image_dir='saved_data/test/images',
                    save_label_dir='saved_data/test/labels'):
    dataset = MyDataset(image_dir=image_dir, label_dir=label_dir, indices=indices)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保存测试集
    if save_test_data:
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)

        for idx in test_dataset.indices:
            # idx 是 dataset 的索引，需要映射到文件名
            img_name = dataset.image_names[idx]
            lbl_name = dataset.label_names[idx]
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(save_image_dir, img_name))
            shutil.copy(os.path.join(label_dir, lbl_name), os.path.join(save_label_dir, lbl_name))

    #################用于仿真数据的###########################
    # 保存测试集
    # if save_test_data:
    #     os.makedirs(save_image_dir, exist_ok=True)
    #     os.makedirs(save_label_dir, exist_ok=True)
    #     test_indices = test_dataset.indices
    #     for idx in test_indices:
    #         img_name = f'GASF_Image_{idx}.jpg'
    #         lbl_name = f'{idx}.txt'
    #         shutil.copy(os.path.join(image_dir, img_name), os.path.join(save_image_dir, img_name))
    #         shutil.copy(os.path.join(label_dir, lbl_name), os.path.join(save_label_dir, lbl_name))
    #################用于仿真数据的###########################
    return train_loader, val_loader, test_loader
