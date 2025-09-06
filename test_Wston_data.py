import shutil

from torchvision.utils import save_image
import torch
from utils.Dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from PIL import Image
import torchvision.transforms as transforms

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from swin_transformer_model import my_swin_tiny_patch4_window7_224
from swin_transformer_model import get_dataloaders
from swin_transformer_model import test
from PIL import Image
import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font support
plt.rcParams['axes.unicode_minus'] = False     # Normal display of minus signs


# ==== 1. Load image ====
image_path = 'datasets/Watson_data/2.56us_2.jpg'  # Ensure path is correct
image = Image.open(image_path).convert('RGB')  # Convert to RGB mode

# ==== 2. Define preprocessing (consistent with training) ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Swin model requires input size 224x224
    transforms.ToTensor(),         # Convert to [C, H, W] Tensor
    # transforms.Normalize(           # Use ImageNet mean and std (or your training values)
    #     mean=[0.485, 0.456, 0.406],
    #     std =[0.229, 0.224, 0.225]
    # )
])

input_tensor = transform(image).unsqueeze(0)  # Add batch dimension => [1, 3, 224, 224]

# ==== 3. Model loading ====
model, device = my_swin_tiny_patch4_window7_224(num_classes=1)
# model.load_state_dict(torch.load('saved_model/swin_best_model.pth'))
model.load_state_dict(torch.load('saved_model/stage1_best_2.pth'))
# model.load_state_dict(torch.load('saved_model/swin_best_model_v10001_20250826_002155.pth'))

model.to(device)
model.eval()

# ==== 4. Inference ====
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    prediction = model(input_tensor)
    predicted_value = prediction.item()  # Extract value

print(f"🧠 Model prediction: {predicted_value:.4f}")


