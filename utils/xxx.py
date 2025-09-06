import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# 替换为：
from timm.layers import DropPath, to_2tuple, trunc_normal_
try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")



