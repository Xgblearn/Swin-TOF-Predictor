
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号


def detect_peak_ueops_mask(signal, fs, skip_time=1200e-9, max_echoes=5):
    """
    基于“全局最大 → 屏蔽窗口 → 再找”方法检测回波峰值位置：

    signal    : 原始波形 (1D numpy array)
    fs        : 采样率 (Hz)
    skip_time : 每个峰值左右屏蔽的时间（秒），默认 1200 ns
    max_echoes: 最多检测几个峰

    返回：峰值索引列表（全局索引）
    """
    skip_pts = int(skip_time * fs)  # 600 点左右
    ueops = []
    masked = signal.copy()

    for _ in range(max_echoes):
        # 找出当前“未屏蔽”区域的最大值位置
        idx = np.argmax(masked)
        amp = masked[idx]
        # 如果连噪声都比这个大，说明没峰可找
        if amp <= 0:
            break

        ueops.append(idx)

        # 屏蔽这个峰左右 skip_pts 区域，避免复选
        left = max(idx - skip_pts, 0)
        right = min(idx + skip_pts, len(signal))
        masked[left:right] = 0

    return ueops
