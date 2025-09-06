import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_threshold_ueops(signal, fs, threshold=0.2, skip_points=None, max_echoes=5):
    """
    基于幅值阈值的方法检测多个回波起始点：
      1. 从头开始，找到第一个 |signal| > threshold 的点；
      2. 在该点之后 skip_points 个采样点后，继续寻找下一个回波起始点；
      3. 重复至找到 max_echoes 个或信号末尾。
    返回：所有检测到的起始点索引列表（全局索引）。
    """
    if skip_points is None:
        skip_points = int(1200e-9 * fs)  # 默认隔 1200 ns
    ueops = []
    idx = 0
    for _ in range(max_echoes):
        # 在 idx 之后寻找第一个满足条件的点
        # 使用 np.where 提高速度
        rel = np.where(signal[idx:] > threshold)[0]
        if rel.size == 0:
            break
        ueop = idx + rel[0]
        ueops.append(ueop)
        idx = ueop + skip_points
        if idx >= len(signal):
            break
    return ueops

# --- 主流程 ---
fs = 500e6  # 采样率
target_dir = r"D:\PyCharm\Project\swin_transformer_paper\compare_method_testdata\target"
label_dir  = r"D:\PyCharm\Project\swin_transformer_paper\compare_method_testdata\label"

target_files = sorted(os.listdir(target_dir), key=lambda x: int(x.split('.')[0]))
label_files  = sorted(os.listdir(label_dir),  key=lambda x: int(x.split('.')[0]))

# 存放结果
predicted_tofs = []
true_tofs      = []
abs_errors     = []
rel_errors     = []

for i, (tfile, lfile) in enumerate(zip(target_files, label_files)):
    # 读取信号和真实 TOF（μs）
    signal    = pd.read_csv(os.path.join(target_dir, tfile), header=None).iloc[:,0].dropna().values
    true_tof  = float(pd.read_csv(os.path.join(label_dir, lfile), header=None).iloc[0,0])
    # 检测起始点
    ueops = detect_threshold_ueops(signal, fs, threshold=0.3, skip_points=int(1200e-9*fs))

    # —— 在这里插入绘图 ——
    if i < 3:  # 仅对前三个信号绘图
        plt.figure(figsize=(12, 4))
        plt.plot(signal, label='Signal')
        plt.axhline(0.3, linestyle='--', linewidth=1.5, label='Threshold = 0.3')
        for j, idx in enumerate(ueops):
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.7)
            plt.text(idx, 0.3 * 1.1, f'UEOP {j + 1}', rotation=90,
                     va='bottom', color='r', fontsize=8)
        plt.title(f"Sample {tfile} (#{i + 1})")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()
    # —— 绘图结束 ——


    if len(ueops) < 2:
        # 若回波少于 2 个，无法计算 TOF，跳过
        continue
    # 取前两个回波，计算 TOF（μs）
    dt_samples     = ueops[1] - ueops[0]
    predicted_tof  = dt_samples / fs * 1e6
    # 统计
    predicted_tofs.append(predicted_tof)
    true_tofs.append(true_tof)
    abs_err = np.abs(predicted_tof - true_tof)
    rel_err = abs_err / true_tof if true_tof != 0 else np.nan
    abs_errors.append(abs_err)
    rel_errors.append(rel_err)

# 转为 numpy 数组，方便计算
abs_errors = np.array(abs_errors)
rel_errors = np.array(rel_errors)

# 误差指标
MAE  = np.nanmean(abs_errors)
MARE = np.nanmean(rel_errors) * 100  # 百分比
Max_rel_err = np.nanmax(rel_errors) * 100  # 百分比

print(f"样本数: {len(abs_errors)}")
print(f"MAE (μs):          {MAE:.4f}")
print(f"MARE (%):         {MARE:.2f}%")
print(f"最大相对误差 (%): {Max_rel_err:.2f}%")

# 绘制误差分布柱状图
plt.figure(figsize=(8,5))
# 以相对误差（%）为例
bins = np.linspace(0, np.nanmax(rel_errors)*100, 30)
plt.hist(rel_errors*100, bins=bins, edgecolor='black')
plt.xlabel("相对误差 (%)")
plt.ylabel("样本数")
plt.title("TOF 相对误差分布")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
