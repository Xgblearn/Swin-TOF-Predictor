# run_odes_aic_eval.py
import os
import numpy as np
import pandas as pd

# ---------------------
# 把你给的函数粘贴过来（没有改动逻辑）
# ---------------------
def compute_odes(signal, window=200):
    odes = np.zeros_like(signal, dtype=float)
    for k in range(window, len(signal)):
        seg = signal[k-window:k]
        psd = np.abs(np.fft.fft(seg, n=2*window))**2
        half = len(psd) // 2
        max_idx = np.argmax(psd[:half])
        low = max(max_idx - window//4, 0)
        high = min(max_idx + window//4, half)
        dom_energy = np.sum(psd[low:high])
        non_dom_energy = np.sum(psd[:half]) - dom_energy
        odes[k] = np.log(dom_energy / (non_dom_energy + 1e-12))
    return odes

def detect_rough_ueop(odes, threshold_ratio=0.5):
    threshold = threshold_ratio * np.max(odes)
    idx = np.where(odes > threshold)[0]
    return idx[0] if len(idx) > 0 else None

def aic_picker(signal, center_idx, L=100):
    start = max(center_idx - L // 2, 1)
    end = min(center_idx + L // 2, len(signal) - 1)
    aic = []
    for k in range(start, end):
        len1 = k - start
        len2 = end - k
        if len1 > 1 and len2 > 1:
            var1 = np.var(signal[start:k])
            var2 = np.var(signal[k:end])
            # aic_val = k * np.log(var1 + 1e-12) + (end - k) * np.log(var2 + 1e-12)
            aic_val = len1 * np.log(var1 + 1e-12) + len2 * np.log(var2 + 1e-12)
        else:
            aic_val = np.inf
        aic.append(aic_val)
    best_k = np.argmin(aic) + start
    return best_k

def estimate_pulse_width(signal, ueop, fs, threshold_ratio=0.1):
    window = 100
    segment = signal[ueop:ueop+window]
    max_amp = np.max(np.abs(segment))
    threshold = threshold_ratio * max_amp
    for i in range(1, len(segment)):
        if np.abs(segment[i]) < threshold:
            return i / fs
    return window / fs

def detect_ueops(signal, fs, max_echoes=5, min_time_diff=0.2e-6, min_peak_amp=0.1,win_value=None):
    ueops = []
    start = 0
    for _ in range(max_echoes):
        if start >= len(signal) - 200:
            break
        odes = compute_odes(signal[start:],win_value)
        rough_idx = detect_rough_ueop(odes)
        if rough_idx is None:
            break
        rough_global = start + rough_idx
        ueop = aic_picker(signal, rough_global)

        ueops.append(ueop)
        # min_time_diff (seconds) -> convert to samples for jump
        jump_samples = int(min_time_diff * fs)
        if jump_samples < 1:
            jump_samples = 1
        start = ueop + jump_samples
    return ueops

# ---------------------
# utils: 读取标签/信号（与你之前一致）
# ---------------------
# def load_labels(label_dir):
#     labels = []
#     for i in range(1, 10):  # point1 ~ point9
#         label_path = os.path.join(label_dir, f"point{i}.txt")
#         with open(label_path, "r") as f:
#             val = float(f.readline().strip())
#         labels.append(val)
#     return np.array(labels)

# def load_labels(label_dir):
#     """加载真实值，同时返回点名"""
#     labels = []
#     point_names = []
#
#     file_list = sorted(
#         [f for f in os.listdir(label_dir) if f.startswith("point") and f.endswith(".txt")],
#         key=lambda x: int(''.join(filter(str.isdigit, x)))
#     )
#
#     for fname in file_list:
#         path = os.path.join(label_dir, fname)
#         with open(path, "r") as f:
#             val = float(f.readline().strip())
#         labels.append(val)
#         point_names.append(fname.replace(".txt",""))
#     return np.array(labels), point_names
#
# def load_signals(data_dir, signal_len=3000):
#     """加载信号，同时返回点名"""
#     signals = []
#     point_names = []
#
#     file_list = sorted(
#         [f for f in os.listdir(data_dir) if f.startswith("point") and f.endswith(".txt")],
#         key=lambda x: int(''.join(filter(str.isdigit, x)))
#     )
#
#     for fname in file_list:
#         path = os.path.join(data_dir, fname)
#         data = np.loadtxt(path)
#         data = np.asarray(data, dtype=np.float32).ravel()
#         L = len(data)
#         if L == signal_len:
#             sig = data
#         elif L > signal_len:
#             start = (L - signal_len) // 2
#             sig = data[start:start+signal_len]
#         else:
#             sig = np.zeros(signal_len, dtype=np.float32)
#             sig[:L] = data
#         m = np.max(np.abs(sig)) if sig.size > 0 else 0.0
#         if m > 0:
#             sig = sig / m
#         signals.append(sig)
#         point_names.append(fname.replace(".txt",""))
#
#     return np.array(signals), point_names





# ---------------------
# 主流程：对每个点做 ODES+AIC 检测，取第一个 ueop（如果没有检测到则置 NaN）
# ---------------------


# ---------------------
# utils: 读取标签/信号
# ---------------------
def load_labels(label_dir):
    """自动加载标签，适用于多值标签文件"""
    labels = []
    point_names = []

    # 获取所有.txt文件
    file_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    # 过滤出纯数字文件名的文件
    file_list = [f for f in file_list if f.replace(".txt", "").isdigit()]

    # 按数字大小排序
    file_list = sorted(file_list, key=lambda x: int(x.replace(".txt", "")))

    for fname in file_list:
        path = os.path.join(label_dir, fname)
        try:
            with open(path, "r") as f:
                line = f.readline().strip()

                # 处理多值标签
                if ',' in line:
                    # 取第一个值作为标签
                    label_val = float(line.split(',')[0])
                else:
                    label_val = float(line)

            labels.append(label_val)
            point_names.append(fname.replace(".txt", ""))
            # print(f"加载标签 {fname}: 使用值 {label_val}")

        except Exception as e:
            print(f"加载标签文件 {fname} 时出错: {str(e)}")
            continue

    return np.array(labels), point_names

def load_signals(data_dir, signal_len=3000):
    """自动加载信号，同时返回点名"""
    signals = []
    point_names = []

    # 获取所有.txt文件
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    # 过滤出纯数字文件名的文件
    file_list = [f for f in file_list if f.replace(".txt", "").isdigit()]

    # 按数字大小排序
    file_list = sorted(file_list, key=lambda x: int(x.replace(".txt", "")))

    for fname in file_list:
        path = os.path.join(data_dir, fname)
        try:
            data = np.loadtxt(path)
            data = np.asarray(data, dtype=np.float32).ravel()
            L = len(data)

            # 处理信号长度
            if L == signal_len:
                sig = data
            elif L > signal_len:
                start = (L - signal_len) // 2
                sig = data[start:start + signal_len]
            else:
                sig = np.zeros(signal_len, dtype=np.float32)
                sig[:L] = data

            # 归一化处理
            m = np.max(np.abs(sig)) if sig.size > 0 else 0.0
            if m > 0:
                sig = sig / m

            signals.append(sig)
            point_names.append(fname.replace(".txt", ""))

        except Exception as e:
            print(f"加载文件 {fname} 时出错: {str(e)}")
            continue

    return np.array(signals), point_names


def get_windos_value(input_value):
    """
    根据输入值判断所在区间并返回对应的值

    参数:
    input_value: 数值类型的输入值

    返回:
    对应的区间值
    """
    try:
        value = float(input_value)
    except (ValueError, TypeError):
        return "输入值必须为数字"

    # if value < 6:
    #     return 250
    # elif 6 <= value < 7:
    #     return 650
    # elif 7 <= value < 8:
    #     return 850
    # elif 8 <= value < 9:
    #     return 950
    # else:  # value >= 9
    #     return 1250
    if value < 5:
        return 250
    elif 5<= value< 6:
        return 450
    elif 6 <= value < 7:
        return 650
    elif 7 <= value < 8:
        return 850
    elif 8 <= value < 9:
        return 1050
    else:  # value >= 9
        return 1250


def main():
    data_dir = "data_test/classified/15DB/target_signal"
    label_dir = "data_test/classified/15DB/label"
    result_dir = "results_tradition_1"
    os.makedirs(result_dir, exist_ok=True)

    # ---- 请根据你实际采样频率修改 fs ----
    fs = 5e8  # 5e8 Hz as you set

    # 缩放因子（若不需要可设 apply_scale=False）
    scale_factor = 5900.0 / 2000.0
    apply_scale = False

    # 读取数据与标签
    # signals = load_signals(data_dir, signal_len=3000)
    # labels = load_labels(label_dir)  # 标签，请确认单位（这里按原来代码处理）
    # 读取信号与标签
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)
    labels_thinck = [x * scale_factor for x in labels]

    if point_names_signals != point_names_labels:
        print("警告：信号和标签的点名不一致，将按顺序匹配")

    n = len(signals)
    preds_us = np.full(n, np.nan, dtype=float)           # 预测值（微秒）
    preds_sample_first = np.full(n, np.nan, dtype=float) # 第1个ueop样点索引
    preds_sample_second = np.full(n, np.nan, dtype=float)# 第2个ueop样点索引

    for i, sig in enumerate(signals):
        #####对于3.983mm以及5.941mm用min_time_diff=0.85e-6，threshold_ratio=0.5,window=200##############
        #####对于8.058mm用min_time_diff=0.85e-6，threshold_ratio=0.5,window=936##############
        #####对于10.102mm用min_time_diff=0.85e-6，threshold_ratio=0.5,window=1250##############
        ueops = detect_ueops(sig, fs=fs, max_echoes=5, min_time_diff=0.85e-6, min_peak_amp=0,win_value=get_windos_value(labels_thinck[i]))
        if len(ueops) < 2:
            print(f"point{i+1}: detected {len(ueops)} ueop(s) -> insufficient for TOF diff")
            continue
        # 取第一个和第二个回波，计算差（微秒）
        idx1 = int(ueops[0])
        idx2 = int(ueops[1])
        predicted_value_us = (idx2 - idx1) / fs * 1e6  # μs
        preds_us[i] = predicted_value_us
        preds_sample_first[i] = idx1
        preds_sample_second[i] = idx2
        print(f"point{i+1}: ueop1={idx1}, ueop2={idx2}, TOF_diff={predicted_value_us:.6f} μs")

    # 组合预测与标签（labels 维持原读取值；如果 labels 单位不是 μs，请先转换到 μs）
    preds = preds_us.copy()
    labels_arr = labels.copy()

    # 有效点掩码（非 NaN）
    valid_mask = ~np.isnan(preds)
    if not np.any(valid_mask):
        raise RuntimeError("所有点都未检测到有效的两个回波，无法计算指标，请检查参数。")

    # 应用缩放（可选）
    if apply_scale:
        preds_scaled = preds * scale_factor
        labels_scaled = labels_arr * scale_factor
    else:
        preds_scaled = preds.copy()
        labels_scaled = labels_arr.copy()

    # 仅对有效点计算指标
    preds_scaled_masked = preds_scaled[valid_mask]
    labels_scaled_masked = labels_scaled[valid_mask]

    errors = preds_scaled_masked - labels_scaled_masked
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled_masked) + 1e-8))
    avg_val = np.mean(preds_scaled_masked)
    max_dev = np.max(preds_scaled_masked) - avg_val
    min_dev = np.min(preds_scaled_masked) - avg_val

    # 保存结果到 DataFrame（保留所有点）
    # df = pd.DataFrame({
    #     "Point": [f"point{i}" for i in range(1, n+1)],
    #     "真实值_orig": labels_arr,
    #     "预测_us_orig": preds,                    # μs（原始预测）
    #     "预测样点_ueop1": preds_sample_first,
    #     "预测样点_ueop2": preds_sample_second,
    #     "预测_scaled": preds_scaled,
    #     "真实_scaled": labels_scaled,
    #     "误差_scaled": (preds_scaled - labels_scaled)
    # })
    df = pd.DataFrame({
        "Point": point_names_signals,
        "真实值_orig": labels,
        "预测_us_orig": preds_us,
        "预测样点_ueop1": preds_sample_first,
        "预测样点_ueop2": preds_sample_second,
        "预测_scaled": preds_scaled,
        "真实_scaled": labels_scaled,
        "误差_scaled": preds_scaled - labels_scaled
    })
    # 追加误差指标（放在表尾）
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "平均值", "最大偏差", "最小偏差"],
        "真实值_orig": ["-"] * 6,
        "预测_us_orig": ["-"] * 6,
        "预测样点_ueop1": ["-"] * 6,
        "预测样点_ueop2": ["-"] * 6,
        "预测_scaled": ["-"] * 6,
        "真实_scaled": ["-"] * 6,
        "误差_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    # 保存 CSV（带容错）
    import datetime, stat
    save_path = os.path.join(result_dir, "odes_aic_results_1.csv")
    try:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print("结果已保存到:", save_path)
    except PermissionError as e:
        print("PermissionError when saving:", e)
        # 尝试改为可写后重试
        try:
            if os.path.exists(save_path):
                os.chmod(save_path, stat.S_IWRITE)
                df.to_csv(save_path, index=False, encoding="utf-8-sig")
                print("Saved after chmod:", save_path)
            else:
                raise
        except Exception:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_path = os.path.splitext(save_path)[0] + f"_{ts}.csv"
            df.to_csv(alt_path, index=False, encoding="utf-8-sig")
            print("Saved to alternative path:", alt_path)

    # 打印统计信息
    print(f"有效点数量: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, MARE={mare:.6e}")
    print(f"平均值={avg_val:.6e}, 最大偏差={max_dev:.6e}, 最小偏差={min_dev:.6e}")


if __name__ == "__main__":
    main()
