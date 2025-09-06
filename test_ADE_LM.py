import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ADE_and_LM import estimate_echo_count_and_params  # 调用你写的ADE+LM模块
from scipy.signal import hilbert

# ---------------------
# utils: 读取标签/信号
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
# def load_signals(data_dir, signal_len=3000):
#     """加载时序信号，同时返回点名"""
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


# def load_signals(data_dir, signal_len=3000):
#     signals = []
#     for i in range(1, 10):
#         file_path = os.path.join(data_dir, f"point{i}.txt")
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"文件缺失: {file_path}")
#         data = np.loadtxt(file_path)
#
#         data = np.asarray(data, dtype=np.float32).ravel()
#         L = len(data)
#         if L == signal_len:
#             sig = data
#         elif L > signal_len:
#             start = (L - signal_len) // 2
#             sig = data[start:start + signal_len]
#         else:
#             sig = np.zeros(signal_len, dtype=np.float32)
#             sig[:L] = data
#
#         m = np.max(np.abs(sig)) if sig.size > 0 else 0.0
#         if m > 0:
#             sig = sig / m  # 单条归一化
#
#         signals.append(sig)
#     return np.array(signals)

# ---------------------
# 主流程
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


def main():
    data_dir = "data_test/classified/15DB/target_signal"
    label_dir = "data_test/classified/15DB/label"
    result_dir = "results_condiction_1"
    os.makedirs(result_dir, exist_ok=True)

    # ---- 请根据你实际采样频率修改 fs ----
    fs = 5e8  # 5e8 Hz

    # 缩放因子（若不需要可设 apply_scale=False）
    scale_factor = 5900.0 / 2000.0
    apply_scale = False

    # 读取数据与标签
    # signals = load_signals(data_dir, signal_len=3000)
    # labels = load_labels(label_dir)  # 标签，请确认单位（这里按原来代码处理）
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)
    if point_names_signals != point_names_labels:
        print("警告：信号和标签的点名不一致，将按顺序匹配")

    n = len(signals)
    preds_us = np.full(n, np.nan, dtype=float)  # 预测值（微秒）

    for i, sig in enumerate(signals):
        print(f"\n>>> 正在处理File_{point_names_labels[i]} ...")
        X = np.arange(len(sig)) / fs
        envelope_true = np.abs(hilbert(sig))

        try:
            result = estimate_echo_count_and_params(
                X, envelope_true, echo_range=(2, 6),
                ade_kwargs={'pop_size': 20, 'max_gen': 100}
            )
        except Exception as e:
            print(f"File_{point_names_labels[i]} 拟合失败: {e}")
            continue

        N_opt = result['best_N']
        params_opt = result['best_params']
        delta_opt = params_opt[N_opt + 1]  # Δ 对应的参数

        predicted_value_us = delta_opt * 1e6
        preds_us[i] = predicted_value_us
        print(f"File_{point_names_labels[i]}: N={N_opt}, 预测 TOF_diff={predicted_value_us:.6f} μs")

    # 有效点掩码（非 NaN）
    labels_arr = labels.copy()
    valid_mask = ~np.isnan(preds_us)
    if not np.any(valid_mask):
        raise RuntimeError("所有点都未检测到有效结果，请检查参数。")

    # 应用缩放（可选）
    if apply_scale:
        preds_scaled = preds_us * scale_factor
        labels_scaled = labels_arr * scale_factor
    else:
        preds_scaled = preds_us.copy()
        labels_scaled = labels_arr.copy()

    # 仅对有效点计算指标
    preds_scaled_masked = preds_scaled[valid_mask]
    labels_scaled_masked = labels_scaled[valid_mask]

    errors = preds_scaled_masked - labels_scaled_masked
    mae = mean_absolute_error(labels_scaled_masked, preds_scaled_masked)
    mse = mean_squared_error(labels_scaled_masked, preds_scaled_masked)
    r2 = r2_score(labels_scaled_masked, preds_scaled_masked)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled_masked) + 1e-8))
    avg_val = np.mean(preds_scaled_masked)
    max_dev = np.max(preds_scaled_masked) - avg_val
    min_dev = np.min(preds_scaled_masked) - avg_val

    # 保存结果到 DataFrame（保留所有点）
    # df = pd.DataFrame({
    #     "Point": [f"point{i}" for i in range(1, n+1)],
    #     "真实值_orig": labels_arr,
    #     "预测_us_orig": preds_us,
    #     "预测_scaled": preds_scaled,
    #     "真实_scaled": labels_scaled,
    #     "误差_scaled": preds_scaled - labels_scaled
    # })

    df = pd.DataFrame({
        "Point": point_names_signals,
        "真实值_orig": labels,
        "预测_us_orig": preds_us,
        "预测_scaled": preds_scaled,
        "真实_scaled": labels_scaled,
        "误差_scaled": preds_scaled - labels_scaled
    })
    # 追加误差指标（放在表尾）
    metrics = {
        "Point": ["MAE", "MSE", "R2", "MARE", "平均值", "最大偏差", "最小偏差"],
        "真实值_orig": ["-"] * 7,
        "预测_us_orig": ["-"] * 7,
        "预测_scaled": ["-"] * 7,
        "真实_scaled": ["-"] * 7,
        "误差_scaled": [mae, mse, r2, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "3.983mm_ADE_LM_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("结果已保存到:", save_path)

    # 打印统计信息
    print(f"有效点数量: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, R²={r2:.6f}, MARE={mare:.6e}")
    print(f"平均值={avg_val:.6e}, 最大偏差={max_dev:.6e}, 最小偏差={min_dev:.6e}")


if __name__ == "__main__":
    main()
