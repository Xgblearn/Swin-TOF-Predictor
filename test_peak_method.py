import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from peak_method import detect_peak_ueops_mask


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
            print(f"加载标签 {fname}: 使用值 {label_val}")

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


# ---------------------
# 主流程
# ---------------------
def main():
    data_dir = "data_test/classified/15DB/target_signal"
    label_dir = "data_test/classified/15DB/label"
    result_dir = "results_condiction"
    os.makedirs(result_dir, exist_ok=True)

    # 创建调试目录
    debug_dir = os.path.join(result_dir, "debug_plots")
    os.makedirs(debug_dir, exist_ok=True)

    # ---- 请根据你实际采样频率修改 fs ----
    fs = 5e8  # 5e8 Hz

    # 缩放因子（若不需要可设 apply_scale=False）
    scale_factor = 5900.0 / 2000.0
    apply_scale = False

    # 加载信号与标签
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)

    # 确保信号和标签的点名一致
    if point_names_signals != point_names_labels:
        print("警告：信号和标签的点名不一致，将按顺序匹配")

    n = min(len(signals), len(labels))  # 取最小长度避免索引错误
    preds_us = np.full(n, np.nan, dtype=float)

    for i in range(n):
        point_name = point_names_signals[i]
        sig = signals[i]
        print(f"\n>>> 处理 {point_name}")

        try:
            # 信号预处理 - 标准化
            sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            # 峰值检测（移除不支持的参数）
            peaks = detect_peak_ueops_mask(
                sig_norm, fs,
                skip_time=1200e-9,  # 缩短跳过时间
                max_echoes=5,
            )

            # 可视化检测结果
            # plt.figure(figsize=(12, 4))
            # plt.plot(sig_norm, label='Signal')
            # if len(peaks) > 0:
            #     plt.scatter(peaks, sig_norm[peaks], c='r', label='Detected Peaks')
            # plt.title(f"{point_name} - {len(peaks)} peaks detected")
            # plt.legend()
            # plt.savefig(os.path.join(debug_dir, f"{point_name}_peaks.png"))
            # plt.close()

            if len(peaks) < 2:
                print(f"警告：只检测到 {len(peaks)} 个峰，至少需要2个")
                continue

            # 计算时间差
            delta_opt = abs((peaks[1] - peaks[0]) / fs)
            predicted_value_us = delta_opt * 1e6
            preds_us[i] = predicted_value_us
            print(f"预测 TOF_diff = {predicted_value_us:.6f} μs")

        except Exception as e:
            print(f"处理失败: {str(e)}")
            # 保存原始信号用于调试
            # plt.figure()
            # plt.plot(sig)
            # plt.title(f"{point_name} - Raw Signal")
            # plt.savefig(os.path.join(debug_dir, f"{point_name}_raw.png"))
            # plt.close()

    # 有效点掩码
    labels_arr = labels[:n].copy()  # 确保长度一致
    valid_mask = ~np.isnan(preds_us)

    if not np.any(valid_mask):
        print("所有点都未检测到有效结果，请检查参数。")

        # 保存部分调试信息
        debug_df = pd.DataFrame({
            "Point": point_names_signals[:n],
            "Signal_Length": [len(s) for s in signals[:n]],
            "Signal_Max": [np.max(s) for s in signals[:n]],
            "Signal_Mean": [np.mean(s) for s in signals[:n]]
        })
        debug_df.to_csv(os.path.join(result_dir, "peak_debug_info.csv"), index=False)

        return


    # 缩放
    if apply_scale:
        preds_scaled = preds_us * scale_factor
        labels_scaled = labels_arr * scale_factor
    else:
        preds_scaled = preds_us.copy()
        labels_scaled = labels_arr.copy()

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

    df = pd.DataFrame({
        "Point": point_names_signals[:n],
        "真实值_orig": labels_arr,
        "预测_us_orig": preds_us,
        "预测_scaled": preds_scaled,
        "真实_scaled": labels_scaled,
        "误差_scaled": preds_scaled - labels_scaled
    })

    metrics = {
        "Point": ["MAE", "MSE", "R2", "MARE", "平均值", "最大偏差", "最小偏差"],
        "真实值_orig": ["-"] * 7,
        "预测_us_orig": ["-"] * 7,
        "预测_scaled": ["-"] * 7,
        "真实_scaled": ["-"] * 7,
        "误差_scaled": [mae, mse, r2, mare, avg_val, max_dev, min_dev]
    }

    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "peak_metrics_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("结果已保存到:", save_path)

    print(f"有效点数量: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, R²={r2:.6f}, MARE={mare:.6e}")
    print(f"平均值={avg_val:.6e}, 最大偏差={max_dev:.6e}, 最小偏差={min_dev:.6e}")


if __name__ == "__main__":
    main()