# run_inference.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from CNN import Paper1DCNN


# ----------------------------
# 数据加载函数（与你提供的保持一致）
# ----------------------------
#################################之前的1到9个点是用这个函数，是必须按照顺序的#################
# def load_labels(label_dir):
#     """加载真实值 point1..point9"""
#     labels = []
#     for i in range(1, 10):  # point1 ~ point9
#         label_path = os.path.join(label_dir, f"point{i}.txt")
#         with open(label_path, "r") as f:
#             val = float(f.readline().strip())
#         labels.append(val)
#     return np.array(labels)

def load_labels(label_dir):
    """加载真实值，同时返回点名"""
    labels = []
    point_names = []

    file_list = sorted([f for f in os.listdir(label_dir) if f.startswith("point") and f.endswith(".txt")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

    for fname in file_list:
        label_path = os.path.join(label_dir, fname)
        with open(label_path, "r") as f:
            val = float(f.readline().strip())
        labels.append(val)
        point_names.append(fname.replace(".txt", ""))  # 去掉后缀
    return np.array(labels), point_names




################之前的1到9个点是用这个函数，是必须按照顺序的#################
# def load_signals(data_dir, signal_len=3000):
#     """加载9个点的时序信号（补零/截断 + 单条归一化）"""
#     signals = []
#     for i in range(1, 10):
#         file_path = os.path.join(data_dir, f"point{i}.txt")
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"信号文件不存在: {file_path}")
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


def load_signals(data_dir, signal_len=3000):
    """加载时序信号（自动处理缺失文件，补零/截断 + 单条归一化）"""
    signals = []

    # 获取目录下所有 point*.txt 文件
    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("point") and f.endswith(".txt")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))  # 按数字排序

    for fname in file_list:
        file_path = os.path.join(data_dir, fname)
        data = np.loadtxt(file_path)
        data = np.asarray(data, dtype=np.float32).ravel()
        L = len(data)
        if L == signal_len:
            sig = data
        elif L > signal_len:
            start = (L - signal_len) // 2
            sig = data[start:start + signal_len]
        else:
            sig = np.zeros(signal_len, dtype=np.float32)
            sig[:L] = data

        m = np.max(np.abs(sig)) if sig.size > 0 else 0.0
        if m > 0:
            sig = sig / m  # 单条归一化

        signals.append(sig)

    return np.array(signals)






# ----------------------------
# 推理主函数
# ----------------------------
def main():
    # --- 用户可修改的路径 ---
    model_path = "saved_model/post_best_cnn.pth"  # 你训练后保存的位置
    data_dir = "datasets/Real_data/10.102mm/target_orignal"
    label_dir = "datasets/Real_data/10.102mm/label"
    result_dir = "results_CNN"

    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建模型（参数按训练时保持一致）
    model = Paper1DCNN(input_channels=1, signal_len=3000, out_dim=1)

    # 加载 checkpoint（兼容多种保存格式）
    checkpoint = torch.load(model_path, map_location=device)
    print("Loaded checkpoint type:", type(checkpoint))
    scaler = None
    if isinstance(checkpoint, dict):
        print("checkpoint keys:", list(checkpoint.keys()))
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # 直接保存了 state_dict
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                # 有时 checkpoint 为 {'state_dict': ...}
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    raise e
        if "scaler" in checkpoint:
            scaler = checkpoint["scaler"]
            print("Loaded scaler from checkpoint.")
    else:
        # 纯 state_dict
        model.load_state_dict(checkpoint)
        print("Checkpoint was plain state_dict (no scaler).")

    model.to(device)
    model.eval()

    # 读取数据
    signals = load_signals(data_dir, signal_len=3000)
    # labels = load_labels(label_dir)  # 原始真实值（物理单位）   ####原来的
    labels, point_names = load_labels(label_dir)
    # 推理
    preds = []
    with torch.no_grad():
        for sig in signals:
            sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
            out = model(sig_tensor)
            # out 可能为 Tensor 或标量
            if isinstance(out, torch.Tensor):
                val = float(out.cpu().numpy().ravel()[0])
            else:
                val = float(out)
            preds.append(val)
    preds = np.array(preds)
    print("raw model outputs stats: min/max/mean:", preds.min(), preds.max(), preds.mean())

    # 如果有 scaler：把模型输出 inverse_transform 回原始单位
    if scaler is not None:
        try:
            preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            print("Applied scaler.inverse_transform to model outputs.")
        except Exception as e:
            print("scaler.inverse_transform 出错:", e)
            preds_orig = preds.copy()
    else:
        print("No scaler found in checkpoint. Keeping raw model outputs as preds_orig.")
        preds_orig = preds.copy()

    # 决定是否应用额外硬编码缩放（5900/2000）
    # 推荐策略：如果 checkpoint 含 scaler 则通常不需要额外缩放；如果没有 scaler 且你训练时确实用过 5900/2000，则开启它
    # 是否应用额外的硬编码缩放（5900/2000）
    apply_extra_scale = True  # 如果不需要额外缩放，设为 False

    scale_factor = 5900.0 / 2000.0

    if apply_extra_scale:
        preds_scaled = preds_orig * scale_factor
        labels_scaled = labels * scale_factor
    else:
        preds_scaled = preds_orig.copy()
        labels_scaled = labels.copy()

    # 误差指标（基于 scaled 结果）
    errors = preds_scaled - labels_scaled
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled) + 1e-8))  # 防止除零

    # 额外统计量
    avg_val = np.mean(preds_scaled)
    max_dev = np.max(preds_scaled) - avg_val
    min_dev = np.min(preds_scaled) - avg_val

    # DataFrame
    df = pd.DataFrame({
        "Point": point_names,  # 使用自动生成的文件名
        "真实值_orig": labels,
        "预测值_orig": preds_orig,
        "真实值_scaled": labels_scaled,
        "预测值_scaled": preds_scaled,
        "误差_scaled": errors
    })

    ########原来的
    # 保存 CSV：同时保留原始单位（preds_orig, labels）和 scaled 单位（preds_scaled, labels_scaled）
    # df = pd.DataFrame({
    #     "Point": [f"point{i}" for i in range(1, 10)],
    #     "真实值_orig": labels,
    #     "预测值_orig": preds_orig,
    #     "真实值_scaled": labels_scaled,
    #     "预测值_scaled": preds_scaled,
    #     "误差_scaled": errors
    # })



    metrics = {
        "Point": ["MAE", "MSE", "MARE", "平均值", "最大偏差", "最小偏差"],
        "真实值_orig": ["-"] * 6,
        "预测值_orig": ["-"] * 6,
        "真实值_scaled": ["-"] * 6,
        "预测值_scaled": ["-"] * 6,
        "误差_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "3.983mm_CNN_test_results_fixed.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 同时打印主要指标
    print("结果已保存到:", save_path)
    print(f"MAE={mae:.6f}, MSE={mse:.6f}, MARE={mare:.6f}")
    print(f"平均值={avg_val:.6f}, 最大偏差={max_dev:.6f}, 最小偏差={min_dev:.6f}")
    print(df)
    # ===== 这里结束替换 =====


if __name__ == "__main__":
    main()
