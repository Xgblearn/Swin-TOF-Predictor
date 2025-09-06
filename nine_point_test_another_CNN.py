import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from another_CNN import PaperCNN_Ultrasonic  # Import CNN model definition


def load_labels(label_dir):
    """Load true values and return point names"""
    labels = []
    point_names = []

    file_list = sorted([f for f in os.listdir(label_dir) if f.startswith("point") and f.endswith(".txt")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

    for fname in file_list:
        label_path = os.path.join(label_dir, fname)
        with open(label_path, "r") as f:
            val = float(f.readline().strip())
        labels.append(val)
        point_names.append(fname.replace(".txt",""))
    return np.array(labels), point_names

def load_signals(data_dir, signal_len=3000):
    """Load time series signals and return corresponding point names"""
    signals = []
    point_names = []

    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith("point") and f.endswith(".txt")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

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
            sig = sig / m
        signals.append(sig)
        point_names.append(fname.replace(".txt",""))

    return np.array(signals), point_names
# def load_signals(data_dir, signal_len=3000):
#     """加载9个点的时序信号（和训练保持一致：补零/截断 + 归一化）"""
#     signals = []
#     for i in range(1, 10):
#         file_path = os.path.join(data_dir, f"point{i}.txt")
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
#             sig = sig / m  # 归一化
#
#         signals.append(sig)
#     return np.array(signals)
def main_fixed():
    model_path = "saved_model/post_another_best_cnn.pth"
    data_dir = "datasets/Real_data/10.102mm/target_orignal"
    label_dir = "datasets/Real_data/10.102mm/label"
    result_dir = "results_another_CNN"

    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperCNN_Ultrasonic(input_channels=1, signal_len=3000, out_dim=1)

    # Load checkpoint (note scaler)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # If old format, directly use state_dict
            model.load_state_dict(checkpoint)
        if "scaler" in checkpoint:
            scaler = checkpoint["scaler"]
            print("Loaded scaler from checkpoint.")
        else:
            scaler = None
            print("Warning: no 'scaler' found in checkpoint.")
    else:
        model.load_state_dict(checkpoint)
        scaler = None
        print("Checkpoint was plain state_dict (no scaler).")

    model.to(device)
    model.eval()

    signals, point_names = load_signals(data_dir, signal_len=3000)
    labels, _ = load_labels(label_dir)  # labels order same as point_names

    preds = []
    # with torch.no_grad():
    #     for sig in signals:
    #         sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    #         out = model(sig_tensor)  # 这是模型输出（训练时是对 scaled label 的回归）
    #         if isinstance(out, torch.Tensor):
    #             val = float(out.cpu().numpy().ravel()[0])
    #         else:
    #             val = float(out)
    #         preds.append(val)
    # preds = np.array(preds)
    with torch.no_grad():
        for sig in signals:
            sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = model(sig_tensor)
            val = float(out.cpu().numpy().ravel()[0])
            preds.append(val)
    preds = np.array(preds)
    # If scaler exists: inverse_transform model outputs back to original units
    if scaler is not None:
        try:
            preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).ravel()
        except Exception as e:
            print("scaler.inverse_transform error:", e)
            preds_orig = preds  # Fallback handling
    else:
        # If no scaler, **don't** randomly use 5900/2000 unless you're sure training used this scaling
        print("No scaler -> keeping raw model outputs (may be inconsistent with training scale)")
        preds_orig = preds

        # 计算误差（用原始标签 labels）
        # errors = preds_orig - labels
        # mae = np.mean(np.abs(errors))
        # mse = np.mean(errors ** 2)
        # mare = np.mean(np.abs(errors) / (np.abs(labels) + 1e-8))

        # # 保存 CSV
        # df = pd.DataFrame({
        #     "Point": [f"point{i}" for i in range(1, 10)],
        #     "真实值": labels,
        #     "预测值": preds_orig,
        #     "误差": errors
        # })
        # metrics = {
        #     "Point": ["MAE", "MSE", "MARE"],
        #     "真实值": ["-"] * 3,
        #     "预测值": ["-"] * 3,
        #     "误差": [mae, mse, mare]
        # }
        # df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)
        # save_path = os.path.join(result_dir, "test_results_fixed.csv")
        # df.to_csv(save_path, index=False, encoding="utf-8-sig")
        # print("结果已保存到:", save_path)
        # print(df)

    # Whether to apply additional hardcoded scaling (5900/2000)
    apply_extra_scale = True  # Set to False if no additional scaling needed

    scale_factor = 5900.0 / 2000.0

    if apply_extra_scale:
        preds_scaled = preds_orig * scale_factor
        labels_scaled = labels * scale_factor
    else:
        preds_scaled = preds_orig.copy()
        labels_scaled = labels.copy()

    # Error metrics (based on scaled results)
    errors = preds_scaled - labels_scaled
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled) + 1e-8))  # Prevent division by zero

    # Additional statistics
    avg_val = np.mean(preds_scaled)
    max_dev = np.max(preds_scaled) - avg_val
    min_dev = np.min(preds_scaled) - avg_val

    # Save CSV: keep both original units (preds_orig, labels) and scaled units (preds_scaled, labels_scaled)
    df = pd.DataFrame({
        "Point": point_names,
        "True_Value_orig": labels,
        "Predicted_Value_orig": preds_orig,
        "True_Value_scaled": labels_scaled,
        "Predicted_Value_scaled": preds_scaled,
        "Error_scaled": errors
    })
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "Average", "Max_Deviation", "Min_Deviation"],
        "True_Value_orig": ["-"] * 6,
        "Predicted_Value_orig": ["-"] * 6,
        "True_Value_scaled": ["-"] * 6,
        "Predicted_Value_scaled": ["-"] * 6,
        "Error_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "3.983mm_another_CNN_test_results_fixed.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # Print main metrics
    print("Results saved to:", save_path)
    print(f"MAE={mae:.6f}, MSE={mse:.6f}, MARE={mare:.6f}")
    print(f"Average={avg_val:.6f}, Max deviation={max_dev:.6f}, Min deviation={min_dev:.6f}")
    print(df)
    # ===== 这里结束替换 =====


if __name__ == "__main__":
    main_fixed()
