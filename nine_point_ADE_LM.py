import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ADE_and_LM import estimate_echo_count_and_params  # Import your ADE+LM module
from scipy.signal import hilbert

# ---------------------
# utils: Read labels/signals
# ---------------------
# def load_labels(label_dir):
#     labels = []
#     for i in range(1, 10):  # point1 ~ point9
#         label_path = os.path.join(label_dir, f"point{i}.txt")
#         with open(label_path, "r") as f:
#             val = float(f.readline().strip())
#         labels.append(val)
#     return np.array(labels)

def load_labels(label_dir):
    """Load true values and return point names"""
    labels = []
    point_names = []

    file_list = sorted(
        [f for f in os.listdir(label_dir) if f.startswith("point") and f.endswith(".txt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    for fname in file_list:
        path = os.path.join(label_dir, fname)
        with open(path, "r") as f:
            val = float(f.readline().strip())
        labels.append(val)
        point_names.append(fname.replace(".txt",""))
    return np.array(labels), point_names

def load_signals(data_dir, signal_len=3000):
    """Load time series signals and return point names"""
    signals = []
    point_names = []

    file_list = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("point") and f.endswith(".txt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    for fname in file_list:
        path = os.path.join(data_dir, fname)
        data = np.loadtxt(path)
        data = np.asarray(data, dtype=np.float32).ravel()
        L = len(data)
        if L == signal_len:
            sig = data
        elif L > signal_len:
            start = (L - signal_len) // 2
            sig = data[start:start+signal_len]
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
# Main process
# ---------------------
def main():
    data_dir = "datasets/Real_data/10.102mm/target_orignal"
    label_dir = "datasets/Real_data/10.102mm/label"
    result_dir = "results_ADE_LM"
    os.makedirs(result_dir, exist_ok=True)

    # ---- Please modify fs according to your actual sampling frequency ----
    fs = 5e8  # 5e8 Hz

    # Scaling factor (set apply_scale=False if not needed)
    scale_factor = 5900.0 / 2000.0
    apply_scale = True

    # Read data and labels
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)
    n = len(signals)
    preds_us = np.full(n, np.nan, dtype=float)  # Predictions (microseconds)

    for i, sig in enumerate(signals):
        print(f"\n>>> Processing point{i+1} ...")
        X = np.arange(len(sig)) / fs
        envelope_true = np.abs(hilbert(sig))

        try:
            result = estimate_echo_count_and_params(
                X, envelope_true, echo_range=(2, 6),
                ade_kwargs={'pop_size': 30, 'max_gen': 150}
            )
        except Exception as e:
            print(f"point{i+1} fitting failed: {e}")
            continue

        N_opt = result['best_N']
        params_opt = result['best_params']
        delta_opt = params_opt[N_opt + 1]  # Parameter corresponding to Δ

        predicted_value_us = delta_opt * 1e6
        preds_us[i] = predicted_value_us
        print(f"point{i+1}: N={N_opt}, predicted TOF_diff={predicted_value_us:.6f} μs")

    # Valid point mask (non-NaN)
    labels_arr = labels.copy()
    valid_mask = ~np.isnan(preds_us)
    if not np.any(valid_mask):
        raise RuntimeError("All points failed to detect valid results, please check parameters.")

    # Apply scaling (optional)
    if apply_scale:
        preds_scaled = preds_us * scale_factor
        labels_scaled = labels_arr * scale_factor
    else:
        preds_scaled = preds_us.copy()
        labels_scaled = labels_arr.copy()

    # Calculate metrics only for valid points
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

    # Save results to DataFrame (keep all points)
    df = pd.DataFrame({
        "Point": point_names_signals,
        "True_Value_orig": labels,
        "Predicted_us_orig": preds_us,
        "Predicted_scaled": preds_scaled,
        "True_scaled": labels_scaled,
        "Error_scaled": preds_scaled - labels_scaled
    })
    # Append error metrics (at end of table)
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "Average", "Max_Deviation", "Min_Deviation"],
        "True_Value_orig": ["-"] * 6,
        "Predicted_us_orig": ["-"] * 6,
        "Predicted_scaled": ["-"] * 6,
        "True_scaled": ["-"] * 6,
        "Error_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "3.983mm_ADE_LM_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Results saved to:", save_path)

    # Print statistics
    print(f"Valid point count: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, R²={r2:.6f}, MARE={mare:.6e}")
    print(f"Average={avg_val:.6e}, Max deviation={max_dev:.6e}, Min deviation={min_dev:.6e}")


if __name__ == "__main__":
    main()
