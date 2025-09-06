import os, re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from peak_method import detect_peak_ueops_mask  # Import your peak detection function

def load_labels(label_dir):
    """Automatically load labels and return point names"""
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
    """Automatically load signals and return point names"""
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



# ---------------------
# Main process
# ---------------------
def main():
    data_dir = "datasets/Real_data/10.102mm/target_orignal"
    label_dir = "datasets/Real_data/10.102mm/label"
    result_dir = "results_peak"
    os.makedirs(result_dir, exist_ok=True)

    # ---- Please modify fs according to your actual sampling frequency ----
    fs = 5e8  # 5e8 Hz

    # Scaling factor (set apply_scale=False if not needed)
    scale_factor = 5900.0 / 2000.0
    apply_scale = True

    # Load signals and labels
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)
    n = len(signals)
    preds_us = np.full(n, np.nan, dtype=float)

    for i, sig in enumerate(signals):
        print(f"\n>>> Processing point{i+1} ...")

        try:
            peaks = detect_peak_ueops_mask(
                sig, fs,  # 🚨 Use original signal directly here
                skip_time=2500e-9,#Adjustable parameter: 1200e-9 for 3.982, 5.941, 8.058; 2500e-9 for 10.102
                max_echoes=5
            )
        except Exception as e:
            print(f"point{i+1} detection failed: {e}")
            continue

        if len(peaks) < 2:
            print(f"point{i+1}: insufficient peaks (< 2), skipping")
            continue

        # Take time difference of first two peaks as TOF_diff
        delta_opt = abs((peaks[1] - peaks[0]) / fs)
        predicted_value_us = delta_opt * 1e6
        preds_us[i] = predicted_value_us

        print(f"point{i+1}: detected {len(peaks)} peaks, predicted TOF_diff={predicted_value_us:.6f} μs")

    # Valid point mask
    labels_arr = labels.copy()
    valid_mask = ~np.isnan(preds_us)
    if not np.any(valid_mask):
        raise RuntimeError("All points failed to detect valid results, please check parameters.")

    # Scaling
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
        "Point": point_names_signals,
        "True_Value_orig": labels,
        "Predicted_us_orig": preds_us,
        "Predicted_scaled": preds_scaled,
        "True_scaled": labels_scaled,
        "Error_scaled": preds_scaled - labels_scaled
    })
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "Average", "Max_Deviation", "Min_Deviation"],
        "True_Value_orig": ["-"] * 6,
        "Predicted_us_orig": ["-"] * 6,
        "Predicted_scaled": ["-"] * 6,
        "True_scaled": ["-"] * 6,
        "Error_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    save_path = os.path.join(result_dir, "3.983mm_peak_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Results saved to:", save_path)

    print(f"Valid point count: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, MARE={mare:.6e}")
    print(f"Average={avg_val:.6e}, Max deviation={max_dev:.6e}, Min deviation={min_dev:.6e}")


if __name__ == "__main__":
    main()
