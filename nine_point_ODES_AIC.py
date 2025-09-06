# run_odes_aic_eval.py
import os
import numpy as np
import pandas as pd

# ---------------------
# Paste your functions here (no logic changes)
# ---------------------
def compute_odes(signal, window=1250):
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

def detect_ueops(signal, fs, max_echoes=5, min_time_diff=0.2e-6, min_peak_amp=0.1):
    ueops = []
    start = 0
    for _ in range(max_echoes):
        if start >= len(signal) - 200:
            break
        odes = compute_odes(signal[start:])
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
# utils: Read labels/signals (consistent with your previous code)
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
    """Load signals and return point names"""
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
# Main process: ODES+AIC detection for each point, take first ueop (set to NaN if not detected)
# ---------------------
def main():
    data_dir = "datasets/Real_data/10.102mm/target_orignal"
    label_dir = "datasets/Real_data/10.102mm/label"
    result_dir = "results_ODES_AIC"
    os.makedirs(result_dir, exist_ok=True)

    # ---- Please modify fs according to your actual sampling frequency ----
    fs = 5e8  # 5e8 Hz as you set

    # Scaling factor (set apply_scale=False if not needed)
    scale_factor = 5900.0 / 2000.0
    apply_scale = True

    # Read data and labels
    signals, point_names_signals = load_signals(data_dir, signal_len=3000)
    labels, point_names_labels = load_labels(label_dir)

    n = len(signals)
    preds_us = np.full(n, np.nan, dtype=float)           # Predictions (microseconds)
    preds_sample_first = np.full(n, np.nan, dtype=float) # First ueop sample index
    preds_sample_second = np.full(n, np.nan, dtype=float)# Second ueop sample index

    for i, sig in enumerate(signals):
        #####For 3.983mm and 5.941mm use min_time_diff=0.85e-6, threshold_ratio=0.5, window=200##############
        #####For 8.058mm use min_time_diff=0.85e-6, threshold_ratio=0.5, window=936##############
        #####For 10.102mm use min_time_diff=0.85e-6, threshold_ratio=0.5, window=1250##############
        ueops = detect_ueops(sig, fs=fs, max_echoes=5, min_time_diff=0.85e-6, min_peak_amp=0.1)
        if len(ueops) < 2:
            print(f"point{i+1}: detected {len(ueops)} ueop(s) -> insufficient for TOF diff")
            continue
        # Take first and second echoes, calculate difference (microseconds)
        idx1 = int(ueops[0])
        idx2 = int(ueops[1])
        predicted_value_us = (idx2 - idx1) / fs * 1e6  # μs
        preds_us[i] = predicted_value_us
        preds_sample_first[i] = idx1
        preds_sample_second[i] = idx2
        print(f"point{i+1}: ueop1={idx1}, ueop2={idx2}, TOF_diff={predicted_value_us:.6f} μs")

    # Combine predictions with labels (labels maintain original read values; if labels unit is not μs, convert to μs first)
    preds = preds_us.copy()
    labels_arr = labels.copy()

    # Valid point mask (non-NaN)
    valid_mask = ~np.isnan(preds)
    if not np.any(valid_mask):
        raise RuntimeError("All points failed to detect valid two echoes, cannot calculate metrics, please check parameters.")

    # Apply scaling (optional)
    if apply_scale:
        preds_scaled = preds * scale_factor
        labels_scaled = labels_arr * scale_factor
    else:
        preds_scaled = preds.copy()
        labels_scaled = labels_arr.copy()

    # Calculate metrics only for valid points
    preds_scaled_masked = preds_scaled[valid_mask]
    labels_scaled_masked = labels_scaled[valid_mask]

    errors = preds_scaled_masked - labels_scaled_masked
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled_masked) + 1e-8))
    avg_val = np.mean(preds_scaled_masked)
    max_dev = np.max(preds_scaled_masked) - avg_val
    min_dev = np.min(preds_scaled_masked) - avg_val

    # Save results to DataFrame (keep all points)
    df = pd.DataFrame({
        "Point": point_names_signals,
        "True_Value_orig": labels,
        "Predicted_us_orig": preds_us,
        "Predicted_Sample_ueop1": preds_sample_first,
        "Predicted_Sample_ueop2": preds_sample_second,
        "Predicted_scaled": preds_scaled,
        "True_scaled": labels_scaled,
        "Error_scaled": preds_scaled - labels_scaled
    })
    # Append error metrics (at end of table)
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "Average", "Max_Deviation", "Min_Deviation"],
        "True_Value_orig": ["-"] * 6,
        "Predicted_us_orig": ["-"] * 6,
        "Predicted_Sample_ueop1": ["-"] * 6,
        "Predicted_Sample_ueop2": ["-"] * 6,
        "Predicted_scaled": ["-"] * 6,
        "True_scaled": ["-"] * 6,
        "Error_scaled": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    # Save CSV (with error handling)
    import datetime, stat
    save_path = os.path.join(result_dir, "3.983mm_odes_aic_results.csv")
    try:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print("Results saved to:", save_path)
    except PermissionError as e:
        print("PermissionError when saving:", e)
        # Try to make writable and retry
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

    # Print statistics
    print(f"Valid point count: {np.sum(valid_mask)} / {n}")
    print(f"MAE={mae:.6e}, MSE={mse:.6e}, MARE={mare:.6e}")
    print(f"Average={avg_val:.6e}, Max deviation={max_dev:.6e}, Min deviation={min_dev:.6e}")


if __name__ == "__main__":
    main()
