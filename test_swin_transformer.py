import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time  # 添加时间模块用于延迟测量
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from swin_transformer_model import my_swin_tiny_patch4_window7_224
from datetime import datetime

# Use a standard Latin font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImageTOFDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.db_name = os.path.basename(os.path.dirname(img_dir))  # extract DB name

        def extract_idx(fname):
            m = re.search(r'GASF_Image_(\d+)', fname)
            return int(m.group(1)) if m else -1

        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))],
            key=extract_idx
        )
        self.label_files = sorted(
            os.listdir(label_dir),
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        df = pd.read_csv(label_path, header=None)
        true_tof = float(df.iloc[0, 0])

        return img, torch.tensor([true_tof], dtype=torch.float32)


def test_one_db(img_dir, label_dir, base_result_dir):
    # Extract DB name from path
    db_name = os.path.basename(os.path.dirname(img_dir))

    # Create result directory with DB name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_result_dir, f"{db_name}_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageTOFDataset(img_dir, label_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = my_swin_tiny_patch4_window7_224(num_classes=1)
    # model_path = 'saved_model/swin_best_model_v10001_20250826_002155.pth'
    model_path = 'saved_model/swin_best_model_v10001.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ==== Batch inference & metrics ====
    preds, trues = [], []
    with torch.no_grad():
        for imgs, tofs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze(1)
            preds.extend(outputs.cpu().tolist())
            trues.extend(tofs.squeeze(1).tolist())

    preds_arr = np.array(preds)
    trues_arr = np.array(trues)

    # Compute metrics
    mae = mean_absolute_error(trues_arr, preds_arr)
    mse = mean_squared_error(trues_arr, preds_arr)
    r2 = r2_score(trues_arr, preds_arr)

    mask = trues_arr != 0
    rel_errors = np.abs((preds_arr[mask] - trues_arr[mask]) / trues_arr[mask])
    mare = np.mean(rel_errors)
    max_rel = np.max(rel_errors)

    errors = preds_arr - trues_arr
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)

    # ==== Single-sample latency test ====
    latency_times = []
    single_sample_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Warm up GPU
    warmup_img = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(warmup_img)

    # Measure latency
    with torch.no_grad():
        for imgs, _ in single_sample_loader:
            imgs = imgs.to(device)

            start_time = time.perf_counter()  # high precision timer
            _ = model(imgs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # wait for CUDA ops
            latency = time.perf_counter() - start_time

            latency_times.append(latency * 1000)  # to ms

    latency_arr = np.array(latency_times)
    latency_metrics = {
        "avg_latency": np.mean(latency_arr),
        "min_latency": np.min(latency_arr),
        "max_latency": np.max(latency_arr),
        "p50_latency": np.percentile(latency_arr, 50),
        "p95_latency": np.percentile(latency_arr, 95),
        "p99_latency": np.percentile(latency_arr, 99),
        "latency_std": np.std(latency_arr)
    }

    # ==== Save metrics ====
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2),
        "MARE": float(mare),
        "max_relative_error": float(max_rel),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "median_error": float(median_error),
        "samples_count": len(preds),
        **latency_metrics
    }

    metrics_path = os.path.join(result_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # ==== Save predictions ====
    results_df = pd.DataFrame({
        "true_value": trues_arr,
        "predicted_value": preds_arr,
        "error": errors
    })
    results_csv_path = os.path.join(result_dir, "predictions.csv")
    results_df.to_csv(results_csv_path, index=False)

    # ==== Save errors to txt ====
    errors_path = os.path.join(result_dir, "errors.txt")
    np.savetxt(errors_path, errors)

    # ==== Plots ====
    # 1) Error histogram
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    bin_width = 0.03
    errors_clean = [e for e in errors if not np.isnan(e)]
    min_error = min(errors_clean)
    max_error = max(errors_clean)
    bins = np.arange(min_error, max_error + bin_width, bin_width)
    hist, bin_edges = np.histogram(errors_clean, bins=bins)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', edgecolor='black')
    plt.xlabel('Δt (μs)')
    plt.ylabel('Sample Count')
    plt.tick_params(width=1)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height,
                     f'{int(height)}',
                     ha='center',
                     va='bottom',
                     fontsize=12,
                     fontname='Times New Roman')

    xtick_pos = bin_edges
    xtick_labels = [f'{x:.2f}' for x in xtick_pos]
    plt.xticks(xtick_pos, labels=xtick_labels, fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    histogram_path = os.path.join(result_dir, "error_histogram.jpg")
    plt.savefig(histogram_path, dpi=600, bbox_inches='tight')
    plt.close()

    # 2) Predicted vs True scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(trues_arr, preds_arr, alpha=0.5)
    min_val = min(np.min(trues_arr), np.min(preds_arr))
    max_val = max(np.max(trues_arr), np.max(preds_arr))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True TOF (μs)')
    plt.ylabel('Predicted TOF (μs)')
    plt.title('Predicted vs. True TOF')
    plt.grid(True)
    scatter_path = os.path.join(result_dir, "pred_vs_true_scatter.jpg")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Error boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(errors_clean)
    plt.ylabel('Error (μs)')
    plt.title('Error Distribution')
    boxplot_path = os.path.join(result_dir, "error_boxplot.jpg")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==== Latency histogram ====
    plt.figure(figsize=(10, 6))
    plt.hist(latency_arr, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Inference Latency (ms)')
    plt.ylabel('Sample Count')
    plt.title(f'{db_name} - Per-sample Inference Latency')
    plt.grid(True, linestyle='--', alpha=0.6)
    latency_hist_path = os.path.join(result_dir, "latency_histogram.jpg")
    plt.savefig(latency_hist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==== Create test report ====
    report = f"""
    Swin Transformer Test Report - {db_name}
    ======================================

    Test time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    Dataset:
    - Image dir: {img_dir}
    - Label dir: {label_dir}
    - Samples: {len(dataset)}

    Model:
    - Name: Swin Transformer Tiny
    - Path: {model_path}

    Metrics:
    - MAE: {mae:.4f} μs
    - MSE: {mse:.4f} μs²
    - R²: {r2:.4f}
    - MARE: {mare:.4f}
    - Max relative error: {max_rel:.4f}
    - Mean error: {mean_error:.4f} μs
    - Std of error: {std_error:.4f} μs
    - Median error: {median_error:.4f} μs

    Latency:
    - Avg: {latency_metrics['avg_latency']:.2f} ms
    - Min: {latency_metrics['min_latency']:.2f} ms
    - Max: {latency_metrics['max_latency']:.2f} ms
    - P50: {latency_metrics['p50_latency']:.2f} ms
    - P95: {latency_metrics['p95_latency']:.2f} ms
    - P99: {latency_metrics['p99_latency']:.2f} ms
    - Std: {latency_metrics['latency_std']:.2f} ms

    Outputs:
    - Metrics: {metrics_path}
    - Predictions: {results_csv_path}
    - Errors: {errors_path}
    - Error histogram: {histogram_path}
    - Pred vs True scatter: {scatter_path}
    - Error boxplot: {boxplot_path}
    - Latency histogram: {latency_hist_path}
    """

    report_path = os.path.join(result_dir, "test_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"DB: {db_name}")
    print(f"{'=' * 50}")
    print(f"Samples:          {len(preds)}")
    print(f"MAE:              {mae:.4f} μs")
    print(f"MSE:              {mse:.4f} μs²")
    print(f"R²:               {r2:.4f}")
    print(f"MARE:             {mare:.4f}")
    print(f"Max Rel Error:    {max_rel:.4f}")
    print(f"Avg latency:      {latency_metrics['avg_latency']:.2f} ms")
    print(f"Latency std:      {latency_metrics['latency_std']:.2f} ms")
    print(f"P99 latency:      {latency_metrics['p99_latency']:.2f} ms")
    print(f"All outputs saved at: {result_dir}")

    return result_dir


def main():
    # Create base result dir
    base_result_dir = "test_results_sw"
    os.makedirs(base_result_dir, exist_ok=True)

    # DB list to test
    db_list = ["-5DB", "0DB", "5DB", "10DB", "15DB"]

    # Base path template
    base_path = r"datasets/data_test/classified"

    all_results = []

    for db in db_list:
        img_dir = os.path.join(base_path, db, "target_ASDF")
        label_dir = os.path.join(base_path, db, "label")

        if not (os.path.exists(img_dir) and os.path.exists(label_dir)):
            print(f"Skip {db}, path not found")
            continue

        print(f"\nStart testing {db}...")
        result_dir = test_one_db(img_dir, label_dir, base_result_dir)
        all_results.append(result_dir)

    # Create summary report
    summary_report = f"""
    Swin Transformer Multi-DB Test Summary
    =====================================

    Test time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Number of DBs: {len(all_results)}

    Result directories:
    """


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()