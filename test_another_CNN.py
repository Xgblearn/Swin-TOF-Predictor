import os
import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import json
import argparse
import csv
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Font setup for plots
# -------------------------
def setup_plot_font():
    """Setup font support for plots"""
    try:
        # Try using SimHei font
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Using SimHei font for display")
    except:
        try:
            # Try using Microsoft YaHei font
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ Using Microsoft YaHei font for display")
        except:
            try:
                # Try using Arial Unicode MS font
                plt.rcParams['font.family'] = 'Arial Unicode MS'
                print("✅ Using Arial Unicode MS font for display")
            except:
                print("⚠ Warning: Unable to set font, charts may not display correctly")

# Initialize font display
setup_plot_font()

# -------------------------
# Dataset class (consistent with another_CNN.py)
# -------------------------
class MyDataset(Dataset):
    def __init__(self, target_dir, label_dir, indices=None, signal_len=3000):
        self.target_dir = target_dir
        self.label_dir = label_dir
        self.signal_len = signal_len

        if indices is None:
            # Automatically detect matching files
            target_files = glob.glob(os.path.join(target_dir, "*.txt"))
            label_files = glob.glob(os.path.join(label_dir, "*.txt"))
            
            # Extract filenames (without extension)
            target_names = {os.path.splitext(os.path.basename(f))[0] for f in target_files}
            label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}
            
            # Find files that exist in both folders
            common_names = sorted(target_names & label_names)
            
            # Try to convert filenames to integer indices
            indices = []
            for name in common_names:
                try:
                    idx = int(name)
                    indices.append(idx)
                except:
                    continue
            
            indices = sorted(indices)

        self.indices = list(indices)
        
        # Create file path list
        self.targets = [os.path.join(target_dir, f"{i}.txt") for i in self.indices]
        self.labels = [os.path.join(label_dir, f"{i}.txt") for i in self.indices]

        if not self.targets:
            raise RuntimeError("No matching sample files found")

        print(f"Found {len(self.targets)} valid samples")

    def __len__(self):
        return len(self.targets)

    def _read_target_signal(self, path):
        try:
            arr = np.loadtxt(path, usecols=(0,))
        except Exception as e:
            vals = []
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if s == '': continue
                    parts = re.split('[,\\s]+', s)
                    if parts:
                        try:
                            vals.append(float(parts[0]))
                        except:
                            continue
            arr = np.array(vals, dtype=np.float32)

        arr = arr.ravel().astype(np.float32)
        L = len(arr)

        if L == self.signal_len:
            sig = arr
        elif L > self.signal_len:
            start = (L - self.signal_len) // 2
            sig = arr[start:start + self.signal_len]
        else:
            sig = np.zeros(self.signal_len, dtype=np.float32)
            sig[:L] = arr

        m = np.max(np.abs(sig))
        if m > 0: sig /= m
        return sig

    def _read_label(self, path):
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if s == '': continue
                parts = re.split('[,\\s]+', s)
                if parts:
                    try:
                        return float(parts[0])
                    except:
                        continue
        raise ValueError(f"Unable to parse label: {path}")

    def __getitem__(self, idx):
        sig = self._read_target_signal(self.targets[idx])
        label = self._read_label(self.labels[idx])
        return (
            torch.tensor(sig, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32)
        )

# -------------------------
# CNN Model (copied from another_CNN.py)
# -------------------------
class PaperCNN_Ultrasonic(nn.Module):
    """
    1D CNN implementation based on paper structure:
    - 3 convolutional layers, 64 kernels each
    - Kernel size k=71
    - BN + ReLU after each layer
    - Dropout(0.2) after last convolutional layer
    - No pooling
    - Direct fully connected layer output after flatten
    """
    def __init__(self, input_channels=1, signal_len=3000, out_dim=1, dropout=0.2):
        super().__init__()
        k = 71
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=k, padding=k//2)
        self.bn3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Dynamically calculate flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, signal_len)
            x = self.relu(self.bn1(self.conv1(dummy)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.dropout(x)
            flat_dim = x.view(1, -1).shape[1]

        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.shape[1] == 1:
            return x.squeeze(1)  # Output (B,)
        return x

# -------------------------
# Utility functions
# -------------------------
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    non_zero = y_true != 0
    mare = np.mean(np.abs((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])) if np.any(non_zero) else float('nan')
    max_rel = np.max(np.abs((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])) if np.any(non_zero) else float('nan')
    return mae, mse, r2, mare, max_rel

def plot_pred_vs_true(true, pred, title, save_path):
    plt.figure(figsize=(8, 7))
    plt.scatter(true, pred, alpha=0.6)
    mn, mx = min(min(true), min(pred)), max(max(true), max(pred))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_relative_error_histogram(rel_errors, save_path, bin_width=0.03):
    """Plot relative error histogram"""
    # Calculate statistics
    min_error = min(rel_errors)
    max_error = max(rel_errors)
    mean_error = np.mean(rel_errors)
    median_error = np.median(rel_errors)

    # Create bins
    bins = np.arange(min_error, max_error + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(rel_errors, bins=bins, alpha=0.7, color='dodgerblue', edgecolor='black')
    
    # Add statistical lines
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.4f}')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_error:.4f}')
    
    # Add title and labels
    plt.title('Relative Error Distribution', fontsize=14)
    plt.xlabel('Relative Error', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Relative error histogram saved to: {save_path}")

def measure_inference_latency(model, device, input_size=(1, 1, 3000), num_runs=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start_time) * 1000 / num_runs

def calculate_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return sum(p.numel() for p in model.parameters()), (param_size + buffer_size) / (1024 ** 2)

# -------------------------
# Error statistics cube plot
# -------------------------
def plot_error_cube(true, pred, save_path):
    """
    Plot error statistics cube with three-dimensional error analysis
    """
    errors = np.abs(np.array(pred) - np.array(true))
    rel_errors = np.abs(errors / np.array(true))
    rel_errors[np.isinf(rel_errors)] = np.nan

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Model Error Statistics Cube', fontsize=16)

    # 1. Absolute error distribution
    ax1 = fig.add_subplot(131)
    ax1.hist(errors, bins=30, alpha=0.7, color='dodgerblue')
    ax1.set_title('Absolute Error Distribution')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Sample Count')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add statistical information
    mean_error = np.nanmean(errors)
    median_error = np.nanmedian(errors)
    ax1.axvline(mean_error, color='red', linestyle='dashed', linewidth=1)
    ax1.text(mean_error, plt.ylim()[1] * 0.9,
             f'Mean: {mean_error:.4f}', color='red')
    ax1.axvline(median_error, color='green', linestyle='dashed', linewidth=1)
    ax1.text(median_error, plt.ylim()[1] * 0.8,
             f'Median: {median_error:.4f}', color='green')

    # 2. Relative error vs true value relationship
    ax2 = fig.add_subplot(132)
    valid_indices = ~np.isnan(rel_errors) & ~np.isinf(rel_errors)
    sc = ax2.scatter(np.array(true)[valid_indices],
                     rel_errors[valid_indices],
                     c=rel_errors[valid_indices],
                     cmap='viridis', alpha=0.6)
    ax2.set_title('True Value vs Relative Error')
    ax2.set_xlabel('True Value')
    ax2.set_ylabel('Relative Error')
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(sc, ax=ax2, label='Relative Error Magnitude')

    # 3. Absolute error boxplot (grouped by true value range)
    ax3 = fig.add_subplot(133)

    # Divide true values into 4 intervals
    true_min, true_max = np.nanmin(true), np.nanmax(true)
    boundaries = np.linspace(true_min, true_max, 5)
    group_labels = [f'{boundaries[i]:.1f}-{boundaries[i + 1]:.1f}'
                    for i in range(len(boundaries) - 1)]

    group_errors = []
    for i in range(len(boundaries) - 1):
        mask = (true >= boundaries[i]) & (true < boundaries[i + 1])
        group_errors.append(errors[mask])

    ax3.boxplot(group_errors, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='darkblue'),
                medianprops=dict(color='darkred'))
    ax3.set_title('Error Distribution by True Value Range')
    ax3.set_xticklabels(group_labels)
    ax3.set_xlabel('True Value Range')
    ax3.set_ylabel('Absolute Error')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------
# Main testing function
# -------------------------
def run_testing(target_dir, label_dir, model_path,
                batch_size=32, signal_len=3000, exp_name="another_cnn_test"):
    """
    Evaluate model performance on independent test set
    Automatically handle non-continuous file names
    """
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"test_results_another_CNN/{exp_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"▶ Test results will be saved to: {save_dir}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙ Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = PaperCNN_Ultrasonic(signal_len=signal_len).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    scaler = checkpoint['scaler']
    print(f"✅ Model loaded: {model_path}")

    # Create test dataset (automatically handle non-continuous file names)
    test_dataset = MyDataset(target_dir, label_dir, signal_len=signal_len)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)
    print(f"📊 Test sample count: {len(test_dataset)}")

    # Execute testing
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)

            # Denormalize predictions
            pred_orig = scaler.inverse_transform(
                pred.cpu().numpy().reshape(-1, 1)
            ).ravel()

            test_preds.extend(pred_orig)
            test_trues.extend(y.cpu().numpy())

    # Calculate absolute and relative errors
    errors = np.array(test_preds) - np.array(test_trues)

    # Calculate metrics
    mae, mse, r2, mare, max_rel = calc_metrics(test_trues, test_preds)

    # Print metrics
    print("\n⭐ Test Results ⭐")
    print(f"- MAE: {mae:.6f}")
    print(f"- MSE: {mse:.6f}")
    print(f"- R²: {r2:.4f}")
    print(f"- MARE: {mare:.4f}")
    print(f"- Max relative error: {max_rel:.4f}")

    # Calculate model information
    num_params, model_size = calculate_model_size(model)
    latency = measure_inference_latency(model, device)

    print("\n📐 Model Information:")
    print(f"- Parameter count: {num_params:,}")
    print(f"- Model size: {model_size:.2f} MB")
    print(f"- Inference latency: {latency:.2f} ms")

    # Save test results
    test_results = {
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mare': float(mare) if not np.isnan(mare) else None,
        'max_rel_error': float(max_rel) if not np.isnan(max_rel) else None,
        'num_samples': len(test_trues),
        'model_params': int(num_params),
        'model_size_mb': float(model_size),
        'inference_latency_ms': float(latency),
        'target_dir': target_dir,
        'label_dir': label_dir,
        'model_path': model_path
    }

    with open(os.path.join(save_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=4)

    # Visualize results
    plot_pred_vs_true(test_trues, test_preds,
                      'Test Set: Predicted vs True Values',
                      os.path.join(save_dir, "pred_vs_true.png"))

    plot_error_cube(test_trues, test_preds,
                    os.path.join(save_dir, "error_cube.png"))

    # Save error histogram
    plot_relative_error_histogram(errors,
                                 os.path.join(save_dir, "relative_error_histogram.png"),
                                 bin_width=0.03)

    # Save error data file
    error_data_path = os.path.join(save_dir, "error_data.csv")
    with open(error_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'True_Value', 'Predicted_Value', 'Absolute_Error', 'Relative_Error'])
        for i, (t, p) in enumerate(zip(test_trues, test_preds)):
            abs_err = abs(p - t)
            rel_err = abs_err / abs(t) if t != 0 else float('nan')
            writer.writerow([i, t, p, abs_err, rel_err])
    print(f"✅ Error data file saved to: {error_data_path}")

    # Save prediction data for further analysis
    prediction_data_path = os.path.join(save_dir, "predictions.csv")
    with open(prediction_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'True_Value', 'Predicted_Value', 'Absolute_Error', 'Relative_Error'])
        for i, (t, p) in enumerate(zip(test_trues, test_preds)):
            abs_err = abs(p - t)
            rel_err = abs_err / abs(t) if t != 0 else float('nan')
            writer.writerow([i, t, p, abs_err, rel_err])
    print(f"✅ Prediction data file saved to: {prediction_data_path}")

    # Print test summary
    print("\n📊 Test Summary:")
    print(f"- Test sample count: {len(test_trues)}")
    print(f"- Mean Absolute Error (MAE): {mae:.4f}")
    print(f"- Mean Absolute Relative Error (MARE): {mare:.4f}")
    print(f"- Max relative error: {max_rel:.4f}")
    print(f"- Model explained variance (R²): {r2:.4f}")
    print(f"✅ Testing completed! All results saved to: {save_dir}")

    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Another CNN model testing script (supports non-continuous file names)')
    parser.add_argument('--target_dir', default="datasets\\data_test\\classified\\15DB\\target_signal", type=str, help='Test set target data directory')
    parser.add_argument('--label_dir', default="datasets\\data_test\\classified\\15DB\\label", type=str, help='Test set label data directory')
    parser.add_argument('--model_path', default="saved\\best_another_cnn.pth", type=str, help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--exp_name', type=str, default="another_cnn_15DB", help='Test experiment name')

    args = parser.parse_args()

    # Run testing
    run_testing(
        target_dir=args.target_dir,
        label_dir=args.label_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        exp_name=args.exp_name
    )