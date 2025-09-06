import shutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from utils.swin_transformer_modules import SwinTransformer
from utils.Dataset import MyDataset
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import csv
import argparse
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=20, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⚠️ Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        if self.verbose:
            print(f'✅ Validation loss improved ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

# Training loop
def train(model, train_loader, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", ncols=100)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        if scaler is not None:
            labels = scaler.fit_transform(labels.cpu().numpy().reshape(-1, 1))
            labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = log_cosh_loss(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
    avg_loss = running_loss / len(train_loader)
    return avg_loss


# Validation loop
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = log_cosh_loss(outputs.squeeze(), labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# Test loop
def test(model, test_loader, device, scaler=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                labels = scaler.fit_transform(labels.cpu().numpy().reshape(-1, 1))
                labels = torch.tensor(labels).to(device)
            outputs = model(images)
            loss = log_cosh_loss(outputs.squeeze(), labels)
            running_loss += loss.item()
            all_preds.append(outputs.squeeze().cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))

    avg_loss = running_loss / len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Plot predicted vs. true scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, color='blue', alpha=0.5)
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.savefig(f'training_SW_result/predicted_vs_true_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

    num_params, model_size_mb = calculate_model_size(model)
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    return avg_loss, all_preds, all_labels


# Compute metrics
def calculate_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    non_zero_mask = all_labels != 0
    if np.any(non_zero_mask):
        relative_errors = np.abs((all_preds[non_zero_mask] - all_labels[non_zero_mask]) / all_labels[non_zero_mask])
        mare = np.mean(relative_errors)
        max_relative_error = np.max(relative_errors)
    else:
        mare = float('nan')
        max_relative_error = float('nan')

    return mae, mse, r2, mare, max_relative_error


# Single-sample inference latency test
def measure_inference_latency(model, device, input_size=(1, 3, 224, 224), num_runs=100, warmup_runs=10):
    model.eval()

    # Create random input
    dummy_input = torch.randn(input_size).to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    # Sync CUDA if necessary
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Compute average latency (ms)
    avg_latency = (end_time - start_time) / num_runs * 1000

    return avg_latency


# Loss functions
def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return loss.mean()


def aleatoric_loss(mu, log_var, target, min_logvar=-5.0, max_logvar=5.0):
    log_var = torch.clamp(log_var, min=min_logvar, max=max_logvar)
    precision = torch.exp(-log_var)
    loss = precision * (target - mu) ** 2 + log_var
    return loss.mean()


# Model definition
def my_swin_tiny_patch4_window7_224(num_classes: int = 1, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        **kwargs
    )
    model = model.to(device)
    return model, device


# Dataloaders
def get_dataloaders(image_dir="datasets/data_to_SW/target",
                    label_dir='datasets/data_to_SW/label',
                    indices=range(1, 10001),
                    batch_size=32,
                    generator=None,
                    save_test_data=False,
                    save_image_dir='saved_data/test/images',
                    save_label_dir='saved_data/test/labels'):
    dataset = MyDataset(image_dir=image_dir, label_dir=label_dir, indices=indices)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if save_test_data:
        print("Saving test images and labels...")
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)

        test_indices = test_dataset.indices
        for idx in test_indices:
            image_name = f'GASF_Image_{idx}.jpg'
            label_name = f'{idx}.txt'

            src_img_path = os.path.join(image_dir, image_name)
            src_label_path = os.path.join(label_dir, label_name)
            dst_img_path = os.path.join(save_image_dir, image_name)
            dst_label_path = os.path.join(save_label_dir, label_name)

            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_label_path, dst_label_path)

    return train_loader, val_loader, test_loader


# Model size computation
def calculate_model_size(model):
    """Compute total parameters and model size (MB)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size
    total_size_mb = total_size / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())

    return num_params, total_size_mb


# Save training params
def save_training_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)


# Save loss logs
def save_loss_logs(train_losses, val_losses, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i + 1, t_loss, v_loss])


# Plot loss curves
def plot_loss_curves(train_losses, val_losses, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Swin Transformer Training')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--flag', type=str, default='test', help='Experiment flag')
    args = parser.parse_args()

    # Init model
    model, device = my_swin_tiny_patch4_window7_224(num_classes=1)

    num_params, model_size_mb = calculate_model_size(model)
    print(f"Initial total parameters: {num_params:,}")
    print(f"Initial model size: {model_size_mb:.2f} MB")

    # Set random seed (optional)
    generator = None
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        generator=generator,
        save_test_data=True,
        save_image_dir='saved_data/test/images',
        save_label_dir='saved_data/test/labels'
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training params
    num_epochs = args.epochs
    flag = args.flag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'saved_model/swin_best_model_{flag}_{timestamp}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs('training_SW_result', exist_ok=True)

    num_params, model_size_mb = calculate_model_size(model)
    print(f"\n最终模型参数数量: {num_params:,}")
    print(f"模型大小: {model_size_mb:.2f} MB")

    # Save training params
    training_params = {
        'model': 'SwinTransformer',
        'epochs': num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'early_stopping': args.early_stop,
        'patience': args.patience if args.early_stop else 'N/A',
        'timestamp': timestamp
    }
    save_training_params(training_params, f'training_SW_result/training_params_{flag}_{timestamp}.json')

    # Init early stopping if enabled
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience, delta=0.001, verbose=True)
    else:
        early_stopping = None

    # Train and validate
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    writer = SummaryWriter(f"training_SW_result/tensorboard_{flag}_{timestamp}")

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练
        model.train()
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Early stopping check
        if args.early_stop:
            early_stopping(val_loss, model, save_path)
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            # Save best model if val loss improves without early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"✅ Validation improved, saved model to {save_path}")


    # Total training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s ({total_time / 60:.2f} min)")

    # Close TensorBoard writer
    writer.close()

    # Save loss logs
    save_loss_logs(train_losses, val_losses, f'training_SW_result/loss_log_{flag}_{timestamp}.csv')

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, f'training_SW_result/loss_curves_{flag}_{timestamp}.png')

    # Load best model and test
    print("\n------ Start Testing ------")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Recompute best model size
    num_params, model_size_mb = calculate_model_size(model)
    print(f"Best model total parameters: {num_params:,}")
    print(f"Best model size: {model_size_mb:.2f} MB")

    test_loss, all_preds, all_labels = test(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    # Compute and print extra metrics
    mae, mse, r2, mare, max_relative_error = calculate_metrics(model, test_loader, device)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, "
          f"MARE: {mare:.4f}, Max Relative Error: {max_relative_error:.4f}")

    # Inference latency
    print("\n------ Inference Latency ------")
    latency = measure_inference_latency(model, device)
    print(f"Average latency per sample: {latency:.2f} ms")

    # 保存测试结果
    test_results = {
        'test_loss': float(test_loss),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mare': float(mare) if not np.isnan(mare) else 'NaN',
        'max_relative_error': float(max_relative_error) if not np.isnan(max_relative_error) else 'NaN',
        'inference_latency_ms': float(latency),
        'total_parameters': int(num_params),
        'model_size_mb': float(model_size_mb),
        'timestamp': timestamp
    }

    with open(f'training_SW_result/test_results_{flag}_{timestamp}.json', 'w') as f:
        json.dump(test_results, f, indent=4)