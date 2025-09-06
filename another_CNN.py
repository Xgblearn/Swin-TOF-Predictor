import os, re, random, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# MyDataset: 1st column of target (length 3000), label: first line first column
# -------------------------
class MyDataset(Dataset):
    def __init__(self, target_dir, label_dir, indices=None, signal_len=3000):
        """
        target_dir: contains 1.txt .. N.txt each with >=3000 rows, we take first column
        label_dir: contains 1.txt .. N.txt each with label in first line first column
        indices: iterable of int (1..N). If None, will auto-detect matching files.
        """
        self.target_dir = target_dir
        self.label_dir = label_dir
        self.signal_len = signal_len

        if indices is None:
            # find all target files named like \d+.txt and ensure label exists
            files = sorted([f for f in os.listdir(target_dir) if f.endswith('.txt')])
            indices = []
            for fn in files:
                name = os.path.splitext(fn)[0]
                lab = os.path.join(label_dir, name + '.txt')
                if os.path.exists(lab):
                    try:
                        idx = int(name)
                        indices.append(idx)
                    except:
                        continue
            indices = sorted(indices)

        self.indices = list(indices)

        # build full paths lists
        self.targets = [os.path.join(target_dir, f"{i}.txt") for i in self.indices]
        self.labels = [os.path.join(label_dir, f"{i}.txt") for i in self.indices]

        if len(self.targets) == 0:
            raise RuntimeError("No samples found. Check target_dir/label_dir and indexing.")

    def __len__(self):
        return len(self.targets)

    def _read_target_signal(self, path):
        # read first column of text file, handle large files robustly
        # use numpy.loadtxt with usecols=0
        try:
            arr = np.loadtxt(path, usecols=(0,))
        except Exception as e:
            # fallback: read lines and parse first value each line
            vals = []
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if s == '':
                        continue
                    # split by whitespace or comma
                    parts = re.split('[,\\s]+', s)
                    try:
                        v = float(parts[0])
                        vals.append(v)
                    except:
                        continue
            arr = np.array(vals, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32).ravel()

        # If arr length != signal_len, crop center or pad zeros at end
        L = len(arr)
        if L == self.signal_len:
            sig = arr
        elif L > self.signal_len:
            start = (L - self.signal_len) // 2
            sig = arr[start:start + self.signal_len]
        else:
            sig = np.zeros(self.signal_len, dtype=np.float32)
            sig[:L] = arr

        # Normalize by absolute max to keep scale stable (paper-like)
        m = np.max(np.abs(sig))
        if m > 0:
            sig = sig / m
        return sig.astype(np.float32)

    def _read_label(self, path):
        # read first line first column as float
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if s == '':
                    continue
                parts = re.split('[,\\s]+', s)
                try:
                    v = float(parts[0])
                    return float(v)
                except:
                    continue
        raise ValueError(f"Cannot parse label in {path}")

    def __getitem__(self, idx):
        target_path = self.targets[idx]
        label_path = self.labels[idx]

        sig = self._read_target_signal(target_path)   # shape (signal_len,)
        label = self._read_label(label_path)

        # return torch tensors later in DataLoader collate, but we'll return numpy and convert outside
        # For convenience return as torch tensors here:
        import torch as _torch
        sig_t = _torch.tensor(sig, dtype=_torch.float32).unsqueeze(0)  # (1, L)
        lab_t = _torch.tensor(label, dtype=_torch.float32)
        return sig_t, lab_t

# -------------------------
# CNN Model for signal length 3000 -> single scalar
# -------------------------
class PaperCNN_Ultrasonic(nn.Module):
    """
    根据论文结构实现的 1D CNN:
    - 3 个卷积层，每层 64 个卷积核
    - 卷积核大小 k=71
    - 每层后接 BN + ReLU
    - 最后一层卷积后加 Dropout(0.2)
    - 不用池化
    - Flatten 后直接接全连接层输出
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

        # 动态计算 flatten 大小
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
            return x.squeeze(1)  # 输出 (B,)
        return x

# -------------------------
# utilities: metrics and plotting
# -------------------------
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    non_zero = y_true != 0
    if np.any(non_zero):
        rel = np.abs((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])
        mare = np.mean(rel)
        max_rel = np.max(rel)
    else:
        mare = float('nan'); max_rel = float('nan')
    return mae, mse, r2, mare, max_rel

def plot_pred_vs_true(true, pred, title='Predicted vs True'):
    plt.figure(figsize=(7,6))
    plt.scatter(true, pred, alpha=0.5)
    mn = min(min(true), min(pred)); mx = max(max(true), max(pred))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel('True'); plt.ylabel('Predicted'); plt.title(title)
    plt.grid(True)
    plt.show()

# -------------------------
# Main training pipeline
# -------------------------
# def run_training(target_dir='data/target1', label_dir='data/label',
#                  indices=range(1,2500), batch_size=32, epochs=30, lr=1e-4, signal_len=3000,
#                  save_model='best_cnn.pth', seed=42):
#     # reproducibility
#     torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
#
#     dataset = MyDataset(target_dir, label_dir, indices=indices, signal_len=signal_len)
#     N = len(dataset)
#     train_n = int(0.7 * N); val_n = int(0.2 * N); test_n = N - train_n - val_n
#     train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n],
#                                              generator=torch.Generator().manual_seed(seed))
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device:", device)
#
#     model = PaperCNN_Ultrasonic(input_channels=1, signal_len=signal_len).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#
#     # fit label scaler on train set (so training is stable) -> we'll inverse transform preds later
#     labels_train = []
#     for _, lab in train_loader:
#         labels_train.append(lab.numpy().ravel())
#     labels_train = np.concatenate(labels_train).reshape(-1,1)
#     scaler = MinMaxScaler((0,1)).fit(labels_train)
#
#     best_val_loss = float('inf')
#     for ep in range(1, epochs+1):
#         model.train()
#         train_loss_sum = 0.0
#         for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=100):
#             x = x.to(device)  # (B,1,L)
#             # scale labels
#             y_np = y.numpy().reshape(-1,1)
#             y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)
#
#             optimizer.zero_grad()
#             out = model(x)
#             loss = loss_fn(out, y_scaled)
#             loss.backward()
#             optimizer.step()
#             train_loss_sum += loss.item() * x.size(0)
#         train_loss = train_loss_sum / len(train_loader.dataset)
#
#         # validation
#         model.eval()
#         val_loss_sum = 0.0
#         val_preds = []; val_trues = []
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x = x.to(device)
#                 y_np = y.numpy().reshape(-1,1)
#                 y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)
#                 out = model(x)
#                 loss = loss_fn(out, y_scaled)
#                 val_loss_sum += loss.item() * x.size(0)
#                 val_preds.append(out.cpu().numpy().ravel())
#                 val_trues.append(y.numpy().ravel())
#         val_loss = val_loss_sum / len(val_loader.dataset)
#         val_preds = np.concatenate(val_preds)
#         val_trues = np.concatenate(val_trues)
#         # inverse transform preds
#         val_preds_orig = scaler.inverse_transform(val_preds.reshape(-1,1)).ravel()
#
#         mae, mse, r2, mare, maxrel = calc_metrics(val_trues, val_preds_orig)
#         print(f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} | val MAE={mae:.6f} MSE={mse:.6f} R2={r2:.4f}")
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save({'model': model.state_dict(), 'scaler': scaler}, save_model)
#             print("Saved best:", save_model)
#
#     # ---- test ----
#     chk = torch.load(save_model, map_location=device)
#     model.load_state_dict(chk['model'])
#     scaler = chk['scaler']
#
#     debug_shapes(model, signal_len=3000)
#
#     preds = []; trues = []
#     model.eval()
#     with torch.no_grad():
#         for x, y in test_loader:
#             x = x.to(device)
#             out = model(x)
#             preds.append(out.cpu().numpy().ravel())
#             trues.append(y.numpy().ravel())
#     preds = np.concatenate(preds)
#     trues = np.concatenate(trues)
#     preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).ravel()
#
#     mae, mse, r2, mare, maxrel = calc_metrics(trues, preds_orig)
#     print("TEST:", f"MAE={mae:.6f}, MSE={mse:.6f}, R2={r2:.4f}, MARE={mare:.6f}, max_rel_err={maxrel:.6f}")
#
#     plot_pred_vs_true(trues, preds_orig, 'Test: Pred vs True')

def run_training(target_dir='data/target1', label_dir='data/label',
                 indices=range(1, 2500), batch_size=32, epochs=30, lr=1e-4, signal_len=3000,
                 save_model='best_cnn.pth', seed=42, output_dir='results'):
    """
    完整的训练函数，包括保存训练过程参数、图片和测试结果指标
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 记录训练过程
    train_losses = []
    val_losses = []
    val_metrics = []  # 存储每个epoch的验证指标

    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建数据集
    dataset = MyDataset(target_dir, label_dir, indices=indices, signal_len=signal_len)
    N = len(dataset)
    train_n = int(0.7 * N)
    val_n = int(0.2 * N)
    test_n = N - train_n - val_n

    # 划分数据集
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(seed)
    )

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 初始化模型
    model = PaperCNN_Ultrasonic(input_channels=1, signal_len=signal_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 准备标签归一化器（使用训练集）
    labels_train = []
    for _, lab in train_loader:
        labels_train.append(lab.numpy().ravel())
    labels_train = np.concatenate(labels_train).reshape(-1, 1)
    scaler = MinMaxScaler((0, 1)).fit(labels_train)

    # 训练循环
    best_val_loss = float('inf')
    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        # 训练阶段
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=100):
            x = x.to(device)  # (B,1,L)

            # 归一化标签
            y_np = y.numpy().reshape(-1, 1)
            y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y_scaled)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_np = y.numpy().reshape(-1, 1)
                y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)
                out = model(x)
                loss = loss_fn(out, y_scaled)
                val_loss_sum += loss.item() * x.size(0)
                val_preds.append(out.cpu().numpy().ravel())
                val_trues.append(y.numpy().ravel())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        # 反归一化预测值
        val_preds_orig = scaler.inverse_transform(val_preds.reshape(-1, 1)).ravel()

        # 计算验证指标
        mae, mse, r2, mare, maxrel = calc_metrics(val_trues, val_preds_orig)
        val_metrics.append({
            'epoch': ep,
            'mae': float(mae),
            'mse': float(mse),
            'r2': float(r2),
            'mare': float(mare),
            'maxrel': float(maxrel)
        })

        print(f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
              f"val MAE={mae:.6f} MSE={mse:.6f} R2={r2:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'scaler': scaler,
                'epoch': ep,
                'val_loss': val_loss
            }, os.path.join(output_dir, save_model))
            print(f"Saved best model: {os.path.join(output_dir, save_model)}")

    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to: {loss_curve_path}")

    # 保存训练过程数据 - 确保所有数据都是可序列化的Python原生类型
    training_history = {
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'val_metrics': val_metrics,
        'best_epoch': int(np.argmin(val_losses) + 1),
        'best_val_loss': float(min(val_losses))
    }

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"Training history saved to: {history_path}")

    # 加载最佳模型进行测试
    best_model_path = os.path.join(output_dir, save_model)
    chk = torch.load(best_model_path, map_location=device)
    model.load_state_dict(chk['model'])
    scaler = chk['scaler']

    # 测试阶段
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy().ravel())
            trues.append(y.numpy().ravel())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    # 计算测试指标
    mae, mse, r2, mare, maxrel = calc_metrics(trues, preds_orig)

    # 保存测试集预测vs真实值散点图
    plt.figure(figsize=(7, 6))
    plt.scatter(trues, preds_orig, alpha=0.5)
    mn = min(min(trues), min(preds_orig))
    mx = max(max(trues), max(preds_orig))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Test: Predicted vs True')
    plt.grid(True)
    scatter_path = os.path.join(output_dir, 'test_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Test scatter plot saved to: {scatter_path}")

    # 保存测试结果指标
    test_metrics = {
        'MAE': float(mae),
        'MSE': float(mse),
        'R2': float(r2),
        'MARE': float(mare),
        'max_rel_err': float(maxrel)
    }

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test metrics saved to: {metrics_path}")

    # 打印测试结果
    print("\nTEST RESULTS:")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"MARE: {mare:.6f}")
    print(f"Max Relative Error: {maxrel:.6f}")

    # 创建汇总报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""Training Report - {timestamp}
==============================

Dataset:
- Target directory: {target_dir}
- Label directory: {label_dir}
- Samples: {len(indices)}
- Signal length: {signal_len}

Training Parameters:
- Batch size: {batch_size}
- Epochs: {epochs}
- Learning rate: {lr}
- Seed: {seed}

Results:
- Best model saved: {os.path.join(output_dir, save_model)}
- Best epoch: {training_history['best_epoch']}
- Best validation loss: {training_history['best_val_loss']:.6f}

Test Metrics:
- MAE: {mae:.6f}
- MSE: {mse:.6f}
- R2: {r2:.6f}
- MARE: {mare:.6f}
- Max Relative Error: {maxrel:.6f}

Output Files:
- Loss curve: {loss_curve_path}
- Test scatter plot: {scatter_path}
- Training history: {history_path}
- Test metrics: {metrics_path}
"""

    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Training report saved to: {report_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")
    return test_metrics


def debug_shapes(model, signal_len=3000):
    import torch
    dummy = torch.randn(1, 1, signal_len).to(next(model.parameters()).device)
    x = dummy
    print("input:", x.shape)

    x = model.relu(model.bn1(model.conv1(x)))
    print("after conv1:", x.shape)

    x = model.relu(model.bn2(model.conv2(x)))
    print("after conv2:", x.shape)

    x = model.relu(model.bn3(model.conv3(x)))
    print("after conv3:", x.shape)

    x = model.dropout(x)
    print("after dropout:", x.shape)

    flat = x.view(1, -1)
    print("flatten:", flat.shape)

    out = model.fc(flat)
    print("output:", out.shape)

# 用法：
# debug_shapes(model, signal_len=3000)

# If run as script:
if __name__ == "__main__":
    # adjust paths if needed
    # run_training(target_dir='data/target1', label_dir='data/label',
    #              indices=range(1,2501), batch_size=32, epochs=10, lr=1e-3, signal_len=3000, save_model='best_cnn.pth')

    # 运行训练并保存所有结果
    test_metrics = run_training(
        target_dir='datasets/data_signal/target',
        label_dir='datasets/data_signal/label',
        indices=range(1, 10001),
        batch_size=32,
        epochs=200,
        lr=1e-4,
        signal_len=3000,
        save_model='best_another_cnn.pth',
        output_dir='results_another_cnn'  # 指定输出目录
    )