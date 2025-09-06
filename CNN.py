import os, re, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# -------------------------
# 数据集类
# -------------------------
class MyDataset(Dataset):
    def __init__(self, target_dir, label_dir, indices=None, signal_len=3000):
        self.target_dir = target_dir
        self.label_dir = label_dir
        self.signal_len = signal_len

        if indices is None:
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
        self.targets = [os.path.join(target_dir, f"{i}.txt") for i in self.indices]
        self.labels = [os.path.join(label_dir, f"{i}.txt") for i in self.indices]

        if len(self.targets) == 0:
            raise RuntimeError("未找到有效样本")

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
        raise ValueError(f"无法解析标签: {path}")

    def __getitem__(self, idx):
        sig = self._read_target_signal(self.targets[idx])
        label = self._read_label(self.labels[idx])
        return (
            torch.tensor(sig, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32)
        )


# -------------------------
# 早停机制
# -------------------------
class EarlyStopping:
    def __init__(self, patience=40, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# -------------------------
# CNN 模型
# -------------------------
class Paper1DCNN(nn.Module):
    def __init__(self, input_channels=1, signal_len=3000, out_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 4, kernel_size=9, stride=4, padding=3)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=9, stride=4, padding=3)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # 动态计算扁平化维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, signal_len)
            x = self.conv1(dummy)
            x = self.relu(self.bn1(x))
            x = self.conv2(x)
            x = self.relu(self.bn2(x))
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 400)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(400, out_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1) if x.shape[1] == 1 else x


# -------------------------
# 工具函数
# -------------------------
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    non_zero = y_true != 0
    mare = np.mean(np.abs((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])) if np.any(non_zero) else float(
        'nan')
    return mae, mse, r2, mare


def plot_pred_vs_true(true, pred, title, save_path):
    plt.figure(figsize=(8, 7))
    plt.scatter(true, pred, alpha=0.6)
    mn, mx = min(min(true), min(pred)), max(max(true), max(pred))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, 'b-', label='训练损失')
    plt.plot(val_loss, 'r-', label='验证损失')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('训练过程损失曲线', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def measure_inference_latency(model, device, input_size=(1, 1, 3000), num_runs=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
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


def convert_to_serializable(obj):
    """将NumPy数据类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.generic):
        return obj.item()  # 转换NumPy标量到Python原生类型
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def save_training_report(report, save_dir):
    # 转换所有NumPy类型到Python原生类型
    serializable_report = convert_to_serializable(report)

    # 保存JSON报告
    with open(os.path.join(save_dir, "training_report.json"), 'w') as f:
        json.dump(serializable_report, f, indent=4)

    # 保存CSV指标
    with open(os.path.join(save_dir, "metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae'])
        for epoch, data in enumerate(zip(
                serializable_report['train_loss'],
                serializable_report['val_loss'],
                serializable_report['val_mae']
        )):
            writer.writerow([epoch + 1] + list(data))


# -------------------------
# 主训练函数
# -------------------------
def run_training(target_dir, label_dir, indices, batch_size=32, epochs=200, lr=1e-3,
                 signal_len=3000, save_model='best_model.pth', exp_flag='exp', seed=42):
    # 创建归档目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_results_CNN/{exp_flag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"▶ 所有结果将保存至: {save_dir}")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ 使用设备: {device}")

    # 初始化模型
    model = Paper1DCNN(signal_len=signal_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping()

    # 数据集加载
    dataset = MyDataset(target_dir, label_dir, indices, signal_len)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 标签归一化器
    labels_train = []
    for _, lab in train_loader:
        labels_train.append(lab.numpy())
    scaler = MinMaxScaler().fit(np.concatenate(labels_train).reshape(-1, 1))

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # 标签归一化
            y_scaled = torch.tensor(
                scaler.transform(y.cpu().numpy().reshape(-1, 1)).ravel(),
                dtype=torch.float32, device=device
            )

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y_scaled)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        train_loss = epoch_loss / len(train_ds)
        history['train_loss'].append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_scaled = torch.tensor(
                    scaler.transform(y.cpu().numpy().reshape(-1, 1)).ravel(),
                    dtype=torch.float32, device=device
                )

                pred = model(x)
                loss = loss_fn(pred, y_scaled)
                val_loss += loss.item() * x.size(0)

                # 反归一化预测
                pred_orig = scaler.inverse_transform(
                    pred.cpu().numpy().reshape(-1, 1)
                ).ravel()

                val_preds.extend(pred_orig)
                val_trues.extend(y.cpu().numpy())

        val_loss /= len(val_ds)
        val_mae = mean_absolute_error(val_trues, val_preds)

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # 早停检查和模型保存
        early_stopper(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'scaler': scaler
            }, os.path.join(save_dir, save_model))
            print(f"✅ 模型保存: val_loss={val_loss:.6f}, val_mae={val_mae:.6f}")

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_mae={val_mae:.6f}")

        if early_stopper.early_stop:
            print(f"🛑 早停触发! 在 {epoch} 个epoch后停止训练")
            break

    # 保存训练报告
    save_training_report(history, save_dir)
    plot_loss_curves(history['train_loss'], history['val_loss'],
                     os.path.join(save_dir, "loss_curve.png"))

    # 测试阶段
    print("\n🔍 测试阶段开始")
    checkpoint = torch.load(os.path.join(save_dir, save_model), map_location=device)
    model.load_state_dict(checkpoint['model'])
    scaler = checkpoint['scaler']
    model.eval()

    # 计算模型信息
    num_params, model_size = calculate_model_size(model)
    latency = measure_inference_latency(model, device)

    # 测试推理
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)

            # 反归一化
            pred_orig = scaler.inverse_transform(
                pred.cpu().numpy().reshape(-1, 1)
            ).ravel()

            test_preds.extend(pred_orig)
            test_trues.extend(y.cpu().numpy())

    # 计算指标
    mae, mse, r2, mare = calc_metrics(test_trues, test_preds)

    # 生成最终报告
    final_report = {
        'model_params': int(num_params),
        'model_size_mb': float(model_size),
        'inference_latency_ms': float(latency),
        'test_mae': float(mae),
        'test_mse': float(mse),
        'test_r2': float(r2),
        'test_mare': float(mare) if not np.isnan(mare) else None,
        'training_epochs': len(history['train_loss']),
        'best_val_loss': float(best_val_loss)
    }

    # 保存测试结果
    with open(os.path.join(save_dir, "test_results.json"), 'w') as f:
        json.dump(final_report, f, indent=4)

    # 可视化
    plot_pred_vs_true(test_trues, test_preds,
                      '测试集: 预测值 vs 真实值',
                      os.path.join(save_dir, "pred_vs_true.png"))

    # 打印摘要
    print("\n⭐ 训练摘要 ⭐")
    print(f"- 模型参数: {num_params:,}")
    print(f"- 模型大小: {model_size:.2f} MB")
    print(f"- 推理延迟: {latency:.2f} ms")
    print(f"- 测试MAE: {mae:.6f}")
    print(f"- 测试R²: {r2:.4f}")
    print(f"✅ 训练完成! 结果保存至: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D CNN训练脚本')
    parser.add_argument('--target_dir', type=str, default="datasets/data_signal/target", help='目标数据目录')
    parser.add_argument('--label_dir', type=str, default="datasets/data_signal/label", help='标签数据目录')
    parser.add_argument('--indices', type=str, default='1-10000', help='样本索引范围 (e.g., 1-10000)')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--exp', type=str, default='cnn_200', help='实验标识')

    args = parser.parse_args()

    # 解析索引范围
    if '-' in args.indices:
        start, end = map(int, args.indices.split('-'))
        indices = range(start, end + 1)
    else:
        indices = list(map(int, args.indices.split(',')))

    run_training(
        target_dir=args.target_dir,
        label_dir=args.label_dir,
        indices=indices,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        exp_flag=args.exp
    )