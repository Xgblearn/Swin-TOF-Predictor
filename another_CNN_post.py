#!/usr/bin/env python3
# cnn_pre_ultrasonic.py
# Training + predict pipeline using PaperCNN_Ultrasonic (1D CNN for ultrasonic signals)

import os
import re
import json
import random
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Seed utility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# Dataset for txt 1D signals
# -------------------------
class MyDataset(Dataset):
    def __init__(self, target_dir: str, label_dir: str, indices: Optional[List[int]] = None, signal_len: int = 3000):
        self.target_dir = target_dir
        self.label_dir = label_dir
        self.signal_len = signal_len

        if not os.path.isdir(target_dir):
            raise RuntimeError(f"Target dir not found: {target_dir}")
        if not os.path.isdir(label_dir):
            raise RuntimeError(f"Label dir not found: {label_dir}")

        target_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
        label_files = set(f for f in os.listdir(label_dir) if f.endswith('.txt'))

        file_map = {}
        for fn in target_files:
            base = os.path.splitext(fn)[0]
            m = re.fullmatch(r'0*(\d+)', base)
            if not m:
                continue
            idx = int(m.group(1))
            if base + '.txt' in label_files:
                file_map[idx] = fn

        if indices is None:
            indices = sorted(file_map.keys())
        else:
            indices = [i for i in indices if i in file_map]

        if len(indices) == 0:
            raise RuntimeError("No samples found. Check target_dir/label_dir and indexing.")

        self.indices = sorted(indices)
        self._file_map = file_map
        self.targets = [os.path.join(target_dir, file_map[i]) for i in self.indices]
        self.labels = [os.path.join(label_dir, os.path.splitext(file_map[i])[0] + '.txt') for i in self.indices]

        if len(self.targets) == 0:
            raise RuntimeError("No valid samples found.")

    def __len__(self):
        return len(self.targets)


    def _read_target_signal(self, path: str):
        try:
            arr = np.loadtxt(path, usecols=(0,))
        except Exception:
            arr = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split('[,\\s]+', line)
                    try:
                        arr.append(float(parts[0]))
                    except:
                        continue
            arr = np.array(arr, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32).ravel()
        L = len(arr)
        if L == self.signal_len:
            sig = arr
        elif L > self.signal_len:
            start = (L - self.signal_len) // 2
            sig = arr[start:start + self.signal_len]
        else:
            sig = np.zeros(self.signal_len, dtype=np.float32)
            sig[:L] = arr

        m = np.max(np.abs(sig)) if sig.size > 0 else 0.0
        if m > 0:
            sig = sig / m
        return sig.astype(np.float32)

    def _read_label(self, path: str):
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = re.split('[,\\s]+', s)
                try:
                    return float(parts[0])
                except:
                    continue
        raise ValueError(f"Cannot parse label in {path}")

    def __getitem__(self, idx: int):
        target_path = self.targets[idx]
        label_path = self.labels[idx]

        sig = self._read_target_signal(target_path)
        label = self._read_label(label_path)

        sig_t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, L)
        lab_t = torch.tensor(label, dtype=torch.float32)
        return sig_t, lab_t

# -------------------------
# Robust split conversion (string/int/filename -> positions)
# -------------------------
def convert_split_to_positions(fixed_split: Dict[str, List[int]], dataset: MyDataset) -> Dict[str, List[int]]:
    if fixed_split is None:
        raise ValueError("fixed_split is None")
    split_copy = {k: list(v) for k, v in fixed_split.items()}

    # normalize possible string numeric items to int
    for k, lst in split_copy.items():
        new_lst = []
        for item in lst:
            if isinstance(item, str):
                s = item.strip()
                s_noext = os.path.splitext(s)[0]
                if re.fullmatch(r'0*\d+', s_noext):
                    new_lst.append(int(s_noext))
                    continue
                new_lst.append(s)
            else:
                new_lst.append(item)
        split_copy[k] = new_lst

    all_vals = []
    for v in split_copy.values():
        all_vals.extend(v)
    if len(all_vals) == 0:
        raise ValueError("fixed_split provided but contains empty lists")

    # Case A: ints and subset of dataset.indices (file-id)
    if all(isinstance(x, int) for x in all_vals) and set(all_vals).issubset(set(dataset.indices)):
        id2pos = {fid: pos for pos, fid in enumerate(dataset.indices)}
        return {k: [id2pos[int(x)] for x in v] for k, v in split_copy.items()}

    # Case B: ints and within 0..len(dataset)-1 (positions)
    if all(isinstance(x, int) and 0 <= x < len(dataset) for x in all_vals):
        return split_copy

    # Case C: strings -> try matching basenames
    basename2pos = {os.path.basename(dataset.targets[pos]): pos for pos in range(len(dataset))}
    basename_noext2pos = {os.path.splitext(k)[0].lstrip("0"): v for k, v in basename2pos.items()}

    def map_str_item(s):
        b = os.path.basename(str(s))
        if b in basename2pos:
            return basename2pos[b]
        b_noext = os.path.splitext(b)[0]
        # direct match without extension
        for kk, vv in basename2pos.items():
            if os.path.splitext(kk)[0] == b_noext:
                return vv
        key = b_noext.lstrip("0")
        if key in basename_noext2pos:
            return basename_noext2pos[key]
        # try numeric conversion
        try:
            mm = re.fullmatch(r'0*(\d+)', b_noext)
            if mm:
                n = int(mm.group(1))
                id2pos = {fid: pos for pos, fid in enumerate(dataset.indices)}
                if n in id2pos:
                    return id2pos[n]
        except:
            pass
        return None

    pos_split = {}
    for k, lst in split_copy.items():
        tmp = []
        for item in lst:
            if isinstance(item, int):
                if 0 <= item < len(dataset):
                    tmp.append(int(item))
                else:
                    id2pos = {fid: pos for pos, fid in enumerate(dataset.indices)}
                    if item in id2pos:
                        tmp.append(id2pos[item])
                    else:
                        print(f"Warning: int index {item} not found; skipping.")
            else:
                mapped = map_str_item(item)
                if mapped is None:
                    print(f"Warning: cannot map split item '{item}' -> skipping.")
                else:
                    tmp.append(mapped)
        pos_split[k] = tmp

    for k in ('train','val','test'):
        if k not in pos_split or len(pos_split[k]) == 0:
            raise RuntimeError(f"After mapping, split '{k}' is empty or missing.")
        if any((x < 0 or x >= len(dataset)) for x in pos_split[k]):
            raise RuntimeError(f"After mapping, fixed_split['{k}'] contains out-of-range positions.")

    return pos_split

# -------------------------
# Your new model: PaperCNN_Ultrasonic
# -------------------------
class PaperCNN_Ultrasonic(nn.Module):
    """
    3 conv layers, each 64 channels, kernel k=71, BN+ReLU, dropout after last conv,
    no pooling, flatten -> fc -> out.
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
        if x.dim() == 2 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

# -------------------------
# Metrics & plotting
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

def plot_pred_vs_true(true, pred, save_path=None, title='Predicted vs True'):
    plt.figure(figsize=(7,6))
    plt.scatter(true, pred, alpha=0.5)
    mn = min(min(true), min(pred)); mx = max(max(true), max(pred))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel('True'); plt.ylabel('Predicted'); plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

# -------------------------
# Save / load split helpers
# -------------------------
def save_split_indices(dataset: MyDataset, save_dir: str = "saved_splits", seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)
    N = len(dataset)
    train_n = int(0.7 * N); val_n = int(0.2 * N); test_n = N - train_n - val_n
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n], generator=gen)

    pos_split = {
        "train": list(train_ds.indices),
        "val": list(val_ds.indices),
        "test": list(test_ds.indices)
    }
    fileid_split = {
        "train": [dataset.indices[pos] for pos in pos_split["train"]],
        "val":   [dataset.indices[pos] for pos in pos_split["val"]],
        "test":  [dataset.indices[pos] for pos in pos_split["test"]]
    }

    with open(os.path.join(save_dir, "split_positions.json"), 'w') as f:
        json.dump(pos_split, f, indent=2)
    with open(os.path.join(save_dir, "split_file_ids.json"), 'w') as f:
        json.dump(fileid_split, f, indent=2)
    return pos_split

def load_split_indices(path: str):
    with open(path, 'r') as f:
        return json.load(f)

# -------------------------
# Dataloader helper
# -------------------------
def get_dataloaders(
    target_dir='data/target2',
    label_dir='data/label',
    batch_size=32,
    fixed_split: Optional[Dict[str, List[int]]] = None,
    save_test_data=False,
    save_target_out_dir='saved_data/test/targets',
    save_label_out_dir='saved_data/test/labels',
    num_workers: int = 0
):
    dataset = MyDataset(target_dir, label_dir)

    if fixed_split is not None:
        fixed_split = convert_split_to_positions(fixed_split, dataset)
        for k in ('train','val','test'):
            if k not in fixed_split:
                raise RuntimeError(f"fixed_split missing key: {k}")
            if any((x < 0 or x >= len(dataset)) for x in fixed_split[k]):
                raise RuntimeError(f"fixed_split['{k}'] has out-of-range positions.")
        train_dataset = Subset(dataset, fixed_split['train'])
        val_dataset   = Subset(dataset, fixed_split['val'])
        test_dataset  = Subset(dataset, fixed_split['test'])
    else:
        n = len(dataset)
        train_dataset = Subset(dataset, list(range(int(0.7*n))))
        val_dataset   = Subset(dataset, list(range(int(0.7*n), int(0.9*n))))
        test_dataset  = Subset(dataset, list(range(int(0.9*n), n)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if save_test_data:
        os.makedirs(save_target_out_dir, exist_ok=True)
        os.makedirs(save_label_out_dir, exist_ok=True)
        for pos in test_dataset.indices:
            target_path = dataset.targets[pos]
            label_path = dataset.labels[pos]
            shutil.copy(target_path, os.path.join(save_target_out_dir, os.path.basename(target_path)))
            shutil.copy(label_path, os.path.join(save_label_out_dir, os.path.basename(label_path)))

    return train_loader, val_loader, test_loader

# -------------------------
# Train / finetune (uses PaperCNN_Ultrasonic)
# -------------------------
def train_or_finetune(
    target_dir='data/target2',
    label_dir='data/label',
    indices=range(1,1200),
    batch_size=32,
    epochs=50,
    lr=1e-4,
    signal_len=3000,
    save_model='best_cnn_ultra.pth',
    split_json='saved_splits/split_positions.json',
    seed=42,
    pretrained_path: Optional[str] = None,
    finetune_lr: Optional[float] = 1e-5,
    early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 1e-6,
    restore_best: bool = True,
    num_workers: int = 0
):
    set_seed(seed)
    dataset = MyDataset(target_dir, label_dir, indices=list(indices), signal_len=signal_len)

    # ensure split exists
    if not os.path.exists(split_json):
        print(f"Split file {split_json} not found -> creating saved_splits/ ...")
        fixed_split = save_split_indices(dataset, save_dir="saved_splits", seed=seed)
    else:
        fixed_split = load_split_indices(split_json)

    # convert
    try:
        fixed_split = convert_split_to_positions(fixed_split, dataset)
    except Exception as e:
        raise RuntimeError(f"Failed to convert split indices to positions: {e}")

    for k in ('train','val','test'):
        if k not in fixed_split:
            raise RuntimeError(f"split JSON missing key '{k}'")
        if any((x < 0 or x >= len(dataset)) for x in fixed_split[k]):
            raise RuntimeError(f"After conversion, fixed_split['{k}'] contains out-of-range positions.")

    train_ds = Subset(dataset, fixed_split['train'])
    val_ds   = Subset(dataset, fixed_split['val'])
    test_ds  = Subset(dataset, fixed_split['test'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = PaperCNN_Ultrasonic(input_channels=1, signal_len=signal_len).to(device)

    # load pretrained if exists
    if pretrained_path is not None and os.path.exists(pretrained_path):
        chk = torch.load(pretrained_path, map_location=device)
        if isinstance(chk, dict) and 'model' in chk:
            model.load_state_dict(chk['model'])
            print("Loaded pretrained model from", pretrained_path)
        else:
            model.load_state_dict(chk)
            print("Loaded state_dict from", pretrained_path)
        current_lr = finetune_lr if finetune_lr is not None else lr
    else:
        current_lr = lr

    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    loss_fn = nn.MSELoss()

    # fit label scaler on train set
    labels_train = []
    for _, lab in train_loader:
        labels_train.append(lab.detach().cpu().numpy().ravel())
    labels_train = np.concatenate(labels_train).reshape(-1,1)
    scaler = MinMaxScaler((0,1)).fit(labels_train)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        train_loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=100):
            x = x.to(device)
            y_np = y.detach().cpu().numpy().reshape(-1,1)
            y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y_scaled)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_preds = []; val_trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_np = y.detach().cpu().numpy().reshape(-1,1)
                y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)
                out = model(x)
                loss = loss_fn(out, y_scaled)
                val_loss_sum += loss.item() * x.size(0)
                val_preds.append(out.cpu().numpy().ravel())
                val_trues.append(y.detach().cpu().numpy().ravel())
        val_loss = val_loss_sum / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')

        if len(val_preds) > 0:
            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            val_preds_orig = scaler.inverse_transform(val_preds.reshape(-1,1)).ravel()
            mae, mse, r2, mare, maxrel = calc_metrics(val_trues, val_preds_orig)
        else:
            mae=mse=r2=mare=maxrel=float('nan')

        print(f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} | val MAE={mae:.6f} MSE={mse:.6f} R2={r2:.4f}")

        improved = False
        if not np.isnan(val_loss) and val_loss + min_delta < best_val_loss:
            improved = True

        if improved:
            best_val_loss = val_loss
            best_epoch = ep
            epochs_no_improve = 0
            # save best model checkpoint
            torch.save({'model': model.state_dict(), 'scaler': scaler}, save_model)
            print(f"Saved best model (epoch {ep}) -> {save_model}")
        else:
            epochs_no_improve += 1

        # if improved:
        #     best_val_loss = val_loss
        #     best_epoch = ep
        #     epochs_no_improve = 0
        #     # save best model checkpoint (include optimizer & epoch)
        #     os.makedirs(os.path.dirname(save_model) or ".", exist_ok=True)
        #     torch.save({
        #         'model': model.state_dict(),
        #         'scaler': scaler,
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': ep
        #     }, save_model)
        #     print(f"Saved best model (epoch {ep}) -> {os.path.abspath(save_model)}")
        # else:
        #     epochs_no_improve += 1

        if early_stopping and len(val_loader.dataset) > 0:
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {epochs_no_improve} epochs. Best epoch: {best_epoch}, best_val_loss: {best_val_loss:.6e}")
                break

    # # optionally save final epoch weights separately
    # final_path = save_model.replace('.pth', '_final.pth')
    # try:
    #     torch.save({'model': model.state_dict(), 'scaler': scaler, 'epoch': ep}, final_path)
    # except Exception:
    #     pass

    # restore best if requested
    if restore_best and os.path.exists(save_model):
        chk = torch.load(save_model, map_location=device)
        if 'model' in chk:
            model.load_state_dict(chk['model'])
            print("Restored best model from", save_model)
        if 'scaler' in chk:
            scaler = chk['scaler']

    return {
        "save_model": save_model,
        "split_json": split_json,
        "test_indices": fixed_split['test'],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_epoch": ep,
        "epochs_trained": ep
    }

# -------------------------
# Predict & save results (works with checkpoint saved above)
# -------------------------
def predict_and_save(
    saved_model_path: str,
    target_dir='data/target2',
    label_dir='data/label',
    out_dir='pred_results',
    batch_size=32,
    sample_limit: Optional[int]=None,
    split_json='saved_splits/split_positions.json',
    device: Optional[torch.device]=None,
    save_plots: bool=True,
    num_workers: int = 0
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chk = torch.load(saved_model_path, map_location=device)
    if 'model' not in chk or 'scaler' not in chk:
        raise RuntimeError("Saved model file must contain keys: 'model' and 'scaler'")
    scaler = chk['scaler']

    model = PaperCNN_Ultrasonic(input_channels=1).to(device)
    model.load_state_dict(chk['model'])
    model.eval()

    dataset = MyDataset(target_dir, label_dir)
    fixed_split = load_split_indices(split_json)
    fixed_split = convert_split_to_positions(fixed_split, dataset)

    test_idx_list = fixed_split['test']
    if sample_limit is not None:
        test_idx_list = test_idx_list[:sample_limit]

    pred_subset = Subset(dataset, test_idx_list)
    pred_loader = DataLoader(pred_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_names = [os.path.basename(dataset.targets[pos]) for pos in test_idx_list]

    all_preds = []
    all_trues = []
    all_names = []
    with torch.no_grad():
        idx_counter = 0
        for x, y in tqdm(pred_loader, desc="Predicting", ncols=100):
            x = x.to(device)
            out = model(x)
            preds = out.cpu().numpy().ravel()
            trues = y.detach().cpu().numpy().ravel()

            preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).ravel()

            all_preds.extend(list(preds_orig))
            all_trues.extend(list(trues))
            for i in range(len(trues)):
                all_names.append(sample_names[idx_counter + i])
            idx_counter += len(trues)

    y_true = np.array(all_trues, dtype=float)
    y_pred = np.array(all_preds, dtype=float)

    import csv
    pred_csv = out_dir / "predictions.csv"
    with open(pred_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "base_path", "y_true", "y_pred", "abs_error", "rel_error"])
        for name, yt, yp in zip(all_names, y_true, y_pred):
            err = float(abs(yt - yp))
            rel = abs((yp - yt) / yt) if yt != 0 else float('nan')
            writer.writerow([name, name, float(yt), float(yp), err, rel])

    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        mae, mse, r2, mare, maxrel = calc_metrics(y_true, y_pred)
        writer.writerow(["mse", mse])
        writer.writerow(["mae", mae])
        writer.writerow(["r2", r2])
        writer.writerow(["mare", mare])
        writer.writerow(["max_relative_error", maxrel])
        writer.writerow(["n_samples", int(len(y_true))])

    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)

    if save_plots:
        plot_pred_vs_true(y_true, y_pred, save_path=str(out_dir / "pred_vs_true.png"), title="Test Pred vs True")
        errors = y_true - y_pred
        plt.figure(figsize=(8,4))
        plt.plot(errors, marker='.', linestyle='-', ms=4)
        plt.title("Residuals (True - Pred)")
        plt.xlabel("Sample Index")
        plt.ylabel("Error")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(str(out_dir / "residuals.png"), dpi=150)
        plt.close()

    print("Saved predictions:", pred_csv)
    print("Saved metrics:", metrics_csv)
    print(f"Metrics: MAE={mae:.6e}, MSE={mse:.6e}, R2={r2:.4f}, MARE={mare:.6e}, MaxRE={maxrel:.6e}")

    return {
        "pred_csv": str(pred_csv),
        "metrics_csv": str(metrics_csv),
        "y_true_npy": str(out_dir / "y_true.npy"),
        "y_pred_npy": str(out_dir / "y_pred.npy"),
        "mae": mae, "mse": mse, "r2": r2, "mare": mare, "max_relative_error": maxrel,
        "n_samples": int(len(y_true))
    }

# -------------------------
# Example main
# -------------------------
if __name__ == "__main__":
    TARGET_DIR = 'datasets/data/target2'
    LABEL_DIR = 'datasets/data/label'
    INDICES = range(1, 1200)
    SPLIT_JSON_PATH = "saved_splits/split_positions.json"
    SAVE_MODEL = 'saved_model/post_another_best_cnn.pth'
    PRETRAINED = 'saved_model/best_another_cnn.pth'
    FINETUNE_LR = 1e-5
    BATCH_SIZE = 32
    EPOCHS = 200
    SIGNAL_LEN = 3000
    NUM_WORKERS = 0


    res = train_or_finetune(
        target_dir=TARGET_DIR,
        label_dir=LABEL_DIR,
        indices=INDICES,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=1e-4,
        signal_len=SIGNAL_LEN,
        save_model=SAVE_MODEL,
        split_json=SPLIT_JSON_PATH,
        seed=42,
        pretrained_path=PRETRAINED,
        finetune_lr=FINETUNE_LR,
        early_stopping=True,
        patience=20,
        restore_best=True,
        num_workers=NUM_WORKERS
    )
    print("Train finished. Summary:", res)

    pred_out = predict_and_save(
        saved_model_path=SAVE_MODEL,
        target_dir=TARGET_DIR,
        label_dir=LABEL_DIR,
        out_dir='pred_results',
        batch_size=BATCH_SIZE,
        sample_limit=100,
        split_json=SPLIT_JSON_PATH,
        save_plots=True,
        num_workers=NUM_WORKERS
    )
    print("Predict finished. Summary:", pred_out)
