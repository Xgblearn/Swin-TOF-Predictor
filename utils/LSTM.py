import torch
import torch.nn as nn
import os, re, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchviz import make_dot
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
# class my_LSTM(nn.Module):
#     """
#     LSTM-based model
#     - Accepts input of shape (B, 1, L) (same as your current dataset output).
#     - Internally permutes to (B, L, feat) and feeds to nn.LSTM (batch_first=True).
#     - Uses last-layer hidden state (and both directions if bidirectional) as sequence summary.
#     - Then FC1(400) -> ReLU -> Dropout -> FC2(out_dim).
#     """
#     def __init__(self,
#                  input_channels=1,
#                  signal_len=3000,     # not strictly used but kept for API parity
#                  hidden_size=128,
#                  num_layers=2,
#                  bidirectional=True,
#                  out_dim=1,
#                  fc_hidden=400,
#                  dropout=0.2):
#         super().__init__()
#         self.input_channels = input_channels
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#
#         # LSTM: input_size = number of features per time-step = input_channels (1)
#         self.lstm = nn.LSTM(input_size=input_channels,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True,
#                             bidirectional=bidirectional,
#                             dropout=dropout if num_layers > 1 else 0.0)
#
#         # feature dim after extracting last-layer hidden state
#         feat_dim = hidden_size * self.num_directions
#
#         # FC head similar to CNN: FC1 (400) -> dropout -> FC2 (out_dim)
#         self.fc1 = nn.Linear(feat_dim, fc_hidden)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(fc_hidden, out_dim)
#
#         # optional small initialization
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
#
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
#
#     def forward(self, x):
#         """
#         x: (B, 1, L)
#         returns: (B,) if out_dim==1 else (B, out_dim)
#         """
#         # ensure float
#         if x.dtype != torch.float32:
#             x = x.float()
#
#         # permute to (B, L, input_channels)
#         # if input is (B, C, L) and C==1 this becomes (B, L, 1)
#         x = x.permute(0, 2, 1)
#
#         # Run LSTM
#         # out: (B, L, num_directions * hidden_size)
#         # h_n: (num_layers * num_directions, B, hidden_size)
#         out_seq, (h_n, c_n) = self.lstm(x)
#
#         # take last layer hidden states
#         # index of last layer (0-based) = num_layers-1
#         last_layer = self.num_layers - 1
#
#         if self.bidirectional:
#             # forward hidden = h_n[last_layer*2], backward = h_n[last_layer*2 + 1]
#             h_fwd = h_n[last_layer * 2]       # (B, hidden_size)
#             h_bwd = h_n[last_layer * 2 + 1]   # (B, hidden_size)
#             h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden_size*2)
#         else:
#             h = h_n[last_layer]  # (B, hidden_size)
#
#         # FC head
#         x = self.fc1(h)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)  # (B, out_dim)
#         if x.shape[1] == 1:
#             return x.squeeze(1)
#         return x
class PureLSTM_Improved(nn.Module):
    """
    纯 LSTM，但做了多项改进：
      - 双向 LSTM（可选）
      - LayerNorm 在序列特征上
      - AttentionPooling 代替仅取最后时刻
      - avg/max/attn 三路池化拼接（可选开/关）
      - FC head (400 -> out)
    保持输入格式 (B,1,L)
    """
    def __init__(self,
                 input_channels=1,
                 signal_len=3000,
                 hidden_size=192,
                 num_layers=2,
                 bidirectional=True,
                 use_avg_max_pool=True,
                 fc_hidden=400,
                 out_dim=1,
                 dropout=0.3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.use_avg_max_pool = use_avg_max_pool

        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)

        feat_dim = hidden_size * self.num_dirs

        # layernorm applied across feature dim for each time step
        self.layernorm = nn.LayerNorm(feat_dim)

        # small attention pooling
        self.attn_w = nn.Linear(feat_dim, 128)
        self.attn_v = nn.Linear(128, 1)

        # final head. If using avg+max+attn concat, input dim increases
        head_in = feat_dim * (3 if use_avg_max_pool else 1)
        self.fc1 = nn.Linear(head_in, fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, out_dim)

        # init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def attention_pool(self, seq):
        # seq: (B, L, D)
        e = torch.tanh(self.attn_w(seq))        # (B, L, 128)
        scores = self.attn_v(e).squeeze(-1)     # (B, L)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        pooled = (seq * alpha).sum(dim=1)       # (B, D)
        return pooled, alpha

    def forward(self, x):
        # x: (B,1,L) -> (B, L, 1)
        if x.dtype != torch.float32:
            x = x.float()
        x = x.permute(0, 2, 1)

        out_seq, (h_n, c_n) = self.lstm(x)   # out_seq: (B, L, D)
        out_seq = self.layernorm(out_seq)    # stabilize

        attn_pooled, attn_weights = self.attention_pool(out_seq)  # (B, D)

        if self.use_avg_max_pool:
            avg_pooled = out_seq.mean(dim=1)     # (B, D)
            max_pooled, _ = out_seq.max(dim=1)  # (B, D)
            feat = torch.cat([attn_pooled, avg_pooled, max_pooled], dim=1)  # (B, D*3)
        else:
            feat = attn_pooled  # (B, D)

        h = self.relu(self.fc1(feat))
        h = self.dropout(h)
        out = self.fc2(h)   # (B, out_dim)
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out
















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
def run_training(target_dir='data/target1', label_dir='data/label',
                 indices=range(1,2500), batch_size=32, epochs=30, lr=1e-4, signal_len=3000,
                 save_model='best_LSTM.pth', seed=42):
    # reproducibility
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    dataset = MyDataset(target_dir, label_dir, indices=indices, signal_len=signal_len)
    N = len(dataset)
    train_n = int(0.7 * N); val_n = int(0.2 * N); test_n = N - train_n - val_n
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n],
                                             generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = PureLSTM_Improved(input_channels=1,
                      signal_len=signal_len,
                      hidden_size=192,
                      num_layers=2,
                      bidirectional=True,
                      out_dim=1,
                      fc_hidden=400,
                      dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # fit label scaler on train set (so training is stable) -> we'll inverse transform preds later
    labels_train = []
    for _, lab in train_loader:
        labels_train.append(lab.numpy().ravel())
    labels_train = np.concatenate(labels_train).reshape(-1,1)
    scaler = MinMaxScaler((0,1)).fit(labels_train)

    best_val_loss = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        train_loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=100):
            x = x.to(device)  # (B,1,L)
            # scale labels
            y_np = y.numpy().reshape(-1,1)
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
                y_np = y.numpy().reshape(-1,1)
                y_scaled = torch.tensor(scaler.transform(y_np).ravel(), dtype=torch.float32).to(device)
                out = model(x)
                loss = loss_fn(out, y_scaled)
                val_loss_sum += loss.item() * x.size(0)
                val_preds.append(out.cpu().numpy().ravel())
                val_trues.append(y.numpy().ravel())
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        # inverse transform preds
        val_preds_orig = scaler.inverse_transform(val_preds.reshape(-1,1)).ravel()

        mae, mse, r2, mare, maxrel = calc_metrics(val_trues, val_preds_orig)
        print(f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} | val MAE={mae:.6f} MSE={mse:.6f} R2={r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model.state_dict(), 'scaler': scaler}, save_model)
            print("Saved best:", save_model)

    # ---- test ----
    chk = torch.load(save_model, map_location=device)
    model.load_state_dict(chk['model'])
    scaler = chk['scaler']

    # debug_shapes(model, signal_len=3000)

    preds = []; trues = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy().ravel())
            trues.append(y.numpy().ravel())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).ravel()

    mae, mse, r2, mare, maxrel = calc_metrics(trues, preds_orig)
    print("TEST:", f"MAE={mae:.6f}, MSE={mse:.6f}, R2={r2:.4f}, MARE={mare:.6f}, max_rel_err={maxrel:.6f}")

    plot_pred_vs_true(trues, preds_orig, 'Test: Pred vs True')

def debug_shapes(model, signal_len=3000):
    """
    Robust shape debugger for both CNN-like and LSTM-like models.
    - For CNNs (with conv1/conv2/pool) prints shapes after conv1/conv2/pool and flatten dim.
    - For LSTMs (with attribute 'lstm') prints out_seq, h_n, c_n shapes and forward output shape.
    - Otherwise tries a forward pass and prints output shape or error.
    """
    import torch
    # determine device from model parameters if possible
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    dummy = torch.randn(1, 1, signal_len, device=device)  # (B=1, C=1, L=signal_len)
    print(f"[debug_shapes] device={device}, dummy.shape={tuple(dummy.shape)}")

    with torch.no_grad():
        # CNN-like path
        if hasattr(model, 'conv1') and hasattr(model, 'conv2'):
            x = dummy
            try:
                x = model.conv1(x); print("after conv1:", tuple(x.shape))
                if hasattr(model, 'bn1'):
                    x = model.bn1(x)
                if hasattr(model, 'relu'):
                    x = model.relu(x)
                x = model.conv2(x); print("after conv2:", tuple(x.shape))
                if hasattr(model, 'bn2'):
                    x = model.bn2(x)
                if hasattr(model, 'pool'):
                    x = model.pool(x); print("after pool:", tuple(x.shape))
                print("flatten dim:", tuple(x.view(1, -1).shape))
            except Exception as e:
                print("Error while stepping through conv layers:", e)

        # LSTM-like path
        elif hasattr(model, 'lstm'):
            try:
                # model.forward expects (B,1,L) for our implementations, so call it too
                out_forward = None
                try:
                    out_forward = model(dummy)
                    print("model forward output shape:", None if out_forward is None else tuple(out_forward.shape))
                except Exception as fe:
                    print("Warning: model.forward(dummy) failed (may require different input). Error:", fe)

                # run LSTM module directly to inspect internal shapes
                lstm = model.lstm
                # permute dummy to (B, L, input_size) where input_size==1
                x_seq = dummy.permute(0, 2, 1)  # (B, L, 1)
                out_seq, (h_n, c_n) = lstm(x_seq)
                print("LSTM out_seq shape:", tuple(out_seq.shape))  # (B, L, num_directions*hidden)
                print("h_n shape:", tuple(h_n.shape), "c_n shape:", tuple(c_n.shape))
                # last layer index info
                num_layers = getattr(model, 'num_layers', getattr(lstm, 'num_layers', None))
                bidirectional = getattr(model, 'bidirectional', getattr(lstm, 'bidirectional', False))
                print("num_layers:", num_layers, "bidirectional:", bidirectional)
            except Exception as e:
                print("Error while debugging LSTM internals:", e)

        # fallback: try a forward pass
        else:
            try:
                out = model(dummy)
                print("model output shape (fallback):", tuple(out.shape))
            except Exception as e:
                print("Cannot auto-debug this model (no conv/lstm found and forward failed). Error:", e)

    # no return value; prints shapes for inspection


# 用法：
# debug_shapes(model, signal_len=3000)

# If run as script:
if __name__ == "__main__":
    # adjust paths if needed
    run_training(target_dir='data/target1', label_dir='data/label',
                 indices=range(1,2501), batch_size=32, epochs=50, lr=1e-3, signal_len=3000)





