# Swin Transformer Fine-tuned (Stage 1 only)
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

# Import from your project structure
from utils.utils import get_dataloaders, train_one_epoch, evaluate, calculate_metrics
from utils.model import my_swin_tiny_patch4_window7_224

# -------------------- Configuration --------------------
batch_size = 32
num_epochs_stage1 = 200
lr_stage1 = 1e-5
checkpoint_path = "saved_model/swin_best_model_v10001.pth"
save_dir = Path("Fine_tuned_swin_transformer")
save_dir.mkdir(parents=True, exist_ok=True)

# CSV log file
csv_path = save_dir / "finetune_metrics.csv"
with open(csv_path, 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["stage", "epoch", "train_loss", "val_loss",
                         "train_mae", "train_mse", "train_r2", "train_mare", "train_maxre",
                         "val_mae", "val_mse", "val_r2", "val_mare", "val_maxre", "time_s"])

# -------------------- Data --------------------
train_loader, val_loader, test_loader = get_dataloaders(
    image_dir='datasets/data/target1', label_dir='datasets/data/label', batch_size=batch_size, save_test_data=False
)

# -------------------- Model --------------------
model, device = my_swin_tiny_patch4_window7_224(num_classes=1)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
print(f"✅ Loaded pretrained model: {checkpoint_path}")

# -------- Helper: Save best model -------
def save_best(state_dict, path: Path):
    torch.save(state_dict, str(path))
    print(f"✅ Saved best model -> {path}")

# -------- Helper: Calculate relative errors (avoid division by 0) --------
def relative_errors(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred
    mask = (np.abs(y_true) > 1e-12)
    if np.any(mask):
        rel_errs = np.abs(errors[mask]) / np.abs(y_true[mask])
        max_re = float(np.max(rel_errs))
        mare = float(np.mean(rel_errs))
    else:
        max_re = float('nan')
        mare = float('nan')
    return errors, mare, max_re

# -------- Plotting functions --------
def plot_train_val_losses(stage1_train, stage1_val, save_path):
    plt.figure(figsize=(9,5))
    epochs_s1 = np.arange(1, len(stage1_train)+1)

    plt.plot(epochs_s1, stage1_train, label="Train Stage1", linewidth=2)
    plt.plot(epochs_s1, stage1_val, label="Val Stage1", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage1 Finetune Loss Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_test_results(y_true, y_pred, save_path):
    errors = y_true - y_pred
    plt.figure(figsize=(12,5))
    # left: true vs pred sorted by true value for clarity
    order = np.argsort(y_true)
    plt.subplot(1,2,1)
    plt.plot(np.array(y_true)[order], label="True", lw=2)
    plt.plot(np.array(y_pred)[order], label="Pred", lw=1)
    plt.title("Test set - True vs Pred (sorted by true)")
    plt.xlabel("Ordered Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.25)

    # right: residuals
    plt.subplot(1,2,2)
    plt.plot(errors, marker='.', linestyle='-', ms=4)
    plt.title("Test set Residuals (True - Pred)")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ---------------- Stage 1 Training ----------------
# Freeze patch_embed + layers 0~2
for name, param in model.named_parameters():
    if any(k in name for k in ("patch_embed", "layers.0", "layers.1", "layers.2")):
        param.requires_grad = False
    else:
        param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1)
print("🔒 Frozen patch_embed + layers.0~2")

best_val_loss = float('inf')
train_losses_s1, val_losses_s1 = [], []

for epoch in range(1, num_epochs_stage1+1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    t1 = time.time()

    train_losses_s1.append(train_loss)
    val_losses_s1.append(val_loss)

    train_mae, train_mse, train_r2, train_mare, train_max_re = calculate_metrics(model, train_loader, device)
    val_mae, val_mse, val_r2, val_mare, val_max_re = calculate_metrics(model, val_loader, device)

    # Write to CSV
    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["stage1", epoch, train_loss, val_loss,
                             train_mae, train_mse, train_r2, train_mare, train_max_re,
                             val_mae, val_mse, val_r2, val_mare, val_max_re, round(t1-t0,3)])

    # Print progress
    print(f"[Stage1] Epoch {epoch}/{num_epochs_stage1} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t1-t0:.2f}s")
    print(f"  Train  -> MSE {train_mse:.6e}  MAE {train_mae:.6e}  R2 {train_r2:.4f}  MARE {train_mare:.6e}  MaxRE {train_max_re:.6e}")
    print(f"  Val    -> MSE {val_mse:.6e}  MAE {val_mae:.6e}  R2 {val_r2:.4f}  MARE {val_mare:.6e}  MaxRE {val_max_re:.6e}")


    # Save best model
    best_file_s1 = save_dir / "Fine_tuned_swin_transformer_model.pth"
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_best(model.state_dict(), best_file_s1)


# ------------ Plot Stage1 Loss Curve ------------
plot_train_val_losses(train_losses_s1, val_losses_s1, save_dir / "Fine_tuned_swin_transformer_loss_curve.png")
print("Saved loss curve.")

# ------------ Test Set Evaluation and Plotting ------------
model.eval()
all_true, all_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X).detach().cpu()
        all_pred.append(pred)
        all_true.append(y.detach().cpu())

y_true = np.concatenate([a.numpy().ravel() for a in all_true])
y_pred = np.concatenate([a.numpy().ravel() for a in all_pred])
errors = y_true - y_pred
_, mare_test, max_re_test = relative_errors(y_true, y_pred)

# Print and save results
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_test = mean_squared_error(y_true, y_pred)
mae_test = mean_absolute_error(y_true, y_pred)
r2_test = r2_score(y_true, y_pred)
print(f"Test -> MSE: {mse_test:.6e}, MAE: {mae_test:.6e}, R2: {r2_test:.4f}, MARE: {mare_test:.6e}, MaxRE: {max_re_test:.6e}")

plot_test_results(y_true, y_pred, save_dir / "test_set_error_curve.png")
print("Saved test plots.")
