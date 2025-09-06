import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

from utils.model import my_swin_tiny_patch4_window7_224  # Import your existing model definition

#############原来用的#################
# def load_labels(label_dir):
#     """加载真实值"""
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

    # Get all point*.txt files in directory
    file_list = sorted([f for f in os.listdir(label_dir) if f.startswith("point") and f.endswith(".txt")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

    for fname in file_list:
        label_path = os.path.join(label_dir, fname)
        with open(label_path, "r") as f:
            val = float(f.readline().strip())
        labels.append(val)
        point_names.append(fname.replace(".txt", ""))  # Remove suffix
    return np.array(labels), point_names


def main():
    # Paths
    model_path = "saved_model/stage1_best_2.pth"
    img_dir = "datasets/Real_data/3.983mm/GASF"
    label_dir = "datasets/Real_data/3.983mm/label"
    result_dir = "results_Swin_transformer"

    os.makedirs(result_dir, exist_ok=True)

    # Device & model
    model, device = my_swin_tiny_patch4_window7_224(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    labels, point_names = load_labels(label_dir)
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Read images and predict
    preds = []
    with torch.no_grad():
        for name in point_names:
            img_path = os.path.join(img_dir, f"{name}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            pred = model(img)
            pred = pred.squeeze().item()
            preds.append(pred)

    preds = np.array(preds)
    labels, point_names = load_labels(label_dir)
    # Scale as required
    preds_scaled = preds * 5900 / 2000
    labels_scaled = labels * 5900 / 2000

    # Error metrics
    errors = preds_scaled - labels_scaled
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mare = np.mean(np.abs(errors) / (np.abs(labels_scaled) + 1e-8))  # Prevent division by zero
    avg_val = np.mean(preds_scaled)
    max_dev = np.max(preds_scaled) - avg_val
    min_dev = np.min(preds_scaled) - avg_val

    # Save results to DataFrame
    df = pd.DataFrame({
        "Point": point_names,
        "True_Value": labels_scaled,
        "Predicted_Value": preds_scaled,
        "Error": errors
    })
    # Append error metrics
    metrics = {
        "Point": ["MAE", "MSE", "MARE", "Average", "Max_Deviation", "Min_Deviation"],
        "True_Value": ["-"] * 6,
        "Predicted_Value": ["-"] * 6,
        "Error": [mae, mse, mare, avg_val, max_dev, min_dev]
    }
    df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)

    # Save to CSV
    save_path = os.path.join(result_dir, "3.983mm_swim_transformer_test_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # Console output
    print("Results saved to:", save_path)
    print(df)


if __name__ == "__main__":
    main()
