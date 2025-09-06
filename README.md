# Swin Transformer for Ultrasound TOF Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A comprehensive deep learning project for Ultrasound Time of Flight (TOF) prediction with Swin Transformer as the primary method**

[Quick Start](#-quick-start) • [Environment](#️-environment--installation) • [Dataset Preparation](#-dataset-preparation) • [Training](#️-training) • [Testing](#-testing--evaluation) • [Fine-tuning](#-fine-tuning) • [Real Data Testing](#-real-dataset-testing) • [Other DATA Testing](#-other-data-testing)

</div>

---

## ✨ Introduction

Abstract:Accurate estimation of Time of Flight Difference (TOFD) is crucial for achieving high-precision material thickness measurement and defect localization. This paper proposes a method for measuring ultrasonic time of flight based on Gramian Angular Field (GAF) image encoding and deep learning network models. The method first applies wavelet denoising to ultrasonic echo signals and extracts envelope signals. Subsequently, the GAF transforms the signals into two-dimensional images to enhance the expressive capability of their temporal and structural features. Deep learning models are then fine-tuned through transfer learning on real datasets, enabling accurate TOFD estimation. Simulation results indicate that the proposed TOFD estimation method achieves further improvements in accuracy compared with other methods. Ultrasonic thickness measurement experiments on stainless steel samples with different thicknesses validate the accuracy and robustness of the proposed method in practical applications.

### 🎯 Primary Method: Swin Transformer
- **🏗️ Modern Architecture**: Hierarchical Vision Transformer with shifted windows
- **🎨 Signal-to-Image Conversion**: Transform 1D ultrasound signals to 2D GASF images
- **⚡ Efficient Training**: Early stopping, learning rate scheduling, mixed precision
- **📈 Superior Performance**: Outperforms traditional methods in accuracy and generalization

### 🔬 Comparison Methods (5 Additional Approaches)
1. **CNN (1D Convolutional Neural Network)**: Traditional deep learning baseline
2. **Another CNN (Improved CNN)**: Enhanced CNN with better architecture
3. **ODES + AIC**: Traditional signal processing method using Optimal Detection of Echo Signals
4. **ADE + LM**: Adaptive Differential Evolution with Levenberg-Marquardt optimization
5. **Peak Method**: Classical peak detection approach

### 🌟 Key Features
- **📊 Comprehensive Evaluation**: 15dB noise testing, real dataset validation
- **🔄 Fine-tuning Capability**: Two-stage fine-tuning for improved performance
- **🌐 Generalization Testing**: Winston paper data validation for cross-domain performance
- **📈 Rich Metrics**: MAE, MSE, R², MARE, inference latency analysis
- **🔧 Flexible Configuration**: CLI flags and comprehensive config management

## 📁 Project Structure

```
swin_transformer_paper/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Dependencies
├── 📄 INSTALL.md                   # Installation guide
│
├── 🎯 Primary Method (Swin Transformer)
│   ├── swin_transformer_model.py   # Main Swin training script
│   ├── Swin_transformer_Fine_tuned.py  # Fine-tuning script
│   ├── test_swin_transformer.py    # Swin testing
│   ├── test_Wston_data.py          # Winston data testing
│   └── utils/
│       ├── swin_transformer_modules.py  # Swin implementation
│       ├── Dataset.py              # Dataset class
│       └── model.py                # Model definitions
│
├── 🔬 Comparison Methods (5 Additional Approaches)
│   ├── CNN Methods
│   │   ├── CNN_pre.py              # Basic CNN model
│   │   ├── pre_another_CNN.py      # Improved CNN
│   │   ├── test_CNN.py             # CNN testing
│   │   └── test_another_CNN.py     # Improved CNN testing
│   │
│   ├── Traditional Signal Processing
│   │   ├── peak_method.py          # Peak detection method
│   │   ├── ADE_and_LM.py           # ADE+LM method
│   │   ├── ODESandAIC.py           # ODES+AIC method
│   │   └── VSS_method.py           # VSS method
│   │
│   └── Additional Models
│       ├── LSTM.py                 # LSTM model
│       └── BNN_model.py            # Bayesian Neural Network
│
├── 🧪 Testing & Evaluation Scripts
│   ├── nine_point_test_swim_transformer.py    # 9-point Swin testing
│   ├── nine_point_test_another_CNN.py         # 9-point CNN testing
│   ├── nine_point_ODES_AIC.py                 # 9-point ODES+AIC testing
│   ├── nine_point_ADE_LM.py                   # 9-point ADE+LM testing
│   └── nine_point_peak_method.py              # 9-point peak method testing
│
├── 📊 Data Organization
│   ├── datasets/
│   │   ├── data_signal/            # Raw signal data
│   │   ├── data_test/              # 15dB test data
│   │   │   └── classified/         # Noise level classification
│   │   └── data_to_SW/             # Swin training data
│   ├── saved_data/                 # Saved test data
│   └── saved_model/                # Trained models
│
├── 📈 Results & Logs
│   ├── logs/                       # Training logs and visualizations
│   ├── training_SW_result/         # Swin training outputs
│   ├── Fine_tuned_swin_transformer/ # Fine-tuning results
│   └── plt_*_results/              # Method comparison visualizations
│
└── 🛠️ Utilities
    ├── test_plt.py                 # Visualization utilities
    ├── test_sw_plt.py              # Swin-specific visualization
    └── utils/tool.py               # General utilities
```

---


## 🚀 Quick Start

Follow this complete workflow to reproduce our results:

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd swin_transformer_paper

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
# Prepare your ultrasound data following the structure:
# datasets/data_to_SW/target/     # GASF images (.jpg)
# datasets/data_to_SW/label/      # Labels (.txt)
```

### 3. Swin Transformer Training
```bash
# Train the primary Swin Transformer model
python swin_transformer_model.py --epochs 100 --flag v100
```

### 4. 15dB Noise Testing
```bash
# Test on 15dB noise data
python test_swin_transformer.py
```

### 5. Fine-tuning
```bash
# Fine-tune the pre-trained model
python Swin_transformer_Fine_tuned.py
```

### 6. Real Dataset Testing
```bash
# Test on real dataset (9-point testing)
python nine_point_test_swim_transformer.py
```

### 7. Winston Paper Data Testing
```bash
# Test generalization on Winston's paper data
python test_Wston_data.py
```

---

## ⚙️ Environment & Installation

#### Option A: requirements.txt
```bash
# Create virtual environment
python -m venv swin_env
source swin_env/bin/activate  # Linux/Mac
# Or
swin_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option B: conda
```bash
conda create -n swin_transformer python=3.9
conda activate swin_transformer
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Original Dataset Preparation

The project requires careful preparation of ultrasound signal data following this structure:

```
datasets/
├── data_signal/              # Raw ultrasound signals
│   ├── target/               # Signal files (.txt)
│   └── label/                # Ground truth labels (.txt)
├── data_test/                # 15dB noise test data
│   ├── classified/           # Noise level classification
│   │   ├── -5DB/            # -5dB noise level
│   │   ├── 0DB/             # 0dB noise level
│   │   ├── 5DB/             # 5dB noise level
│   │   ├── 10DB/            # 10dB noise level
│   │   └── 15DB/            # 15dB noise level
│   │       ├── target_ASDF/ # GASF images (.jpg)
│   │       └── label/       # Labels (.txt)
└── data_to_SW/               # Swin Transformer training data
    ├── target/               # GASF images (.jpg)
    └── label/                # Labels (.txt)
```

### Data Preprocessing Pipeline

1. **Signal-to-Image Conversion**: Transform 1D ultrasound signals to 2D GASF (Gramian Angular Summation Field) images
2. **Noise Level Classification**: Organize data by different SNR levels (-5dB to 15dB)
3. **Data Augmentation**: Random crop, rotation, and flip for robust training
4. **Normalization**: Automatic standardization for optimal model performance

### Dataset Usage Example

```python
from utils.Dataset import MyDataset

# Build Swin Transformer dataset
dataset = MyDataset(
    image_dir='datasets/data_to_SW/target',
    label_dir='datasets/data_to_SW/label',
    indices=range(1, 10001)
)

# Create DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Data Requirements

- **Signal Format**: 1D ultrasound signals as .txt files
- **Label Format**: Ground truth TOF values as .txt files
- **Image Format**: 224x224 RGB GASF images as .jpg files
- **File Naming**: Consistent naming convention for signal-label pairing

---

## 🏋️ Training

### Primary Method: Swin Transformer Training

The Swin Transformer serves as our primary method for TOF prediction. It leverages the hierarchical Vision Transformer architecture with shifted windows to process GASF-converted ultrasound images.

#### Basic Training

```bash
# Default Swin Transformer training
python swin_transformer_model.py

# Custom parameters for better performance
python swin_transformer_model.py \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4 \
    --flag "swin_experiment_v1"
```

#### Advanced Training Options

```bash
# Enable early stopping for better convergence
python swin_transformer_model.py --early_stop --patience 20

# Specify custom data paths
python swin_transformer_model.py \
    --image_dir "datasets/data_to_SW/target" \
    --label_dir "datasets/data_to_SW/label"

# Save test split for evaluation
python swin_transformer_model.py --save_test_data
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--lr` | 1e-4 | Learning rate |
| `--early_stop` | False | Enable early stopping |
| `--patience` | 20 | Patience for early stopping |
| `--flag` | "v10001" | Experiment identifier |

#### Training Outputs

During Swin Transformer training, the following files are generated:
- **Loss curves**: `training_SW_result/loss_curves_*.png`
- **Training parameters**: `training_SW_result/training_params_*.json`
- **Best model**: `saved_model/swin_best_model_*.pth`
- **Test results**: `training_SW_result/test_results_*.json`

### Comparison Methods Training

For comprehensive evaluation, we also provide training scripts for 5 additional methods:

#### CNN Methods
```bash
# Basic CNN training
python CNN_pre.py

# Improved CNN training
python post_another_CNN.py
```

#### Traditional Methods
```bash
# These methods don't require training as they are rule-based
# They can be directly tested using the test scripts
```

---

## 🧪 Testing & Evaluation

### 15dB Noise Testing

After training, the primary evaluation is performed on 15dB noise data to test model robustness:

```bash
# Test Swin Transformer on 15dB noise data
python test_swin_transformer.py

# Test other methods for comparison
python test_CNN.py
python test_another_CNN.py
```

#### 15dB Test Results

The 15dB testing evaluates model performance under challenging noise conditions:

- **Swin Transformer**: Demonstrates superior robustness to noise
- **CNN Methods**: Show good performance but lower than Swin Transformer
- **Traditional Methods**: Baseline performance for comparison

#### Test Outputs

Each test generates comprehensive evaluation results:
- **Performance metrics**: MAE, MSE, R², MARE
- **Visualizations**: Error histograms, prediction vs. true plots
- **Detailed reports**: JSON files with complete statistics

### Performance Metrics

All methods are evaluated using consistent metrics:

| Metric | Description | Importance |
|--------|-------------|------------|
| **MAE** | Mean Absolute Error | Primary accuracy measure |
| **MSE** | Mean Squared Error | Penalizes large errors |
| **R²** | R-squared | Model fit quality |
| **MARE** | Mean Absolute Relative Error | Relative accuracy |
| **Latency** | Inference time | Real-time applicability |

---

## 🔧 Fine-tuning

### Overview

Fine-tuning allows you to adapt a pre-trained Swin Transformer model to your specific dataset with improved performance. The fine-tuning process uses a two-stage approach with layer freezing for better convergence.

### Fine-tuning Process

#### Stage 1: Partial Fine-tuning
- **Frozen layers**: `patch_embed` + `layers.0~2` (early feature extraction layers)
- **Trainable layers**: `layers.3` + `head` (deeper layers and classification head)
- **Learning rate**: 1e-5 (lower than initial training)
- **Purpose**: Adapt high-level features while preserving low-level representations

#### Configuration

```bash
# Basic fine-tuning
python Swin_transformer_Fine_tuned.py

# The script automatically:
# - Loads pre-trained model from saved_model/
# - Freezes early layers (patch_embed, layers.0-2)
# - Trains for 200 epochs with learning rate 1e-5
# - Saves best model as Fine_tuned_swin_transformer_model.pth
```

### Fine-tuning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_epochs_stage1` | 200 | Training epochs for fine-tuning |
| `lr_stage1` | 1e-5 | Learning rate (lower than initial training) |
| `batch_size` | 32 | Batch size for fine-tuning |
| `checkpoint_path` | `saved_model/swin_best_model_*.pth` | Pre-trained model path |

### Data Requirements

Fine-tuning uses the same data structure as training:

```
datasets/
├── data/
│   ├── target1/          # GASF images (.jpg)
│   └── label/            # Labels (.txt)
```
### Fine-tuning Outputs

During fine-tuning, the following files are generated:

- **Best model**: `Fine_tuned_swin_transformer/Fine_tuned_swin_transformer_model.pth`
- **Loss curves**: `Fine_tuned_swin_transformer/Fine_tuned_swin_transformer_loss_curve.png`
- **Training metrics**: `Fine_tuned_swin_transformer/finetune_metrics.csv`
- **Test results**: `Fine_tuned_swin_transformer/test_set_error_curve.png`

---

## 🌐 Real Dataset Testing

### Nine-Point Testing

After fine-tuning, the model is tested on real dataset using the nine-point testing methodology to validate practical performance:

```bash
# Test Swin Transformer on real dataset
python nine_point_test_swim_transformer.py

# Test other methods for comparison
python nine_point_test_another_CNN.py
python nine_point_ODES_AIC.py
python nine_point_ADE_LM.py
python nine_point_peak_method.py
```

#### Real Dataset Structure

The real dataset testing uses a structured approach:
- **9 measurement points**: Systematic evaluation across different positions
- **Real ultrasound data**: Actual experimental measurements
- **Multiple methods**: Comprehensive comparison across all 6 approaches

#### Real Dataset Results

Real dataset testing demonstrates:
- **Swin Transformer**: Superior performance on real-world data
- **Generalization capability**: Consistent performance across different measurement points
- **Practical applicability**: Real-time inference capabilities

---

## 📊 other real Data Testing

### Generalization Validation

To prove the model's generalization capability, we test on Winston's paper data:

```bash
# Test generalization on Winston's paper data
python test_Wston_data.py
```

#### Winston Data Testing Purpose

This testing phase validates:
- **Cross-domain generalization**: Performance on external dataset
- **Method robustness**: Consistency across different data sources
- **Research reproducibility**: Validation against published results

#### Generalization Results

Winston data testing proves:
- **Swin Transformer**: Maintains high performance on external data
- **Superior generalization**: Outperforms traditional methods
- **Research contribution**: Validates the method's broad applicability

## 🛠️ Other Methods

### CNN Methods

#### Basic CNN (`CNN_pre.py`)
- **Architecture**: 1D Convolutional Neural Network
- **Purpose**: Traditional deep learning baseline
- **Features**: Batch normalization, dropout, ReLU activation

#### Improved CNN (`pre_another_CNN.py`)
- **Architecture**: Enhanced CNN with 3 convolutional layers
- **Purpose**: Improved deep learning baseline
- **Features**: 64 kernels per layer, kernel size 71, better regularization

### Traditional Signal Processing Methods

#### ODES + AIC (`ODESandAIC.py`)
- **Method**: Optimal Detection of Echo Signals with Akaike Information Criterion
- **Purpose**: Traditional signal processing baseline
- **Features**: Rule-based approach, no training required

#### ADE + LM (`ADE_and_LM.py`)

- **Method**: Adaptive Differential Evolution with Levenberg-Marquardt optimization
- **Purpose**: Optimization-based approach
- **Features**: Parameter estimation, curve fitting

#### Peak Method (`peak_method.py`)
- **Method**: Classical peak detection
- **Purpose**: Simple baseline method
- **Features**: Threshold-based detection, fast inference

### Additional Models

#### LSTM (`LSTM.py`)
- **Architecture**: Long Short-Term Memory network
- **Purpose**: Sequential modeling approach
- **Features**: Time series processing, recurrent architecture

#### BNN (`BNN_model.py`)
- **Architecture**: Bayesian Neural Network
- **Purpose**: Uncertainty quantification
- **Features**: Probabilistic predictions, uncertainty estimation


---

## !!!NOTICE!!!

Note that the trained model was tested on our local machine with the following hardware configuration: 12th Gen Intel Core i9-12900H CPU (14 cores / 20 threads, 2.5 GHz base frequency), NVIDIA GeForce RTX 3060 GPU with 6 GB dedicated memory, Intel Iris Xe integrated GPU, and 16 GB RAM, running on Windows 11 with SSD storage. Due to differences in hardware architectures and floating-point computation implementations across different machines, minor numerical discrepancies may occur when the model is run on other systems. These differences, however, do not affect the overall performance or the validity of the results reported.

---


## 📜 License & Notices

This project is released under the MIT License. Please observe:

1. **Code usage**:
   - Free to use, modify, distribute
   - Keep original license and notices
   - Do not use for illegal purposes

2. **Data usage**:
   - Example data for research only
   - Use your own data in production
   - Do not use copyrighted data without permission

3. **Model weights**:
   - Pretrained weights are non-commercial by default
   - Commercial use requires written permission

4. **Citation**:
   - Please cite related work if used in research
   - Suggested format:
     ```
     @misc{swin_tof_project,
       author = {Huiqin Jia, Jingui Jiang , Fei Li, and Guangbin Xiao},
       title = {High-precision ultrasonic flight time prediction based on GAF image encoding and deep learning model},
       year = {2025}}
     }
     ```

5. **Disclaimer**:
   - No liability for outcomes from using the code
   - Use at your own risk
   - No warranty of accuracy or fitness

[View full license](LICENSE)


**If you find this project useful, please consider leaving a ⭐ Star!**

[Open an issue](https://github.com/your-repo/issues) • [Feature request](https://github.com/your-repo/issues) • [Pull requests](https://github.com/your-repo/pulls)

</div>

