# ALPHA Framework

**A**utomated **L**abeling **P**rocess using a **H**uman-in-the-Loop Framework with **A**rtificial Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📋 Overview

ALPHA is a novel software engineering framework that implements human-in-the-loop methodology through collaborative AI components for biological image analysis. The framework specifically integrates **object detection models** and **validation filters** to create robust automated labeling systems that significantly reduce the annotation burden on domain experts while maintaining high accuracy.

## 🏗️ Architecture

![Image](https://github.com/user-attachments/assets/8ad2d97d-a16c-42d8-9def-fb41bca8cd22)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/alpha-framework.git
   cd alpha-framework

2. **Install dependencies**
   ```bash
   chmod +x install_requirements.sh
   ./install_requirements.sh
   pip install -r requirements.txt   

3. **Verify installation**
   ```bash
   python -c "import torch; from ultralytics import YOLO; print('✅ Installation successful!')"

## 📁 Dataset Preparation

### Required Directory Structure
```
dataset/
├── images/           # Your image files (.jpg, .jpeg, .png, .bmp)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/           # YOLO format annotation files (.txt)
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### YOLO Label Format
Each label file should contain bounding box annotations in YOLO format:
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
```
Where coordinates are normalized (0-1).

## 🎮 Execution Methods

### Method 1: Complete Pipeline (Recommended)
Run the entire ALPHA framework from start to finish:

```bash
python main.py
```

This executes all four steps sequentially:
1. **Initial YOLO Training** - Trains YOLO models with different data ratios
2. **First Inference + Manual Labeling** - Runs inference and provides labeling interface
3. **Classification Training** - Trains DenseNet classifier on labeled data
4. **Iterative Process** - Performs active learning cycles

### Method 2: Step-by-Step Execution
Execute specific steps individually:

#### Step 1: Initial YOLO Training
```bash
python main.py --step 1
```
- Trains YOLO models with different data percentages (10%, 20%, ..., 100%)
- Outputs trained models to `./results/01_initial_yolo/`

#### Step 2: First Inference + Manual Labeling
```bash
python main.py --step 2
```
- Runs inference on images using the best YOLO model
- Provides 4 labeling options:
  1. **GUI Labeling** (Recommended) - Interactive graphical interface
  2. **CLI Labeling** - Terminal-based labeling
  3. **Batch Labeling** - File-based labeling
  4. **Auto Labeling** - Confidence-based automatic labeling

#### Step 3: Classification Training
```bash
python main.py --step 3
```
- Trains DenseNet121 classifier on manually labeled data
- Uses different data ratios for robust training

#### Step 4: Iterative Active Learning
```bash
python main.py --step 4
```
- Runs iterative cycles combining YOLO detection and classification
- Performs active learning to improve model performance

### Method 3: Custom Configuration
Create and use custom configuration files:

#### Create Default Configuration
```bash
python main.py --create-config my_config.json
```

#### Run with Custom Configuration
```bash
python main.py --config my_config.json
```

#### Sample Configuration Parameters
```json
{
  "dataset_root": "./dataset",
  "images_dir": "./dataset/images",
  "labels_dir": "./dataset/labels",
  "yolo_epochs": 100,
  "classification_epochs": 30,
  "data_percentages": [10, 20, 50, 100],
  "conf_threshold": 0.25,
  "gpu_num": 0
}
```

### Method 4: Command Line Arguments
Override default settings with command line arguments:

```bash
# Specify custom directories
python main.py --images_dir /path/to/images --labels_dir /path/to/labels

# Use specific GPU
python main.py --gpu_num 1

# Set custom output directory
python main.py --output_dir ./my_results

# Combine multiple options
python main.py --step 1 --gpu_num 0 --config my_config.json
```

## 🏷️ Manual Labeling Options

### 1. GUI Labeling (Recommended)
- **Interactive Interface**: Point-and-click labeling with visual feedback
- **Real-time Preview**: See detection results immediately
- **Easy Navigation**: Browse through detected objects efficiently
- **Requirements**: GUI libraries (tkinter)
- 
![Image](https://github.com/user-attachments/assets/85f62cd2-c232-4878-8046-79f8b12e4b61)

**Ubuntu/Debian Setup:**
```bash
sudo apt-get install python3-tk
```

**CentOS/RHEL Setup:**
```bash
sudo yum install tkinter
```

### 2. Auto Labeling
- **Confidence-based**: Automatically classifies based on detection confidence
- **Threshold Control**: Adjustable confidence threshold (0.3-0.9)
- **Fast Processing**: Suitable for large datasets
- **Usage**: Enter threshold when prompted (default: 0.6)

### 3. CLI/Batch Labeling
- **Fallback Options**: Available when GUI is not accessible
- **Simplified Interface**: Currently redirects to auto labeling

## 📊 Expected Outputs

### Directory Structure After Execution
```
results/
├── 01_initial_yolo/        # Trained YOLO models
│   ├── yolov8_10.pt
│   ├── yolov8_20.pt
│   └── ...
├── 02_first_inference/     # Inference visualizations
├── 03_manual_labeling/     # Labeled training data
│   ├── class0/            # Objects to keep
│   └── class1/            # Objects to filter
├── 04_classification/      # Trained classifiers
│   ├── densenet121_10.pth
│   └── ...
└── 05_iterative_process/   # Final results
    ├── cycle_1/
    ├── cycle_2/
    └── summary.json
```

### Performance Metrics
The framework outputs detailed performance metrics including:
- **F1-scores** for each model and data ratio
- **Precision and Recall** values
- **Cross-validation results**
- **Active learning cycle improvements**

## ⚙️ Configuration Options

### Key Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `yolo_epochs` | YOLO training epochs | 100 | 50-300 |
| `classification_epochs` | Classifier training epochs | 30 | 10-100 |
| `conf_threshold` | Detection confidence threshold | 0.25 | 0.1-0.9 |
| `class_conf_threshold` | Classification confidence threshold | 0.5 | 0.1-0.9 |
| `max_cycles` | Maximum active learning cycles | 10 | 1-20 |
| `batch_size` | Training batch size | 16 | 8-64 |

### Data Ratio Settings
- **YOLO Training**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`
- **Classification**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Memory Error
```bash
# Reduce batch size
python main.py --config my_config.json  # Edit batch_size in config
```

#### 2. GUI Labeling Not Available
```bash
# Install GUI libraries
sudo apt-get install python3-tk  # Ubuntu/Debian
sudo yum install tkinter          # CentOS/RHEL
```

#### 3. No Trained Models Found
```bash
# Run previous steps first
python main.py --step 1  # Train YOLO models first
python main.py --step 2  # Then run labeling
```

#### 4. Insufficient Labeled Data
- Ensure both `class0/` and `class1/` directories contain images
- Try auto labeling with different confidence thresholds
- Use GUI labeling for better control

### Performance Optimization

#### GPU Utilization
```bash
# Check GPU usage
nvidia-smi

# Use specific GPU
python main.py --gpu_num 1
```

#### Memory Management
- Reduce `batch_size` if encountering OOM errors
- Use smaller `img_size` for YOLO training
- Close other GPU-intensive applications

## 🎯 Key Features
- **Human-in-the-Loop Design**: Seamlessly integrates human expertise with AI automation
- **Dual AI Components**: Combines YOLO object detection with DenseNet classification for robust performance
- **Noise Reduction**: Advanced validation filters reduce annotation errors by 83%
- **Data Efficiency**: Achieves near-optimal performance using only 10% of original labeled data
- **Cross-Domain Generalization**: Robust performance across different biological datasets
- **Modular Architecture**: Easy to extend and customize for various biological applications

## 📊 Performance Highlights
- **F1-scores**: 0.89-0.95 on blood smear datasets with minimal data
- **Cross-domain F1-scores**: 0.88-0.97 across different domains
- **Error Reduction**: 83% reduction in intentional annotation errors
- **Data Requirement**: Only 10% of original labeled data needed

## 📞 Support
If you have any questions or provide your cell images, please contact us by email(kc.jeong-isw@chungbuk.ac.kr, gc.jo-isw@chungbuk.ac.kr).


