# ALPHA Framework

**A**utomated **L**abeling **P**rocess using a **H**uman-in-the-Loop Framework with **A**rtificial Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📋 Overview

ALPHA is a novel software engineering framework that implements human-in-the-loop methodology through collaborative AI components for biological image analysis. The framework specifically integrates **object detection models** and **validation filters** to create robust automated labeling systems that significantly reduce the annotation burden on domain experts while maintaining high accuracy.

### 🎯 Key Features

- **Human-in-the-Loop Design**: Seamlessly integrates human expertise with AI automation
- **Dual AI Components**: Combines YOLO object detection with DenseNet classification for robust performance
- **Noise Reduction**: Advanced validation filters reduce annotation errors by 83%
- **Data Efficiency**: Achieves near-optimal performance using only 10% of original labeled data
- **Cross-Domain Generalization**: Robust performance across different biological datasets
- **Modular Architecture**: Easy to extend and customize for various biological applications

### 📊 Performance Highlights

- **F1-scores**: 0.89-0.95 on blood smear datasets with minimal data
- **Cross-domain F1-scores**: 0.88-0.97 across different domains
- **Error Reduction**: 83% reduction in intentional annotation errors
- **Data Requirement**: Only 10% of original labeled data needed

## 🏗️ Architecture

The ALPHA framework consists of four main components:

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

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


markdown# ALPHA Framework

**A**utomated **L**abeling **P**rocess using a **H**uman-in-the-Loop Framework with **A**rtificial Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](#)

## 📋 Overview

ALPHA is a novel software engineering framework that implements human-in-the-loop methodology through collaborative AI components for biological image analysis. The framework specifically integrates **object detection models** and **validation filters** to create robust automated labeling systems that significantly reduce the annotation burden on domain experts while maintaining high accuracy.

### 🎯 Key Features

- **Human-in-the-Loop Design**: Seamlessly integrates human expertise with AI automation
- **Dual AI Components**: Combines YOLO object detection with DenseNet classification for robust performance
- **Noise Reduction**: Advanced validation filters reduce annotation errors by 83%
- **Data Efficiency**: Achieves near-optimal performance using only 10% of original labeled data
- **Cross-Domain Generalization**: Robust performance across different biological datasets
- **Modular Architecture**: Easy to extend and customize for various biological applications

### 📊 Performance Highlights

- **F1-scores**: 0.89-0.95 on blood smear datasets with minimal data
- **Cross-domain F1-scores**: 0.88-0.97 across different domains
- **Error Reduction**: 83% reduction in intentional annotation errors
- **Data Requirement**: Only 10% of original labeled data needed

## 🏗️ Architecture

The ALPHA framework consists of four main components:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Initial YOLO  │───▶│  Human-in-Loop  │───▶│ Classification  │───▶│   Iterative     │
│    Training     │    │    Labeling     │    │    Training     │    │   Refinement    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
│                       │                       │                       │
▼                       ▼                       ▼                       ▼
Multi-ratio            Manual/Auto              DenseNet121             Active Learning
YOLO models            Validation               Classifier              with Filtering

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/alpha-framework.git
   cd alpha-framework

Install dependencies
bash# Automatic installation (recommended)
chmod +x install_requirements.sh
./install_requirements.sh

# Or manual installation
pip install -r requirements.txt

Verify installation
bashpython -c "import torch; from ultralytics import YOLO; print('✅ Installation successful!')"


Basic Usage

Prepare your data
dataset/
├── images/          # Your biological images (.jpg, .png, etc.)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/          # YOLO format labels (.txt)
    ├── image1.txt
    ├── image2.txt
    └── ...

Create configuration
bashpython main.py --create-config config.json
# Edit config.json to match your data paths

Run the complete pipeline
bashpython main.py --config config.json


📚 Detailed Usage
Step-by-Step Execution
For more control, you can run each step individually:
bash# Step 1: Initial YOLO Training with multiple data ratios
python main.py --config config.json --step 1

# Step 2: Human-in-the-Loop Labeling (4 methods available)
python main.py --config config.json --step 2

# Step 3: Classification Model Training
python main.py --config config.json --step 3

# Step 4: Iterative Active Learning Process
python main.py --config config.json --step 4
Human-in-the-Loop Labeling Options
The framework provides multiple labeling approaches:

GUI Labeling (Recommended): Interactive graphical interface
CLI Labeling: Terminal-based interactive labeling
Batch Labeling: File explorer-based manual sorting
Auto Labeling: Confidence-based automatic classification

Configuration Examples
For High-Performance GPUs
json{
  "yolo_model_type": "yolov8m.pt",
  "batch_size": 32,
  "img_size": 640,
  "yolo_epochs": 150,
  "classification_epochs": 50
}
For Limited Resources
json{
  "yolo_model_type": "yolov8n.pt", 
  "batch_size": 8,
  "img_size": 416,
  "yolo_epochs": 50,
  "classification_epochs": 20
}
For Research/Testing
json{
  "data_percentages": [10, 50, 100],
  "classification_ratios": [0.1, 0.5, 1.0],
  "max_cycles": 5
}
📖 Use Cases
Biological Image Analysis

Blood Smear Analysis: Cell detection and classification
Microscopy Images: Tissue and cellular structure analysis
Medical Diagnostics: Automated screening and analysis
Research Applications: Quantitative biological measurements

Supported Data Types

Microscopy Images: Brightfield, fluorescence, phase contrast
Medical Images: Blood smears, tissue sections, cell cultures
File Formats: JPEG, PNG, TIFF, BMP
Label Format: YOLO format (class x_center y_center width height)

🔧 Advanced Features
Custom Model Integration
python# Use your own pre-trained YOLO model
config['yolo_model_type'] = 'path/to/your/model.pt'

# Customize classification architecture
trainer = ClassificationTrainer(
    model_architecture='densenet121',  # or 'resnet50', 'efficientnet'
    pretrained=True
)
Batch Processing
python# Process multiple datasets
datasets = ['dataset1', 'dataset2', 'dataset3']
for dataset in datasets:
    config['images_dir'] = f'./{dataset}/images'
    config['labels_dir'] = f'./{dataset}/labels'
    pipeline = CompletePipeline(config)
    pipeline.run_complete_pipeline()
Performance Monitoring
python# Access detailed metrics
results = pipeline.run_complete_pipeline()
metrics = results['combined_metrics']
print(f"Final F1-Score: {metrics['F1-Score'].iloc[-1]:.3f}")
📊 Output Structure
results/
├── 01_initial_yolo/
│   ├── yolov8_10pct.pt
│   ├── yolov8_50pct.pt
│   └── yolov8_100pct.pt
├── 02_first_inference/
│   └── inference_*.jpg
├── 03_manual_labeling/
│   ├── class0/              # Objects to keep
│   └── class1/              # Objects to filter
├── 04_classification/
│   ├── densenet121_*.pth
│   └── training_plots/
└── 05_iterative_process/
    ├── combined_performance_metrics.csv
    ├── experiment_report.txt
    └── model_comparisons/
🔬 Research Applications
This framework has been successfully applied to:

Automated Blood Cell Analysis: Achieving 0.95 F1-score with minimal annotation
Cross-Domain Medical Imaging: Robust generalization across different datasets
Error Correction Studies: Demonstrating 83% reduction in annotation errors
Data Efficiency Research: Proving effectiveness with only 10% of labeled data

🤝 Contributing
We welcome contributions from the research community! Please see our Contributing Guidelines for details.
Development Setup
bashgit clone https://github.com/your-username/alpha-framework.git
cd alpha-framework
pip install -e .
pre-commit install
📄 Citation
If you use the ALPHA framework in your research, please cite our paper:
bibtex@article{alpha2024,
  title={ALPHA: Automated Labeling Process using a Human-in-the-Loop Framework with Artificial Intelligence for Biological Image Analysis},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  doi={your-doi}
}
📞 Support

Documentation: Wiki
Issues: GitHub Issues
Discussions: GitHub Discussions
Email: your.email@institution.edu

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

YOLO Team: For the excellent object detection framework
PyTorch Team: For the deep learning infrastructure
Research Community: For valuable feedback and contributions
Domain Experts: For providing biological expertise and validation

🔗 Related Projects

Ultralytics YOLO
PyTorch
OpenCV


Note: This framework is part of ongoing research in automated biological image analysis. For the latest updates and experimental results, please refer to our published paper and this repository.
📈 Performance Benchmarks
DatasetMethodF1-ScoreData UsedTraining TimeBlood Smear AALPHA0.9510%2.3 hoursBlood Smear BALPHA0.9210%2.1 hoursBlood Smear CALPHA0.8910%2.5 hoursCross-DomainALPHA0.88-0.9710%2-3 hours
Benchmarks performed on NVIDIA RTX 3080 GPU
