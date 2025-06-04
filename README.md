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

