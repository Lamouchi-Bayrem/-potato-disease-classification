# Potato Disease Classification

## Overview
This project classifies potato leaf images into three categories: Early Blight, Late Blight, and Healthy using a CNN built with TensorFlow. It demonstrates data science and AI skills in computer vision.

## Skills Demonstrated
- Data loading and partitioning
- Preprocessing and augmentation
- CNN model building
- Training and evaluation
- Model optimization (compilation)
- Visualization of results
- Model saving and TFLite conversion for deployment
- Optional: Quantization Aware Training (QAT)

## Setup
1. Clone the repo: `git clone https://github.com/yourusername/potato-disease-classification.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village) and place in `data/PlantVillage/`
4. Run: `python src/main.py`

## Directory Structure
- `src/`: Source code
- `data/`: Dataset (not committed)
- `models/`: Saved models (not committed)

## Usage
- Train and evaluate: Run `main.py`
- Predict on new image: Use `prediction.py` functions
- TFLite inference: Use `tflite_inference.py`

Dataset credit: Kaggle Plant Village.

Author: [Your Name] (bayrem) - Data Science & AI Expert
