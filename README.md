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
- Automated dataset download using Kaggle API

## Setup
1. Clone the repo: `git clone https://github.com/Lamouchi-Bayrem/potato-disease-classification.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Kaggle API:
   - Sign up on [Kaggle](https://www.kaggle.com) and generate an API key (`kaggle.json`).
   - Place `kaggle.json` in `~/.kaggle/` (on Unix) or `C:\Users\<username>\.kaggle\` (on Windows).
   - Run `chmod 600 ~/.kaggle/kaggle.json` for permissions.
4. Run: `python src/main.py` – This will download the dataset if not present, train the model, evaluate, and save models/TFLite files in `models/`.

## Directory Structure
- `src/`: Source code
- `data/`: Dataset (downloaded automatically)
- `models/`: Saved models and TFLite files (generated on run)

## Usage
- Train and evaluate: Run `main.py` (downloads data, trains, saves model)
- Predict on new image: Use `prediction.py` functions
- TFLite inference: Use `tflite_inference.py`

If download fails, manually download from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village) and extract to `data/PlantVillage/`.

Dataset credit: Kaggle Plant Village.

Author: Bayrem - Data Science & AI Expert

