import os

from data_download import download_dataset
from data_loading import load_dataset
from dataset_partition import get_dataset_partitions_tf
from model_building import build_model
from model_optimization import compile_model
from training import train_model
from evaluation import evaluate_model
from visualization import plot_training_history, visualize_predictions
from model_saving import save_model, save_h5_model
from tflite_conversion import convert_to_tflite, convert_with_quantization
from qat import train_qat_model  # Optional: Uncomment for QAT

def main():
    # Create directories if not exist
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    # Download dataset if not present
    download_dataset()
    
    dataset, class_names = load_dataset()
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    
    model = build_model()
    model.summary()
    model = compile_model(model)
    
    history = train_model(model, train_ds, val_ds)
    plot_training_history(history)
    
    evaluate_model(model, test_ds)
    visualize_predictions(model, test_ds, class_names)
    
    save_model(model)
    save_h5_model(model)
    
    convert_to_tflite(model, "../models/model.tflite")
    convert_with_quantization(model, train_ds, "../models/model_quant.tflite")
    
    # Optional: QAT (Quantization Aware Training)
    # q_aware_model, q_history = train_qat_model(train_ds, val_ds)
    # convert_to_tflite(q_aware_model, "../models/model_qat.tflite")

if __name__ == "__main__":
    main()
