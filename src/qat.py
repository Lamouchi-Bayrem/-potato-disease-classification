import tensorflow as tf
import tensorflow_model_optimization as tfmot

from model_building import build_model
from model_optimization import compile_model
from training import train_model

def apply_qat(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    return q_aware_model

def train_qat_model(train_ds, val_ds):
    base_model = build_model()
    q_aware_model = apply_qat(base_model)
    q_aware_model = compile_model(q_aware_model)
    history = train_model(q_aware_model, train_ds, val_ds)
    return q_aware_model, history
