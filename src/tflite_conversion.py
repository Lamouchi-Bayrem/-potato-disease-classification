# tflite_conversion.py
import tensorflow as tf
import numpy as np

def convert_to_tflite(model, filename="model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)

def convert_with_quantization(model, dataset, filename="model_quant.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        for input_value, _ in dataset.take(100):
            yield [input_value]
    
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model_quant = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model_quant)
