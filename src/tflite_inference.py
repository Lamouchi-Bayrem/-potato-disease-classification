import tensorflow as tf
import numpy as np

def load_tflite_model(filename):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, img, class_names):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0).astype(np.float32)
    
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    
    predicted_class = class_names[digit]
    confidence = np.max(output()[0]) * 100
    return predicted_class, confidence
