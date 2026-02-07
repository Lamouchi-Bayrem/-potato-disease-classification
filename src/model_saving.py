import os

def save_model(model, model_dir="../models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_version = max([int(i) for i in os.listdir(model_dir) + [0]]) + 1
    model.save(f"{model_dir}/{model_version}")

def save_h5_model(model, filename="../models/potatoes.h5"):
    model.save(filename)
def save_onnx_model(model, filename="../models/model.onnx", input_shape=(1, 224, 224, 3)):
    """
    Convert and save TensorFlow/Keras model to ONNX
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Define input signature
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=filename
    )

    print(f"✅ ONNX model saved at: {filename}")
