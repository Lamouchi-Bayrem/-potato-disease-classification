import tensorflow as tf
from tensorflow.keras import layers

def preprocess_dataset(dataset):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return dataset

# Normalization layer
normalization_layer = layers.Rescaling(1./255)
