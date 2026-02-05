import tensorflow as tf

from constants import BATCH_SIZE, IMAGE_SIZE, DATA_DIR, CLASS_NAMES

def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    class_names = dataset.class_names  # Or use CLASS_NAMES if mismatched
    return dataset, class_names
