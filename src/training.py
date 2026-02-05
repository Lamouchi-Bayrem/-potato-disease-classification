from constants import EPOCHS
from data_preprocessing import preprocess_dataset

def train_model(model, train_ds, val_ds):
    train_ds = preprocess_dataset(train_ds)
    val_ds = preprocess_dataset(val_ds)
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )
    return history
