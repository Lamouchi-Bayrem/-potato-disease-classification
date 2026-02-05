import os

def save_model(model, model_dir="../models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_version = max([int(i) for i in os.listdir(model_dir) + [0]]) + 1
    model.save(f"{model_dir}/{model_version}")

def save_h5_model(model, filename="../models/potatoes.h5"):
    model.save(filename)
