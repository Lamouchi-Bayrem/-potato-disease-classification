def evaluate_model(model, test_ds):
    scores = model.evaluate(test_ds)
    print(f"Test Loss: {scores[0]}, Test Accuracy: {scores[1]}")
    return scores
