from joblib import load
from numpy import argsort


def load(model_file_path):
    return load(model_file_path)


def predict(model, input):
    return model.predict_proba(input)


def top_n_predictions_ids(predicted, number=3):
    return argsort(predicted, axis=1)[:, -number:]


def format_predictions():
    pass
