from json import dumps
from collections import OrderedDict

from joblib import load
from numpy import argsort


def load(model_file_path):
    return load(model_file_path)


def predict(model, input):
    return model.predict_proba(input)


def top_n_predictions_ids(predicted, number=3):
    return argsort(predicted, axis=1)[:, -number:]


def format_predictions(input_data, sorted_predictions, raw_predictions, labels, labels_ids):
    return [
        OrderedDict(
            {
                "label": labels[labels_ids[predicted_label]],
                "probability": f"{raw_predictions[row][predicted_label]:.0%}"
            }
        )
        for row, _ in enumerate(input_data)
        for predicted_label in sorted_predictions[row]
    ]


def serialize_predictions(predictions):
    return dumps(predictions)
