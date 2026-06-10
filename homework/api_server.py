"""Flask API for house price prediction."""

from __future__ import print_function

import os
import pickle

from flask import Flask, jsonify, request
import pandas as pd

FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_predictor.pkl")

app = Flask(__name__)


def load_model():
    """Load the serialized model from disk."""

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


MODEL = load_model() if MODEL_PATH.exists() else None


@app.get("/")
def index():
    """Health endpoint."""

    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    """Return a predicted house price."""

    payload = request.get_json(silent=True) or request.form.to_dict()
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        return jsonify({"error": "Missing fields: {0}".format(", ".join(missing))}), 400

    input_frame = pd.DataFrame([[payload[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    input_frame = input_frame.astype(
        {
            "bedrooms": float,
            "bathrooms": float,
            "sqft_living": float,
            "sqft_lot": float,
            "floors": float,
            "waterfront": int,
            "condition": float,
        }
    )

    model = MODEL or load_model()
    prediction = float(model.predict(input_frame)[0])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)