"""Train a house price prediction model."""

from __future__ import print_function

import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "files", "input", "house_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "house_predictor.pkl")


def build_model():
    """Create the training pipeline."""

    numeric_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
    ]
    categorical_features = ["waterfront"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def load_training_data():
    """Load the training data from disk."""

    data = pd.read_csv(DATA_PATH)
    features = data[FEATURE_COLUMNS].copy()
    target = data["price"].copy()
    return features, target


def train_model():
    """Fit the model and persist it to disk."""

    features, target = load_training_data()
    model = build_model()
    model.fit(features, target)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    return model


if __name__ == "__main__":
    train_model()