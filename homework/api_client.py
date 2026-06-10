"""Client for the house price prediction API."""

from __future__ import print_function

import argparse

import requests

DEFAULT_URL = "http://127.0.0.1:5000/predict"


def predict(url, payload):
    """Send a prediction request and return the result."""

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return float(response.json()["prediction"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the house prediction API.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--bedrooms", type=float, required=True)
    parser.add_argument("--bathrooms", type=float, required=True)
    parser.add_argument("--sqft-living", dest="sqft_living", type=float, required=True)
    parser.add_argument("--sqft-lot", dest="sqft_lot", type=float, required=True)
    parser.add_argument("--floors", type=float, required=True)
    parser.add_argument("--waterfront", type=int, required=True)
    parser.add_argument("--condition", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = {
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "sqft_living": args.sqft_living,
        "sqft_lot": args.sqft_lot,
        "floors": args.floors,
        "waterfront": args.waterfront,
        "condition": args.condition,
    }
    print(predict(args.url, payload))