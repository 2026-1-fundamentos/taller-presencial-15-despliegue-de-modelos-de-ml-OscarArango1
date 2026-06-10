"""Flask web application for house price prediction."""

from __future__ import print_function

import os
import pickle

from flask import Flask, render_template_string, request
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


with open(MODEL_PATH, "rb") as model_file:
    MODEL = pickle.load(model_file)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <title>House Predictor</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background: #f6f7fb; }
      .card { max-width: 720px; margin: auto; background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,.08); }
      .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
      label { display: block; font-weight: 700; margin-bottom: .35rem; }
      input { width: 100%; padding: .7rem; border: 1px solid #cfd5e2; border-radius: 10px; }
      .full { grid-column: 1 / -1; }
      button { margin-top: 1rem; padding: .85rem 1.2rem; border: 0; border-radius: 10px; background: #0b5fff; color: white; font-weight: 700; }
      .result { margin-top: 1.5rem; font-size: 1.25rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>House Predictor</h1>
      <form method="post">
        <div class="grid">
          {% for field in fields %}
          <div class="{% if field == 'condition' %}full{% endif %}">
            <label for="{{ field }}">{{ field }}</label>
            <input id="{{ field }}" name="{{ field }}" type="number" step="any" required value="{{ request.form.get(field, '') }}">
          </div>
          {% endfor %}
        </div>
        <button type="submit">Predecir</button>
      </form>
      {% if prediction is not none %}
      <div class="result">Precio estimado: {{ '{:,.2f}'.format(prediction) }}</div>
      {% endif %}
    </div>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and prediction result."""

    prediction = None
    if request.method == "POST":
        frame = pd.DataFrame(
            [[request.form.get(field, 0) for field in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        ).astype(
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
        prediction = float(MODEL.predict(frame)[0])

    return render_template_string(
        PAGE_TEMPLATE,
        fields=FEATURE_COLUMNS,
        prediction=prediction,
        request=request,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)