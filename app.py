import math
import os
from pathlib import Path

import mlflow.pyfunc
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from housing_model import FEATURES

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{Path(__file__).with_name('mlflow.db').resolve().as_posix()}",
)
MODEL_URI = os.environ.get(
    "MLFLOW_MODEL_URI",
    "models:/CaliforniaHousingLinearRegression@champion",
)

mlflow.set_tracking_uri(TRACKING_URI)

app = Flask(__name__)
model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def index():
    return jsonify(
        {
            "message": "California Housing prediction API",
            "endpoints": {
                "health": "GET /health",
                "predict": "POST /predict",
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    try:
        payload = request.get_json()
    except BadRequest:
        return jsonify({"error": "Request body must contain valid JSON"}), 400

    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        return jsonify({"error": "Expected a JSON object with an 'instances' list"}), 400
    if not payload["instances"]:
        return jsonify({"error": "'instances' must contain at least one item"}), 400

    rows = []
    for index, instance in enumerate(payload["instances"]):
        if not isinstance(instance, dict):
            return jsonify({"error": f"Instance {index} must be a JSON object"}), 400

        missing = [feature for feature in FEATURES if feature not in instance]
        extra = [feature for feature in instance if feature not in FEATURES]
        if missing or extra:
            return jsonify(
                {
                    "error": f"Instance {index} has incorrect features",
                    "missing": missing,
                    "unexpected": extra,
                }
            ), 400

        values = []
        for feature in FEATURES:
            value = instance[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return jsonify(
                    {"error": f"Instance {index} feature '{feature}' must be a number"}
                ), 400
            if not math.isfinite(value):
                return jsonify(
                    {"error": f"Instance {index} feature '{feature}' must be finite"}
                ), 400
            values.append(value)
        rows.append(values)

    predictions = model.predict(rows)
    return jsonify({"predictions": [float(value) for value in predictions]})


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
