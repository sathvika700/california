import os
from pathlib import Path

import mlflow
import mlflow.sklearn


PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}",
)
MODEL_URI = os.environ.get(
    "MLFLOW_MODEL_URI",
    "models:/CaliforniaHousingLinearRegression@champion",
)
OUTPUT_DIR = Path(os.environ.get("MODEL_OUTPUT_DIR", PROJECT_DIR / "model"))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    mlflow.sklearn.save_model(sk_model=model, path=str(OUTPUT_DIR))
    print(f"Exported {MODEL_URI} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
