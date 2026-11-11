import math
from itertools import product

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlflow.tracking import MlflowClient


RANDOM_STATE = 42
EXPERIMENT_NAME = "california-housing-linear-regression"
REGISTERED_MODEL_NAME = "CaliforniaHousingLinearRegression"
CHAMPION_ALIAS = "champion"
PREPROCESSORS = (
    ("StandardScaler", StandardScaler),
    ("MinMaxScaler", MinMaxScaler),
)
TEST_SIZES = (0.2, 0.25)
FIT_INTERCEPT_VALUES = (True, False)


def select_best_logged_model(experiment_id, client):
    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        max_results=10_000,
        output_format="list",
    )
    candidates = []

    for logged_model in logged_models:
        if (
            logged_model.name != "model"
            or logged_model.status.value != "READY"
            or not logged_model.source_run_id
        ):
            continue

        run = client.get_run(logged_model.source_run_id)
        if run.info.status != "FINISHED":
            continue

        metrics = {metric.key: metric.value for metric in logged_model.metrics or []}
        mse = metrics.get("mean_squared_error")
        r2 = metrics.get("r2_score")
        if mse is None or r2 is None or not math.isfinite(mse) or not math.isfinite(r2):
            continue

        candidates.append((mse, -r2, run.info.run_id, logged_model, run, r2))

    if not candidates:
        raise RuntimeError(
            f"No finished, ready models with MSE and R² were found in "
            f"experiment {EXPERIMENT_NAME!r}."
        )

    return min(candidates, key=lambda candidate: candidate[:3])


def register_champion(experiment_id):
    client = MlflowClient()
    _, _, _, best_model, best_run, best_r2 = select_best_logged_model(
        experiment_id, client
    )
    best_mse = next(
        metric.value
        for metric in best_model.metrics
        if metric.key == "mean_squared_error"
    )

    existing_versions = client.search_model_versions(
        filter_string=f"name='{REGISTERED_MODEL_NAME}'",
        max_results=10_000,
    )
    version = next(
        (
            model_version
            for model_version in existing_versions
            if model_version.run_id == best_run.info.run_id
        ),
        None,
    )
    if version is None:
        version = mlflow.register_model(
            model_uri=best_model.model_uri,
            name=REGISTERED_MODEL_NAME,
            tags={
                "selection.mean_squared_error": str(best_mse),
                "selection.r2_score": str(best_r2),
                "selection.run_id": best_run.info.run_id,
            },
        )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    return best_model, best_run, best_mse, best_r2, version


def main():
    housing = fetch_california_housing()
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    results = []

    configurations = product(
        PREPROCESSORS, TEST_SIZES, FIT_INTERCEPT_VALUES
    )
    for run_number, ((scaler_name, scaler_class), test_size, fit_intercept) in enumerate(
        configurations, start=1
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            housing.data,
            housing.target,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        pipeline = Pipeline(
            [
                ("scaler", scaler_class()),
                ("regressor", LinearRegression(fit_intercept=fit_intercept)),
            ]
        )

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"run-{run_number:02d}-{scaler_name}-test-{test_size}-intercept-{fit_intercept}",
        ) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            mlflow.log_params(
                {
                    "dataset": "California Housing",
                    "model": "LinearRegression",
                    "scaler": scaler_name,
                    "test_size": test_size,
                    "random_state": RANDOM_STATE,
                    "fit_intercept": fit_intercept,
                    "training_rows": len(y_train),
                    "feature_count": housing.data.shape[1],
                }
            )
            mlflow.log_metrics({"mean_squared_error": mse, "r2_score": r2})
            mlflow.sklearn.log_model(pipeline, name="model")

            results.append(
                {
                    "run_id": run.info.run_id,
                    "scaler": scaler_name,
                    "test_size": test_size,
                    "fit_intercept": fit_intercept,
                    "mean_squared_error": mse,
                    "r2_score": r2,
                }
            )

    print("\nRuns from this experiment, sorted by R2 (highest first):")
    print(
        f"{'Run ID':<34} {'Scaler':<16} {'Test size':>9} "
        f"{'Fit intercept':>13} {'MSE':>12} {'R²':>10}"
    )
    for result in sorted(results, key=lambda item: item["r2_score"], reverse=True):
        print(
            f"{result['run_id']:<34} {result['scaler']:<16} "
            f"{result['test_size']:>9.2f} {str(result['fit_intercept']):>13} "
            f"{result['mean_squared_error']:>12.4f} {result['r2_score']:>10.4f}"
        )

    best_model, best_run, best_mse, best_r2, version = register_champion(
        experiment.experiment_id
    )
    print("\nBest model across all finished runs in this experiment:")
    run_name = best_run.data.tags.get("mlflow.runName", "unnamed")
    print(f"  Run: {best_run.info.run_id} ({run_name})")
    print(f"  MSE: {best_mse:.4f}")
    print(f"  R2:  {best_r2:.4f}")
    print(f"  Scaler: {best_model.params.get('scaler', 'unknown')}")
    print(f"  Test size: {best_model.params.get('test_size', 'unknown')}")
    print(f"  Registry: {REGISTERED_MODEL_NAME} version {version.version}")
    print(f"  Alias: {CHAMPION_ALIAS}")
    print(f"\nOpen the MLflow UI and compare runs in '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
