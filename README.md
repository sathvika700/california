# California Home Value Predictor

A Streamlit interface for the registered California Housing MLflow model.
The Docker image contains an exported copy of the model, so it does not need
the MLflow tracking database or tracking server at runtime.

## Run locally

From the project directory, activate the project virtual environment and install
the dependencies:

```powershell
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
python export_model.py
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. The export step reads the model registered under
the `champion` alias and writes a portable MLflow model to `model/`.

## Build and run with Docker

Export the model first if `model/` does not already exist. Then, from the project
directory, run:

```powershell
docker build -t california-housing-ui .
docker run --rm -p 8501:8501 california-housing-ui
```

Open `http://localhost:8501`. To use another exported model directory, set
`MODEL_OUTPUT_DIR` when running `export_model.py`; the Docker build expects its
model bundle at `model/`.

The app also supports `MODEL_PATH` at runtime to load a model bundle from a
different path.

## GitHub Actions

The workflow in `.github/workflows/docker-ci.yml` runs on every push and pull
request. It builds the Docker image, starts the container, checks the Streamlit
health endpoint, and verifies that the bundled model returns a finite
prediction. It does not publish the image to a registry.

Commit the exported `model/` bundle along with the project so the GitHub runner
can build the image without access to the local MLflow database. The project
`.gitignore` excludes the local virtual environment and MLflow tracking data.
