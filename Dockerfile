FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

COPY requirements.txt ./
COPY model/requirements.txt ./model-requirements.txt
RUN pip install --no-cache-dir -r model-requirements.txt -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=10001:10001 housing_model.py streamlit_app.py ./
COPY --chown=10001:10001 model ./model

USER 10001

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
