import os
from pathlib import Path

import mlflow.pyfunc
import streamlit as st

from housing_model import FEATURES


MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "model")
)

st.set_page_config(
    page_title="California Home Value Predictor",
    page_icon="🏡",
    layout="wide",
)

st.title("California Home Value Predictor")
st.caption(
    "Enter details about a California neighborhood to estimate its median home value."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model bundle not found at {MODEL_PATH}. "
        "Run `python export_model.py` to export the registered MLflow model."
    )
    st.stop()


@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(str(MODEL_PATH))


model = load_model()

with st.form("prediction_form"):
    st.subheader("Neighborhood details")
    st.write("All fields are required. Use the same units shown in each label.")

    income_col, age_col, rooms_col, bedrooms_col = st.columns(4)
    with income_col:
        median_income = st.number_input(
            "Median income ($10,000s)",
            min_value=0.0,
            value=8.3252,
            step=0.1,
            help="Median household income for the neighborhood, in tens of thousands of dollars.",
        )
    with age_col:
        house_age = st.number_input(
            "Median house age (years)",
            min_value=0.0,
            value=41.0,
            step=1.0,
            help="Median age of houses in the neighborhood.",
        )
    with rooms_col:
        average_rooms = st.number_input(
            "Average rooms per household",
            min_value=0.0,
            value=6.9841,
            step=0.1,
            help="Average number of rooms per household.",
        )
    with bedrooms_col:
        average_bedrooms = st.number_input(
            "Average bedrooms per household",
            min_value=0.0,
            value=1.0238,
            step=0.1,
            help="Average number of bedrooms per household.",
        )

    population_col, occupancy_col, latitude_col, longitude_col = st.columns(4)
    with population_col:
        population = st.number_input(
            "Neighborhood population",
            min_value=0,
            value=322,
            step=10,
            help="Total population of the neighborhood.",
        )
    with occupancy_col:
        average_occupancy = st.number_input(
            "Average household occupancy",
            min_value=0.0,
            value=2.5556,
            step=0.1,
            help="Average number of people per household.",
        )
    with latitude_col:
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=37.88,
            step=0.01,
            format="%.4f",
            help="California latitude, approximately 32 to 42 degrees north.",
        )
    with longitude_col:
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-122.23,
            step=0.01,
            format="%.4f",
            help="California longitude, approximately -125 to -114 degrees.",
        )

    submitted = st.form_submit_button(
        "Estimate home value",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_values = {
        "MedInc": median_income,
        "HouseAge": house_age,
        "AveRooms": average_rooms,
        "AveBedrms": average_bedrooms,
        "Population": population,
        "AveOccup": average_occupancy,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    prediction = float(
        model.predict([[input_values[feature] for feature in FEATURES]])[0]
    )
    estimated_dollars = prediction * 100_000

    st.divider()
    st.subheader("Prediction")
    st.metric(
        "Estimated median home value",
        f"${estimated_dollars:,.0f}",
        help="The model predicts values in units of $100,000; this display converts that value to dollars.",
    )
    st.caption(
        f"Model output: {prediction:.3f} (in units of $100,000). "
        "This is an estimate based on the trained model, not a property appraisal."
    )
