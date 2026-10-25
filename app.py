

from pathlib import Path
import json

import streamlit as st
import sklearn
import joblib
import numpy
import pandas as pd

st.write("Diagnostic: app.py started")
st.write("scikit-learn:", sklearn.__version__)
st.write("joblib:", joblib.__version__)
st.write("NumPy:", numpy.__version__)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR / "tourism_project" / "deployment"

MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)

# --------------------------------------------------
# Load the trained model and metadata
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Ensure the pipeline commits best_model.joblib "
            "to tourism_project/deployment/."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Application
# --------------------------------------------------
st.title("✈️ Tourism Package Prediction")
st.write(
    "Enter the customer details to predict whether "
    "the customer is likely to purchase the tourism package."
)

try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


feature_columns = metadata.get("feature_columns")

if not feature_columns:
    st.error(
        "The metadata file must contain 'feature_columns' "
        "in the same order used during model training."
    )
    st.stop()

# --------------------------------------------------
# Collect customer inputs
# --------------------------------------------------
with st.form("customer_details"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30
        )
        contact_type = st.selectbox(
            "Type of Contact",
            ["Company Invited", "Self Inquiry"],
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Freelancer",
            ],
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced", "Unmarried"],
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        )

    with col2:
        persons_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1, max_value=20, value=2
        )
        property_star = st.selectbox(
            "Preferred Property Star", [3, 4, 5]
        )
        number_of_trips = st.number_input(
            "Number of Trips", min_value=0, max_value=100, value=2
        )
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
        children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0, max_value=20, value=0
        )
        monthly_income = st.number_input(
            "Monthly Income", min_value=0.0,
            value=30000.0, step=1000.0
        )

    submitted = st.form_submit_button(
        "Predict Tourism Package Purchase",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Build dataframe and predict
# --------------------------------------------------
if submitted:
    input_values = {
        "Unnamed: 0": 0,
        "Age": age,
        "TypeofContact": contact_type,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "Designation": designation,
        "NumberOfPersonVisiting": persons_visiting,
        "PreferredPropertyStar": property_star,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children_visiting,
        "MonthlyIncome": monthly_income,
    }

    missing_features = [
        column for column in feature_columns
        if column not in input_values
    ]

    if missing_features:
        st.error(f"Missing input fields: {missing_features}")
        st.stop()

    input_df = pd.DataFrame(
        [[input_values[column] for column in feature_columns]],
        columns=feature_columns,
    )

    st.subheader("Customer Input Data")
    st.dataframe(input_df, use_container_width=True)

    try:
        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The customer is likely to purchase the tourism package."
            )
        else:
            st.info(
                "The customer is unlikely to purchase the tourism package."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )
                st.metric(
                    "Estimated Purchase Probability",
                    f"{probability:.1%}",
                )
                st.progress(probability)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
