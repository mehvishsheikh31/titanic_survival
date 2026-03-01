import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("titanic_rf_pipeline.pkl")

# ----------------------------
# Header
# ----------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    "Predict whether a passenger would survive the Titanic disaster "
    "using a trained Machine Learning model."
)

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)

with col2:
    fare = st.number_input("Fare", min_value=0.0, value=15.0)
    sibsp = st.number_input("Siblings / Spouses", min_value=0, value=0)
    parch = st.number_input("Parents / Children", min_value=0, value=0)

embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

st.divider()

# ----------------------------
# Feature Engineering
# ----------------------------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict Survival", use_container_width=True):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
        "FamilySize": [family_size],
        "IsAlone": [is_alone]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability:.2%})")

    st.progress(probability)

st.divider()

st.caption("Built using Scikit-learn Pipeline and Streamlit.")
