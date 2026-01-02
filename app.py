# ==============================
# IMPORT LIBRARIES
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ==============================
# STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==============================
st.set_page_config(
    page_title="Insurance Cost Prediction",
    layout="wide"
)


# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()


# ==============================
# APP TITLE
# ==============================
st.title("💰 Insurance Cost Prediction Project")
st.write("Complete ML project with EDA, model training, evaluation & prediction")


# ==============================
# DATASET OVERVIEW
# ==============================
st.header("📊 Dataset Overview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.text(df.info())

st.subheader("Statistical Summary")
st.dataframe(df.describe())


# ==============================
# EDA & VISUALIZATION (FIXED FOR STREAMLIT CLOUD)
# ==============================
st.header("📈 Exploratory Data Analysis")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# BMI vs Charges
st.subheader("BMI vs Charges (Smoker Highlighted)")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, ax=ax)
st.pyplot(fig)

# Charges Distribution
st.subheader("Charges Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df["charges"], bins=30, kde=True, ax=ax)
st.pyplot(fig)


# ==============================
# PREPROCESSING (ENCODING)
# ==============================
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])


# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop("charges", axis=1)
y = df["charges"]


# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# TRAIN MODELS
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }


# ==============================
# MODEL COMPARISON
# ==============================
st.header("📌 Model Performance Comparison")
results_df = pd.DataFrame(results).T
st.dataframe(results_df)


# ==============================
# FINAL MODEL (BEST)
# ==============================
final_model = models["Random Forest"]


# ==============================
# FINAL MODEL EVALUATION
# ==============================
y_pred_final = final_model.predict(X_test)

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

st.header("✅ Final Model Performance")
st.write(f"**R2 Score:** {r2:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([[
        age,
        le_sex.transform([sex])[0],
        bmi,
        children,
        le_smoker.transform([smoker])[0],
        le_region.transform([region])[0]
    ]], columns=X.columns)

    prediction = final_model.predict(input_df)
    return round(prediction[0], 2)


# ==============================
# STREAMLIT PREDICTION UI
# ==============================
st.header("🧮 Predict Insurance Cost")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("🔮 Predict Insurance Cost"):
    cost = predict_insurance_cost(age, sex, bmi, children, smoker, region)
    st.success(f"💵 Predicted Insurance Cost: ₹ {cost}")
