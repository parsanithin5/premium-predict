# ==============================
# Step 1: Import Libraries
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
# Step 2: Load Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Insurance Cost Prediction", layout="wide")
st.title("💰 Insurance Cost Prediction Project")
st.write("Complete ML project with EDA, model training & prediction")

# ==============================
# Step 3: Dataset Preview
# ==============================
st.header("📊 Dataset Overview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.text(df.info())

st.subheader("Statistical Summary")
st.dataframe(df.describe())

# ==============================
# Step 4: EDA & Visualization
# ==============================
st.header("📈 Exploratory Data Analysis")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
st.pyplot(plt.gcf())
plt.clf()

# BMI vs Charges
st.subheader("BMI vs Charges")
plt.figure(figsize=(8, 6))
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
st.pyplot(plt.gcf())
plt.clf()

# Charges Distribution
st.subheader("Charges Distribution")
plt.figure(figsize=(8, 6))
sns.histplot(df["charges"], bins=30, kde=True)
st.pyplot(plt.gcf())
plt.clf()

# ==============================
# Step 5: Encode Categorical Columns
# ==============================
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])

# ==============================
# Step 6: Features & Target
# ==============================
X = df.drop("charges", axis=1)
y = df["charges"]

# ==============================
# Step 7: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Step 8: Train Models
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
# Step 9: Model Comparison
# ==============================
st.header("📌 Model Comparison")
st.dataframe(pd.DataFrame(results).T)

# ==============================
# Step 10: Final Model
# ==============================
final_model = models["Random Forest"]

# ==============================
# Step 11: Prediction Function
# ==============================
def predict_cost(age, sex, bmi, children, smoker, region):
    input_data = np.array([[
        age,
        le_sex.transform([sex])[0],
        bmi,
        children,
        le_smoker.transform([smoker])[0],
        le_region.transform([region])[0]
    ]])
    return round(final_model.predict(input_data)[0], 2)

# ==============================
# Step 12: Prediction UI
# ==============================
st.header("🧮 Predict Insurance Cost")

age = st.number_input("Age", 1, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("🔮 Predict"):
    result = predict_cost(age, sex, bmi, children, smoker, region)
    st.success(f"💵 Predicted Insurance Cost: ₹ {result}")
