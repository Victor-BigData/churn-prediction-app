import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# CONFIGURACIÓN
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict customer churn based on profile data")

# CARGAR DATOS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(csv_path)

# LIMPIEZA
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = df.drop(columns=["customerID"], errors="ignore")

df = pd.get_dummies(df, drop_first=True)

# VARIABLES
X = df.drop("Churn", axis=1)
y = df["Churn"]

# MODELO
model = RandomForestClassifier()
model.fit(X, y)

# SIDEBAR INPUTS
st.sidebar.header("📥 Customer Data")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 50)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# CREAR INPUT
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "InternetService_Fiber optic": [1 if internet == "Fiber optic" else 0],
    "InternetService_No": [1 if internet == "No" else 0],
    "PaymentMethod_Electronic check": [1 if payment == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment == "Mailed check" else 0],
})

# COMPLETAR COLUMNAS
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# PREDICCIÓN
st.subheader("🔍 Prediction")

if st.button("Predict Churn"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{proba:.2%}")

    with col2:
        if pred == 1:
            st.error("⚠️ High risk of churn")
        else:
            st.success("✅ Customer likely to stay")

# FEATURE IMPORTANCE
st.markdown("---")
st.subheader("📊 Top Factors Influencing Churn")

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

st.bar_chart(top_features)

# INSIGHT NEGOCIO
st.markdown("---")
st.subheader("💡 Business Insight")

st.info(
    "Customers with short tenure, month-to-month contracts, and high monthly charges "
    "are more likely to churn. Offering discounts or longer contracts can reduce churn."
)