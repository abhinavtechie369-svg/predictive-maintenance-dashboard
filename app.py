# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("data/ai4i2020.csv")


# ==============================
# SELECT FEATURES AND TARGET
# ==============================

X = data[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

y = data["Machine failure"]


# ==============================
# TRAIN MACHINE LEARNING MODEL
# ==============================

model = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)


# ==============================
# SIDEBAR NAVIGATION
# ==============================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset Explorer",
        "EDA Visualizations",
        "Model Performance",
        "Prediction System"
    ]
)


# ==============================
# HOME PAGE
# ==============================

if menu == "Home":

    st.title("AI Predictive Maintenance Dashboard")

    st.write(
        """
        This dashboard analyzes machine sensor data and predicts
        whether a machine is likely to fail.

        Technologies used:
        - Python
        - Pandas
        - Scikit-learn
        - Random Forest
        - Streamlit
        """
    )


# ==============================
# DATASET EXPLORER
# ==============================

elif menu == "Dataset Explorer":

    st.header("Dataset Preview")

    st.write("First 50 rows of the dataset")

    st.dataframe(data.head(50))


# ==============================
# EDA VISUALIZATIONS
# ==============================

elif menu == "EDA Visualizations":

    st.header("Exploratory Data Analysis")

    # Failure distribution
    st.subheader("Machine Failure Distribution")

    fig1, ax1 = plt.subplots()

    sns.countplot(x="Machine failure", data=data, ax=ax1)

    st.pyplot(fig1)

    # Torque distribution
    st.subheader("Torque Distribution")

    fig2, ax2 = plt.subplots()

    sns.histplot(data["Torque [Nm]"], bins=30, ax=ax2)

    st.pyplot(fig2)

    # Correlation heatmap
    st.subheader("Sensor Correlation Heatmap")

    fig3, ax3 = plt.subplots(figsize=(10,6))

    sns.heatmap(data.corr(numeric_only=True), annot=True, ax=ax3)

    st.pyplot(fig3)


# ==============================
# MODEL PERFORMANCE
# ==============================

elif menu == "Model Performance":

    st.header("Model Performance")

    st.write("Model Accuracy:", accuracy)

    st.subheader("Confusion Matrix")

    st.write(cm)

    # Feature importance
    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = X.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance", ascending=False
    )

    st.bar_chart(importance_df.set_index("Feature"))


# ==============================
# PREDICTION SYSTEM
# ==============================

elif menu == "Prediction System":

    st.header("Predict Machine Failure")

    air_temp = st.number_input("Air Temperature (K)", value=300.0)

    process_temp = st.number_input("Process Temperature (K)", value=310.0)

    rpm = st.number_input("Rotational Speed (rpm)", value=1500)

    torque = st.number_input("Torque (Nm)", value=40.0)

    tool_wear = st.number_input("Tool Wear (min)", value=100)

    if st.button("Predict Machine Failure"):

        input_data = [[air_temp, process_temp, rpm, torque, tool_wear]]

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        failure_prob = probability[0][1]

        st.write(
            "Failure Probability:",
            round(failure_prob * 100, 2),
            "%"
        )

        if prediction[0] == 1:
            st.error("⚠ Machine Likely to Fail")
        else:
            st.success("✅ Machine Operating Normally")

        # Create report
        report = pd.DataFrame({
            "Air Temperature": [air_temp],
            "Process Temperature": [process_temp],
            "RPM": [rpm],
            "Torque": [torque],
            "Tool Wear": [tool_wear],
            "Failure Probability": [failure_prob]
        })

        st.download_button(
            label="Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="prediction_report.csv",
            mime="text/csv"
        )