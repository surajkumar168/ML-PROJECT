import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Wine Quality Analysis - Naive Bayes",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction using Naive Bayes Classifier")
st.markdown("This application predicts wine quality for both **Red and White wines**.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_alcohol.csv")

    # Clean Type column
    df['Type'] = df['Type'].str.strip()
    df['Type'] = df['Type'].replace({'White Wine': 1, 'Red Wine': 0})

    return df

try:
    df = load_data()
    st.success("✅ Data Loaded Successfully!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Exploratory Analysis", "Model Performance", "Predict Wine Quality"]
)

# =========================================================
# 1️⃣ DATA OVERVIEW
# =========================================================
if page == "Data Overview":

    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    with col2:
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes,
            "Non-Null Count": df.count(),
            "Unique Values": df.nunique()
        })
        st.dataframe(info_df)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Quality Distribution")
    fig = px.histogram(df, x="quality", color="Type",
                       labels={"Type": "Wine Type (1=White, 0=Red)"})
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2️⃣ EXPLORATORY ANALYSIS
# =========================================================
elif page == "Exploratory Analysis":

    st.header("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_cols = st.multiselect("Select Columns", numeric_cols,
                                   default=['alcohol', 'quality'])

    if selected_cols:
        for col in selected_cols:
            if col != "quality":
                fig, ax = plt.subplots(figsize=(8,4))
                sns.boxplot(data=df, x="quality", y=col)
                st.pyplot(fig)
                plt.close()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)
    plt.close()

# =========================================================
# 3️⃣ MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.header("Naive Bayes Model Performance")

    # Correct feature selection
    X = df.drop("quality", axis=1)
    y = df["quality"]

    test_size = st.slider("Test Size", 0.1, 0.4, 0.3)
    random_state = st.number_input("Random State", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB()
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Comparison Graph
    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df["Model"],
            y=results_df[metric]
        ))

    fig.update_layout(barmode="group", yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    selected_model = st.selectbox("Select Model", list(models.keys()))
    cm = confusion_matrix(y_test, predictions[selected_model])

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)
    plt.close()

# =========================================================
# 4️⃣ PREDICTION PAGE
# =========================================================
elif page == "Predict Wine Quality":

    st.header("Predict Wine Quality")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    model = GaussianNB()
    model.fit(X, y)

    input_data = []

    col1, col2 = st.columns(2)
    features = X.columns.tolist()

    with col1:
        for feature in features[:len(features)//2]:
            value = st.number_input(feature,
                                    float(df[feature].mean()),
                                    step=0.1)
            input_data.append(value)

    with col2:
        for feature in features[len(features)//2:]:
            value = st.number_input(feature,
                                    float(df[feature].mean()),
                                    step=0.1)
            input_data.append(value)

    if st.button("Predict"):
        prediction = model.predict([input_data])[0]
        probabilities = model.predict_proba([input_data])[0]

        st.success(f"Predicted Wine Quality: {prediction}")

        proba_df = pd.DataFrame({
            "Quality": model.classes_,
            "Probability": probabilities
        })

        fig = px.bar(proba_df, x="Quality", y="Probability")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Use the navigation menu to explore different sections.")
