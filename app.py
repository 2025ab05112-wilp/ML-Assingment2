import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


# Streamlit Page 
st.set_page_config(page_title="ML Assingment 2", layout="wide")
st.title("Classification Application")

st.markdown("""
Application flow:
- Upload a **CSV dataset**
- Select a **ML model**
- View **Evaluation Metrics**
- Analyze the results using **confusion matrix** and **classification reoprt**
""")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

 if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "fetal_health" not in data.columns:
        st.error("Dataset must contain a 'fetal_health' target column.")
        st.stop()

    X = data.drop(columns=["fetal_health"])
    # Data Preprocessing 
    # Convert 1 -> 0, 2 -> 1 and 3 -> 2
    y = data["fetal_health"] - 1

    # Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Selection
    model_dict = {
        "Logistic Regression ": OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        )
    }

    selected_options = st.multiselect("Select Classification Model's", model_dict.keys())
    #st.write("You selected:", model_names)
    #model = model_dict[model_name]
    
    if selected_options:
        st.write("You selected the following options:")
        # Iterate over the list of selected options
        for model in selected_options:
            st.write(f"- {model}")


