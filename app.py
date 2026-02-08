import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from model.util import get_results
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

    
    # Model Selection
    model_dict = [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Gaussian Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    selected_options = st.multiselect("Select Classification Model's", model_dict)
    
    if selected_options:
        st.subheader("Evaluation Metrics")
        results = get_results(selected_options, data)

        results_df = pd.DataFrame(results, index=["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]).T.round(4)

        st.table(results_df)
        # Iterate over the list of selected options
        # for model_name in selected_options:
        #     model = model_dict[model_name]
        #     y_pred, y_prob, y_test_bin = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        #     metrics = evalualte_metric(y_test, y_pred, y_prob, y_test_bin)
        #     #st.write(metrics)
        #     col1, col2, col3, col4, col5, col6, col7 = st.columns(6)
        #     col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        #     col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        #     col2.metric("Precision", f"{metrics['precision']:.4f}")
        #     col3.metric("Recall", f"{metrics['recall']:.4f}")
        #     col4.metric("F1 Score", f"{metrics['f1']:.4f}")
        #     col5.metric("AUC", f"{metrics['auc']:.4f}")
        #     col6.metric("MCC", f"{metrics['mcc']:.4f}")



