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
uploaded_file = st.file_uploader("**Upload CSV Dataset**", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.markdown("**Dataset Preview**")
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

    modl_name = st.selectbox("**Select Classification Model's**", model_dict)
    
    if modl_name:
        st.markdown("**Evaluation Metrics**")
        results = get_results(modl_name, data)
        
        results_df = pd.DataFrame(results, index=["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]).T.round(4)

        st.table(results_df)

        st.markdown("**Confusion Metrics**")

        fig, ax = plt.subplots(figsize=(2, 1))
        sns.heatmap(
            results[modl_name]["Confusion Matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Suspect", "Pathological"],
            yticklabels=["Normal", "Suspect", "Pathological"],
            ax=ax, annot_kws={"size":8}
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width=400)

        st.markdown("**Classification Report**")
        report = results[modl_name]['Classification Report']
        st.dataframe(pd.DataFrame(report).transpose().round(4))