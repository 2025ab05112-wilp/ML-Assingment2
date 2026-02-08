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

