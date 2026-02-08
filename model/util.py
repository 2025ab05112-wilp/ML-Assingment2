from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

def get_model(model_name):
    if model_name == "Logistic Regression":
        return OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "K-Nearest Neighbors":
        return  KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Gaussian Naive Bayes":
        return GaussianNB()
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        )


def evalualte_model(model_name, data):

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

    #Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Model Selection
    model = get_model(model_name)
    model.fit(X_train_scaled, y_train)
    
    # Model Prediction
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    y_test_bin = label_binarize(y_test, classes=[0,1,2])

    return {
        "Accuracy":accuracy_score(y_test, y_pred),
        "AUC":roc_auc_score(y_test_bin, y_prob, multi_class="ovr"),
        "Precision":precision_score(y_test, y_pred, average="weighted"),
        "Recall":recall_score(y_test, y_pred, average="weighted"),
        "F1 Score":f1_score(y_test, y_pred, average="weighted"),
        "MCC":matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Pathological"], output_dict=True)
    }

def get_results(model_name, data):
    results = {
        model_name: evalualte_model(model_name, data) #for model_name in models
    }
    return results