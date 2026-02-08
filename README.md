**Problem Statement:**

Classify fetal health in order to prevent child and maternal mortality.

**Dataset description:**

Dataset link: https://www.kaggle.com/code/fahim6205/fetal-health-classification

This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:

    1. Normal
    2. Suspect
    3. Pathological

**Metrics Comparison**

 
| Model                  | Accuracy | AUC      | Precision | Recall  | F1 Score |   MCC   |
| -----------------------| ---------| ---------|-----------|---------|----------|---------|
| Logistic Regression    | 0.8756   | 0.9565   | 0.8794    | 0.8756  | 0.8751   | 0.6520  |
| Decision Tree          | 0.9014   | 0.8580   | 0.8983    | 0.9014  | 0.8994   | 0.7252  |
| K-Nearest Neighbors    | 0.8709   | 0.9393   | 0.8615    | 0.8709  | 0.8629   | 0.6191  |
| Gaussian Naive Bayes   | 0.8099   | 0.8759   | 0.8611    | 0.8099  | 0.8254   | 0.5737  |	
| Random Forest          | 0.9272   | 0.9792   | 0.9246    | 0.9272  | 0.9243   | 0.7942  |	
| XGBoost                | 0.9413   | 0.9834   | 0.9395    | 0.9413  | 0.9393   | 0.8352  |	

**Observation**

 
| Model                  | Observation                                                                    |
| -----------------------|--------------------------------------------------------------------------------|
| Logistic Regression    | Lower MCC indicates limited performance in handling large datasets             |
| Decision Tree          | Lower AUC indicates weaker generalization and overfitting                      |
| K-Nearest Neighbors    | Lower Precision and F1 indicates sensitivity to ouliers                        |
| Gaussian Naive Bayes   | Lower MCC and Accuracy inidictes limited performance in handling large datasets|
| Random Forest          | Higher values for all metrics indicates generalization and stablity            |
| XGBoost                | Overall best performer                                                         |
