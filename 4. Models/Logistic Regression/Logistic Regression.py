# MultiOutputClassifier with XGboost
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')
X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
param_grid = {
    'solver' : ['liblinear'],
    'C' : [10]}

model = linear_model.LogisticRegression(solver='liblinear',C=10,random_state=35)
halving_cv = HalvingGridSearchCV(
    model, param_grid, scoring='accuracy', n_jobs=-1, min_resources="exhaust", factor=3
)
model.fit(X_train, y_train)

preds=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, preds)
probs=model.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)
print(precision_recall_fscore_support(y_test,preds))
plot_roc(y_test,probs[:,1],'ROC Curve - Logistic Regression')

results_lr=get_result(model,'lr')
results_lr.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\Logistic Regression\results_lr.xlsx',index=False)
    
