from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')

X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


param_grid = {'bootstrap': [True, False],
 'max_depth': [5, 7, 9, 11,13],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

param_grid={'bootstrap': False,
 'max_depth': 11,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 2,
 'n_estimators': 1600}
from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)



#Create a RF Classifier
model=RandomForestClassifier(bootstrap=False,max_depth=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=2,n_estimators=1600,random_state=35)
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
halving_cv = HalvingGridSearchCV(
    model, param_grid, scoring='accuracy', n_jobs=-1, min_resources="exhaust", factor=3
)

model.fit(X_train, y_train)

preds=model.predict(X_test)
preds_train=model.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, preds)
accuracy_train=accuracy_score(y_train, preds_train)
probs=model.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)
print(precision_recall_fscore_support(y_test,preds))
plot_roc(y_test,probs[:,1],'ROC Curve - Random Forests')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_test.columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Random Forests Feature Importances')
plt.tight_layout()
plt.show()


results_rf=get_result(model,'rf')
results_rf.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\Random Forests\results_rf.xlsx',index=False)