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
import lightgbm as lgb
import pandas as pd
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')
X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


param_grid = {'n_estimators' : [10, 50, 100, 500, 1000, 2000],
              "learning_rate": [0.1, 0.05, 0.025, 0.0125],
              'num_leaves':[21,26,31,36,41,46,51],
              'max_depth': [6, 9,12,15],
              'subsample':[0.1]}

{'learning_rate': 0.05,
 'max_depth': 9,
 'n_estimators': 500,
 'num_leaves': 31,
 'subsample': 0.1}

from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)


model=lgb.LGBMClassifier(learning_rate= 0.05,max_depth= 9,n_estimators= 500,num_leaves= 31,subsample= 0.1,random_state=35)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
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
plot_roc(y_test,probs[:,1],'ROC Curve - Light GBM')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_test.columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Feature Importances')
plt.tight_layout()
plt.show()

results_lgbm=get_result(model,'lgbm')
results_lgbm.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\LightGBM\results_lgbm.xlsx',index=False)
