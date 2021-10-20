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

data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')

X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


param_grid = {"learning_rate": [0.1, 0.01, 0.001],
"gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
"max_depth": [2, 4, 7, 10],
"colsample_bytree": [0.3, 0.6, 0.8, 1.0],
"subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
"reg_alpha": [0, 0.5, 1],
"reg_lambda": [1, 1.5, 2, 3, 4.5],
"min_child_weight": [1, 3, 5, 7],
"n_estimators": [100, 250, 500, 1000]}

param_grid = {
"learning_rate" : [0.1 ],
'max_depth': [6]}


from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

model = xgb.XGBClassifier(learning_rate=0.1,max_depth=6,use_label_encoder=False,random_state=35)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
halving_cv = HalvingGridSearchCV(
    model, param_grid, scoring='accuracy', n_jobs=-1, min_resources="exhaust", factor=3
)
xgb_cv= GridSearchCV(estimator = model,cv=cv, param_grid = param_tuning, n_jobs=-1,scoring='accuracy')
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
plot_roc(y_test,probs[:,1],'ROC Curve - XGBoost')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_test.columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.show()

results_xgb=get_result(model,'xgb')
results_xgb.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\XGBoost\results_xgb.xlsx',index=False)