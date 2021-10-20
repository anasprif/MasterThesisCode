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
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')
X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


param_grid = {'n_estimators' : [200,400,600,800,1000,1200,1400],
              'learning_rate': [0.1, 0.01, 0.001]}
{'learning_rate': 0.1, 'n_estimators': 1400}

from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)



#Create a ADABoost Classifier
model=AdaBoostClassifier(learning_rate=0.3, n_estimators=500,random_state=35)

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

plot_roc(y_test,probs[:,1],'ROC Curve - ADA Boost')


results_ada=get_results(model,'ada')
results_ada.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\ADABoost\results_ada.xlsx',index=False)