import xgboost as xgb
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')
X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


param_grid = {'criterion': ['gini', 'entropy'],
             'splitter': ['best', 'random'],
             'max_depth': [3,5,7,9,11],
             'min_samples_split': [3,5],
             'min_samples_leaf': [1,3,5,7,9],
             'max_features': ['auto', 'log2', None]
}

best_params={'criterion': 'gini',
 'max_depth': 9,
 'max_features': None,
 'min_samples_leaf': 5,
 'min_samples_split': 5,
 'splitter': 'best'}


from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)



#Create a Decision Tree Classifier
model=DecisionTreeClassifier(criterion='gini',max_depth=9,max_features=None,min_samples_leaf=5,min_samples_split=5,splitter='best',random_state=1234)

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

plot_roc(y_test,probs[:,1],'ROC Curve - Decision Trees')

results_dt=get_result(model,'dt')
results_dt.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Scripts\Models\Decision Tree\results_dt.xlsx',index=False)
