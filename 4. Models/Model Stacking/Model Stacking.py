from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,roc_curve
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')

X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)
level0 = list()
level0.append(('lr', LogisticRegression(solver='liblinear',C=10,random_state=35)))
level0.append(('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights= 'uniform')))
level0.append(('dtc', DecisionTreeClassifier(criterion='gini',max_depth=9,max_features=None,min_samples_leaf=5,min_samples_split=5,splitter='best',random_state=1234)))
level0.append(('rfc', RandomForestClassifier(bootstrap=False,max_depth=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=2,n_estimators=1600,random_state=35)))
level0.append(('xgb', xgb.XGBClassifier(learning_rate=0.1,max_depth=6,use_label_encoder=False,random_state=35)))
level0.append(('lgb', lgb.LGBMClassifier(learning_rate= 0.05,max_depth= 9,n_estimators= 500,num_leaves= 31,subsample= 0.1,random_state=35)))

# define meta learner model
level1 = LogisticRegression(solver='liblinear',C=10,random_state=35)
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5,n_jobs=-1)

model.fit(X_train, y_train)

preds=model.predict(X_test)
preds_train=model.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train, preds_train)
accuracy=accuracy_score(y_test, preds)
probs=model.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)

plot_roc(y_test,probs[:,1],'ROC Curve - Model Stacking')
plot_confusion_matrix(model, X_test, y_test, values_format='d')
plt.grid(False)
plt.show()