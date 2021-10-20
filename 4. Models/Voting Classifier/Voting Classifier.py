# importing libraries
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx')

X_train, X_test, y_train, y_test = train_test_split( data_cleaned.drop(columns=['churn']),data_cleaned['churn'], test_size=0.23, random_state=35)


# group / ensemble of models
level0 = list()
level0.append(('lr', LogisticRegression(solver='liblinear',C=10,random_state=35)))
level0.append(('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights= 'uniform')))
level0.append(('dtc', DecisionTreeClassifier(criterion='gini',max_depth=9,max_features=None,min_samples_leaf=5,min_samples_split=5,splitter='best',random_state=1234)))
level0.append(('rfc', RandomForestClassifier(bootstrap=False,max_depth=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=2,n_estimators=1600,random_state=35)))
level0.append(('xgb', xgb.XGBClassifier(learning_rate=0.1,max_depth=6,use_label_encoder=False,random_state=35)))
level0.append(('lgb', lgb.LGBMClassifier(learning_rate= 0.05,max_depth= 9,n_estimators= 500,num_leaves= 31,subsample= 0.1,random_state=35)))
level0.append(('gnb', GaussianNB()))
level0.append(('svc', SVC(random_state=35)))

# Voting Classifier with hard voting
vot_hard = VotingClassifier(estimators = level0, voting ='hard')
vot_hard.fit(X_train, y_train)
preds = vot_hard.predict(X_test)
hard_preds=vot_hard.predict(X_test)
hard_preds_train=vot_hard.predict(X_train)
from sklearn.metrics import accuracy_score
hard_accuracy_train=accuracy_score(y_train, hard_preds_train)
hard_accuracy=accuracy_score(y_test, hard_preds)
from sklearn.metrics import classification_report
class_report=classification_report(y_test,hard_preds)
print(class_report)
plot_confusion_matrix(vot_hard, X_test, y_test, values_format='d')
plt.grid(False)
plt.show()

# group / ensemble of models
level0 = list()
level0.append(('lr', LogisticRegression(solver='liblinear',C=10,random_state=35)))
level0.append(('knn', KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights= 'uniform')))
level0.append(('dtc', DecisionTreeClassifier(criterion='gini',max_depth=9,max_features=None,min_samples_leaf=5,min_samples_split=5,splitter='best',random_state=1234)))
level0.append(('rfc', RandomForestClassifier(bootstrap=False,max_depth=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=2,n_estimators=1600,random_state=35)))
level0.append(('xgb', xgb.XGBClassifier(learning_rate=0.1,max_depth=6,use_label_encoder=False,random_state=35)))
level0.append(('lgb', lgb.LGBMClassifier(learning_rate= 0.05,max_depth= 9,n_estimators= 500,num_leaves= 31,subsample= 0.1,random_state=35)))
level0.append(('gnb', GaussianNB()))
level0.append(('svc', SVC(random_state=35,probability=True)))

# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators = level0, voting ='soft')
vot_soft.fit(X_train, y_train)
soft_preds = vot_soft.predict(X_test)
soft_preds_train=vot_soft.predict(X_train)
from sklearn.metrics import accuracy_score
soft_accuracy_train=accuracy_score(y_train, soft_preds_train)
soft_accuracy=accuracy_score(y_test, soft_preds)
probs=vot_soft.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
from sklearn.metrics import classification_report
class_report=classification_report(y_test,soft_preds)
print(class_report)
plot_roc(y_test,probs[:,1],'ROC Curve - Soft Voting Classifier')
plot_confusion_matrix(vot_soft, X_test, y_test, values_format='d')
plt.grid(False)
plt.show()


