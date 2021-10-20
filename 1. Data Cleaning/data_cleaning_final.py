#φόρτωση βιβλιοθηκών
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency ,f_oneway,pointbiserialr

#φόρτωση δεδομένων
data=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn.xlsx')
data_copy=data.copy()
correct_dtypes=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\dtypes_old.xlsx',index_col=0)

#αφαίρεση της μεταβλητής csa από την ανάλυση
data=data.drop(columns=['csa'])

#εύρεση των ποσοτικών μεταβλητών
num_cols=[]
for i in data.columns:
    if (correct_dtypes.loc[i,'correct_dtypes'] == 'float64' or correct_dtypes.loc[i,'correct_dtypes'] == 'int64') and i!='Customer_ID':
        num_cols.append(i)

#εύρεση του αριθμού των ελλειπουσών τιμών του συνόλου δεδομένων
nulls_num=data.isna().sum()

#διόρθωση πολυσυγγραμικότητας
corr_df = data[num_cols].corr(method ='spearman').abs()
components = list()
visited = set()
for col in corr_df.columns:
    if col in visited:
        continue

    component = set([col, ])
    just_visited = [col, ]
    visited.add(col)
    while just_visited:
        c = just_visited.pop(0)
        for idx, val in corr_df[c].items():
            if abs(val) > 0.9 and idx not in visited:
                just_visited.append(idx)
                visited.add(idx)
                component.add(idx)
    components.append(list(component))
    
comp_ov1=[]
for i in components:
    if len(i)!=1:
        comp_ov1.append(i.copy())

corr_point_b=[]
for i in num_cols:
    point_b_df=data[[i,'churn']].dropna()
    corr, p = pointbiserialr(point_b_df[i],point_b_df['churn'])
    corr_point_b.append(corr)
s=np.array([abs(ele) for ele in corr_point_b]).mean()
corr_point_b_df=pd.DataFrame(corr_point_b,index=num_cols)
corr_point_b_df=corr_point_b_df.rename(columns={0:'churn_corr'})

nulls_num_cols=data[num_cols].isna().sum()
comp_sel=corr_point_b_df.copy()
comp_sel=comp_sel.abs()
comp_sel['nulls_num']=nulls_num_cols.copy()
col_to_stay=[]
for i in comp_ov1:
    col_to_stay.append(comp_sel.loc[i,:].sort_values(['nulls_num', 'churn_corr'], ascending=[True, False]).index[0])

n=0
for i in comp_ov1:
    comp_ov1[n].remove(comp_sel.loc[i,:].sort_values(['nulls_num', 'churn_corr'], ascending=[True, False]).index[0])
    n=n+1

cols_to_drop=[]
for i in comp_ov1:
    for j in i:
        cols_to_drop.append(j)

data=data.drop(columns=cols_to_drop)

#μετατροπή σε binary των κατηγορικών μεταβλητών με 1 κατηγορία
data['wrkwoman']=data['wrkwoman'].replace('Y', 1)
data['wrkwoman']=data['wrkwoman'].fillna(0)
data['pcowner']=data['pcowner'].replace('Y', 1)
data['pcowner']=data['pcowner'].fillna(0)
data['mailresp']=data['mailresp'].replace('R', 1)
data['mailresp']=data['mailresp'].fillna(0)
data['mailordr']=data['mailordr'].replace('B', 1)
data['mailordr']=data['mailordr'].fillna(0)
data[['wrkwoman','pcowner','mailresp','mailordr']]=data[['wrkwoman','pcowner','mailresp','mailordr']].astype('int64')

#αντικατάσταση για την μεταβλητή hnd_webcap των ελλειπουσών τιμών με την κατηγορία NWC και των μη γνωστών τιμών με την επικρατέστερη κατηγορία(WCMB)
data['hnd_webcap']=data['hnd_webcap'].fillna('NWC')
data['hnd_webcap']=data['hnd_webcap'].replace('UNKW','WCMB')

#ομαδοποίηση των δεδομένων για την μεταβλητή income
data['income']=data['income'].replace([1,2,3],1)
data['income']=data['income'].replace([4,5],2)
data['income']=data['income'].replace([6,7],3)
data['income']=data['income'].replace([8,9],4)

#ομαδοποίηση των δεδομένων για την μεταβλητή lor
data['lor']=data['lor'].replace([0,1,2,3],0)
data['lor']=data['lor'].replace([4,5,6,7],1)
data['lor']=data['lor'].replace([8,9,10,11],2)
data['lor']=data['lor'].replace([12,13,14,15],3)

#μετατροπή της μεταβλητής pre_hnd_price σε binary και μετονομασία της σε pre_hnd_buy
s=[]
for i in data['pre_hnd_price']:
    if pd.isna(i):
        s.append(0)
    else:
        s.append(1)
data['pre_hnd_price']=s
data = data.rename(columns={'pre_hnd_price': 'pre_hnd_buy'})

#αντικατάσταση των ελλειπουσών τιμών με την τιμη 0 για τις μεταβλητές rmrev, rmcalls, REF_QTY, crtcount
data['rmrev']=data['rmrev'].fillna(0)
data['rmcalls']=data['rmcalls'].fillna(0)
data['REF_QTY']=data['REF_QTY'].fillna(0)
data['crtcount']=data['crtcount'].fillna(0)

#μετατροπή της ποσοτικής μεταβλητής tot_acpt σε κατηγορική και μετονομασία της σε offer_acpt
data['offer_acpt']=data['tot_acpt'].copy()
data['offer_acpt']=data['offer_acpt'].fillna('not_approached')
data['offer_acpt']=data['offer_acpt'].replace(0,'accept_0offer')
data['offer_acpt']=data['offer_acpt'].replace([1,2,3,4],'accept_1+offers')
data=data.drop(columns='tot_acpt')

#ομαδοποίηση για την μεταβλητή last_swap
for i in data['last_swap']:
    if isinstance(i,str)==True:
        data['last_swap']=data['last_swap'].replace(i, datetime.strptime(i,'%m/%d/%Y'))
s=[]
for i in data['last_swap']:
    if i in pd.date_range(start='1997-03-20', end='1997-12-31'):
        s.append(5)
    elif i in pd.date_range(start='1998-01-01', end='1998-12-31'):
        s.append(4)
    elif i in pd.date_range(start='1999-01-01', end='1999-12-31'):
        s.append(3)
    elif i in pd.date_range(start='2000-01-01', end='2000-12-31'):
        s.append(2)
    elif i in pd.date_range(start='2001-01-01', end='2002-01-06'):
        s.append(1)
    else:
        s.append(0)
data['last_swap']=s

#ομαδοποίηση για την μεταβλητή crclscod
s=[]
for i in data['crclscod']:
    if i.startswith('A'):
        s.append(1)
    elif i.startswith('B'):
        s.append(2)
    elif i.startswith('C'):
        s.append(3)
    elif i.startswith('D'):
        s.append(4)
    elif i.startswith('E'):
        s.append(5)
    elif pd.isna(i):
        s.append(np.nan)
    else:
        s.append(6)
data['crclscod']=s

#ομαδοποίηση για τη μεταβλητή retdays και μετονομασία της σε retdays_new
s=[]
for i in data['retdays']:
    if i <=365:
        s.append('0-1_year')
    elif i <=730 and i>365:
        s.append('1-2_years')
    elif i >730:
        s.append('2-3_years')
    else:
        s.append('not_approached')
data['retdays_new']=s
data=data.drop(columns='retdays')


data['tot_ret']=data['tot_ret'].fillna(0)

#ελλείπουσες τιμές - γραμμές
nan_rows=data.isnull().sum(axis=1).value_counts()
nan_rows=nan_rows[nan_rows.index.sort_values()]

#πλήθος κατηγοριών για τις κατηγορικές μεταβλητές
obj_ind=[]
cnt_un_obj=[]
for i in data.columns:
        if correct_dtypes.loc[i,'correct_dtypes'] == 'object':
            obj_ind.append(i)
            cnt_un_obj.append(len(data[i].value_counts()))
objcnt = pd.DataFrame(cnt_un_obj, index = obj_ind)
objcnt=objcnt[0].sort_values(ascending=True)

#δημιουργία dummy μεταβλητών
def one_hot_enc(a,col):
    a=pd.get_dummies(a, columns=col)
    return a

#encoding στις κατηγορικές μεταβλητές(output στήλη)
def label_enc(c,col):
    a=c.copy()
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder()
    a[col] = label_encoder.fit_transform(a[col])
    return a[col]

#φιλτράρισμα μεταβλητών - εύρεση των πιο σημαντικών
def significance(data,columns,target_col):
    from scipy.stats import chi2_contingency 
    dep_cols=[]
    num_cols=[]
    if correct_dtypes.loc[target_col,'correct_dtypes']=='object' and objcnt[target_col]==2:
        for i in columns:
            if correct_dtypes.loc[i,'correct_dtypes']=='object':
                data_crosstab = pd.crosstab(data[i], data[target_col], margins = False) 
                # defining the table 
                stat, p, dof, expected = chi2_contingency(data_crosstab)
                # interpret p-value 
                alpha = 0.05
                if p <= alpha:
                    if i != 'Customer_ID' and i != target_col:
                        dep_cols.append(i)
            else:
                num_cols.append(i)
        corr_point_b=[]
        for i in num_cols:
            point_b_df=data[[i,target_col]].dropna()
            corr, p = pointbiserialr(point_b_df[i],point_b_df[target_col])
            corr_point_b.append(corr)
        s=np.array([abs(ele) for ele in corr_point_b]).mean()
        corr_point_b_df=pd.DataFrame(corr_point_b,index=num_cols)
        corr_point_b_df=corr_point_b_df.rename(columns={0:'correlation'})
        for i in corr_point_b_df.index:
            if abs(corr_point_b_df.loc[i,'correlation'])>s:
                if i != 'Customer_ID' and i != target_col:
                    dep_cols.append(i)
    elif correct_dtypes.loc[target_col,'correct_dtypes']=='object' and objcnt[target_col]>2:
        for i in columns:
            if correct_dtypes.loc[i,'correct_dtypes']=='object':
                data_crosstab = pd.crosstab(data[i], data[target_col], margins = False) 
                # defining the table 
                stat, p, dof, expected = chi2_contingency(data_crosstab) 
                # interpret p-value 
                alpha = 0.05
                if p <= alpha:
                    if i != 'Customer_ID' and i != target_col:
                        dep_cols.append(i)
            else:
                num_cols.append(i)
        for i in num_cols:
            values_per_group = [col for col_name, col in data.groupby(target_col)[i]]
            F, p = f_oneway(*values_per_group)
            # interpret p-value 
            alpha = 0.05
            if p <= alpha:
                if i != 'Customer_ID' and i != target_col:
                    dep_cols.append(i)
    elif correct_dtypes.loc[target_col,'correct_dtypes']=='int64' or correct_dtypes.loc[target_col,'correct_dtypes']=='float64':
        for i in columns:
            if correct_dtypes.loc[i,'correct_dtypes']=='object':
                values_per_group = [col for col_name, col in data.groupby(i)[target_col]]
                F, p = f_oneway(*values_per_group)
                # interpret p-value 
                alpha = 0.05
                if p <= alpha:
                    if i != 'Customer_ID' and i != target_col:
                        dep_cols.append(i)
            else:
                num_cols.append(i)
        correlations=data.corr(method ='spearman')
        correlations=correlations.loc[num_cols,target_col]
        s=abs(correlations).mean()
        for i in correlations.index:
            if abs(correlations[i])>s:
                if i != 'Customer_ID' and i != target_col:
                    dep_cols.append(i)
    return dep_cols
sign_cols=significance(data,data.columns,'churn')

#ελλείπουσες τιμές - γραμμές(σημαντικές μεταλητές)
nan_rows_sign=data[sign_cols].isnull().sum(axis=1).value_counts()
nan_rows_sign=nan_rows_sign[nan_rows_sign.index.sort_values()]

#προσμέτρηση και των κατηγοριών 'Unknown' ως ελλείπουσες τιμές
nulls_num=data.isna().sum()
nulls_num['marital']=nulls_num['marital']+37333
nulls_num['ethnic']=nulls_num['ethnic']+10945
nulls_num['kid0_2']=nulls_num['kid0_2']+94256
nulls_num['car_buy']=nulls_num['car_buy']+56399
nulls_num['dualband']=nulls_num['dualband']+222
nulls_num['new_cell']=nulls_num['new_cell']+66914
nulls_num['kid3_5']=nulls_num['kid3_5']+93572
nulls_num['kid6_10']=nulls_num['kid6_10']+90195
nulls_num['kid11_15']=nulls_num['kid11_15']+89454
nulls_num['kid16_17']=nulls_num['kid16_17']+88304
nulls_num_sign=nulls_num[sign_cols].copy()
nulls_num_sign=nulls_num_sign.sort_values(ascending=False)

data['ethnic']=data['ethnic'].replace('U',np.nan)
data['dualband']=data['dualband'].replace('U',np.nan)

#οι μεταβλητές που δεν έχουν καθόλου ή σχεδόν καθόλου(<=50) ελλείπουσες τιμές (για τις σημαντικές για την churn)
ind_non_nulls=[]
ind_non_nulls_to_repl=[]
ind_nulls=[]
for i in sign_cols:
    if nulls_num[i]<=15000:
        ind_non_nulls.append(i)
        if nulls_num[i]>0:
            ind_non_nulls_to_repl.append(i)
    else:
        ind_nulls.append(i)

#οι σημαντικές μεταβλητές που δεν έχουν καθόλου ή σχεδόν καθόλου(<=50) ελλείπουσες τιμές (για όλες τις μεταβλητές)
ind_non_nulls_all=[]
ind_non_nulls_to_repl_all=[]
ind_nulls_all=[]
for i in data.columns:
    if nulls_num[i]<=15000:
        ind_non_nulls_all.append(i)
        if nulls_num[i]>0:
            ind_non_nulls_to_repl_all.append(i)
    else:
        ind_nulls_all.append(i)



#συνάρτηση για αντικατάσταση των ελλειπουσών τιμών των κατηγορικών μεταβλητών
def categ_cleaning(a):
    unique_values=a.value_counts()
    most_freq_str=unique_values.index[0]
    a=a.replace(np.nan, most_freq_str)
    return a

#συνάρτηση για αντικατάσταση των ελλειπουσών τιμών των ποσοτικών μεταβλητών
def num_cleaning(a,b):
    if b==0:
        mean=a.mean()
        a=a.replace(np.nan, mean)
    elif b==1:
        median=a.median()
        a=a.replace(np.nan, median)
    return a

#αντικατάσταση των ελλειπουσών τιμών για τις μεταβλητές που έχουν <2.000
for i in ind_non_nulls_to_repl_all:
    if correct_dtypes.loc[i,'correct_dtypes'] == 'object':
        data[i]=categ_cleaning(data[i])
    elif correct_dtypes.loc[i,'correct_dtypes'] == 'int64':
        data[i]=num_cleaning(data[i],1)
        data[i]=round(data[i])
    else:
        data[i]=num_cleaning(data[i],1)

#εξαίρεση απο την ανάλυση των μεταβλητών με ελλείπουσες τιμές >35%
for i in ['children','cartype','occu1','div_type','marital','kid0_2','car_buy','HHstatin']:
    if i in sign_cols:
        sign_cols.remove(i)
    if i in ind_nulls:
        ind_nulls.remove(i)
    if i in ind_non_nulls:
        ind_non_nulls.remove(i)
    if i in ind_non_nulls_to_repl:
        ind_non_nulls_to_repl.remove(i)

#εξαίρεση απο την ανάλυση των μεταβλητών με πολλές τιμές στην κατηγορία 'Unknown'
for i in ['Customer_ID','car_buy','kid0_2','kid3_5','kid6_10','kid11_15','kid16_17','marital','new_cell']:
    if i in ind_non_nulls_all:
        ind_non_nulls_all.remove(i)

#συνάρτηση προετοιμασίας των δεδομένων για την πρόβλεψη των ελλειπουσών τιμών
def final_cleaning(fc_data,target_col):
    fc_data_copy=fc_data.copy()
    if 'churn' in ind_non_nulls_all:
        ind_non_nulls_all.remove('churn')
    if correct_dtypes.loc[target_col,'correct_dtypes'] == 'object':
        if objcnt[target_col]>=2:
            targ_enc=pd.DataFrame()
            targ_enc[target_col]=fc_data_copy[target_col].dropna()
            targ_enc[target_col]=label_enc(targ_enc,target_col)
            fc_data_copy[target_col]=targ_enc[target_col].copy()
    sign_cols_all=significance(fc_data_copy,ind_non_nulls_all,target_col)
    sign_cols_all_return=sign_cols_all.copy()
    wp=fc_data_copy[sign_cols_all]
    sign_cols_all.append(target_col)
    w=fc_data_copy[sign_cols_all].dropna()
    X=w.drop(columns=target_col)
    y=w[target_col]
    q1=[]
    q2=[]
    for i in X.columns:
        if correct_dtypes.loc[i,'correct_dtypes'] == 'object' and objcnt[i]>2:
            q1.append(i)
        elif correct_dtypes.loc[i,'correct_dtypes'] == 'object' and objcnt[i]==2:
            q2.append(i)
    X=one_hot_enc(X,q1)
    wp=one_hot_enc(wp,q1)
    for i in q2:
        X[i]=label_enc(X,i)
        wp[i]=label_enc(wp,i)
    return X,y,wp,sign_cols_all_return

#προβλεψη των ελλειπουσών τιμών της μεταβλητής ownrent
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


X,y,wp,z=final_cleaning(data,'ownrent')


X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.25, random_state=35)


param_tuning = {'learning_rate': [0.1],
                'n_estimators' : [200],
                'max_depth': [3],
                'min_child_weight': [5]}

from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

model = xgb.XGBClassifier(use_label_encoder=False)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

xgb_cv= GridSearchCV(estimator = model,cv=cv, param_grid = param_tuning, n_jobs=-1,scoring='balanced_accuracy')
xgb_cv.fit(X_train, y_train,sample_weight=sample_weights)

preds=xgb_cv.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, preds)
probs=xgb_cv.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
preds_ownrent=xgb_cv.predict(wp)
from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)

data['ownrent']=data['ownrent'].mask(pd.isnull, preds_ownrent)
data['ownrent']=data['ownrent'].replace(0,'O')
data['ownrent']=data['ownrent'].replace(1,'R')

#προβλεψη των ελλειπουσών τιμών της μεταβλητής 'dwlltype'
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# define dataset
X,y,wp,z=final_cleaning(data,'dwlltype')


X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.25, random_state=1)


param_tuning = {'learning_rate': [0.1],
                'n_estimators' : [200],
                'max_depth': [3],
                'min_child_weight': [7]}
from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

model = xgb.XGBClassifier(use_label_encoder=False)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

xgb_cv= GridSearchCV(estimator = model,cv=cv, param_grid = param_tuning, n_jobs=-1,scoring='balanced_accuracy')
xgb_cv.fit(X_train, y_train,sample_weight=sample_weights)
preds=xgb_cv.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, preds)
probs=xgb_cv.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
preds_dwlltype=xgb_cv.predict(wp)

from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)

data['dwlltype']=data['dwlltype'].mask(pd.isnull, preds_dwlltype)
data['dwlltype']=data['dwlltype'].replace(0,'M')
data['dwlltype']=data['dwlltype'].replace(1,'S')

#πρόβλεψη των ελλειπουσών τιμών της μεταβλητής 'infobase'
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# define dataset
X,y,wp,z=final_cleaning(data,'infobase')


X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.25, random_state=35)


param_tuning = {'learning_rate': [0.1],
                'n_estimators' : [200]
                }
from sklearn.utils import class_weight
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

model = xgb.XGBClassifier(use_label_encoder=False)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

xgb_cv= GridSearchCV(estimator = model,cv=cv, param_grid = param_tuning, n_jobs=-1,scoring='balanced_accuracy')
xgb_cv.fit(X_train, y_train,sample_weight=sample_weights)
preds=xgb_cv.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, preds)
probs=xgb_cv.predict_proba(X_test)
auc=roc_auc_score(y_test,probs[:,1])
preds_infobase=xgb_cv.predict(wp)

from sklearn.metrics import classification_report
class_report=classification_report(y_test,preds)
print(class_report)

data['infobase']=data['infobase'].mask(pd.isnull, preds_infobase)
data['infobase']=data['infobase'].replace(0,'M')
data['infobase']=data['infobase'].replace(1,'N')

#πρόβλεψη των ελλειπουσών τιμών της μεταβλητής 'income'
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# define dataset
X,y,wp,z=final_cleaning(data,'income')


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=1)

from sklearn.base import clone
class OrdinalClassifier():
    
    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
    
    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
    
    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


xlf=OrdinalClassifier(xgb.XGBClassifier(use_label_encoder=False))
xlf.fit(X_train,y_train)
preds=xlf.predict(X_test)
rmse_income=mean_squared_error(y_test,preds,squared=False)

preds_income=xlf.predict(wp)
preds_income=[i+1 for i in preds_income]
data['income']=data['income'].mask(pd.isnull, preds_income)

#πρόβλεψη των ελλειπουσών τιμών της μεταβλητής 'lor'
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# define dataset
X,y,wp,z=final_cleaning(data,'lor')


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=1)

from sklearn.base import clone
class OrdinalClassifier():
    
    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
    
    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
    
    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


xlf=OrdinalClassifier(xgb.XGBClassifier(use_label_encoder=False))
xlf.fit(X_train,y_train)
preds=xlf.predict(X_test)

rmse_lor=mean_squared_error(y_test,preds,squared=False)
preds_lor=xlf.predict(wp)

data['lor']=data['lor'].mask(pd.isnull, preds_lor)

#export του καθαρισμένου συνόλου δεδομένων
final_cols=sign_cols.copy()
final_cols.append('churn')
data[final_cols].to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned.xlsx',index=False)




