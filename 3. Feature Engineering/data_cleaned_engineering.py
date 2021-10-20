#φόρτωση βιβλιοθηκών
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

#φόρτωση δεδομένων
data_cleaned=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned.xlsx')

correct_dtypes=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\dtypes_old.xlsx',index_col=0)

#προσδιορισμός των κατηγορικών,ποσοτικών,διατάξιμων μεταβλητών
num_cols=[]
cat_cols=[]
ord_cols=[]
for i in data_cleaned.columns:
    if correct_dtypes.loc[i,'correct_dtypes'] == 'float64' or correct_dtypes.loc[i,'correct_dtypes'] == 'int64':
        num_cols.append(i)
    else:
        cat_cols.append(i)
    if correct_dtypes.loc[i,'ordinal'] == 1:
        ord_cols.append(i)

'''αντικατάσταση των ακραίων τιμών με τη διάμεσο της κάθε μεταβλητής
for variable in num_cols: 
    q1 = data_cleaned[variable].quantile([0.25]).values[0]
    q3 = data_cleaned[variable].quantile([0.75]).values[0]
    median = float(data_cleaned[variable].median())
    data_cleaned.loc[data_cleaned[variable]>(q3 + 1.5*(q3-q1)), variable] = np.nan
    data_cleaned.loc[data_cleaned[variable]<(q1 - 1.5*(q3-q1)), variable] =np.nan
    data_cleaned.fillna(median,inplace=True)'''

#χρήση της συνάρτησης log(x) για τον μετασχηματισμό των μεταβλητών με skewness>0.5
for i in num_cols:
    if abs(data_cleaned[i].skew())>0.5:
        if data_cleaned[i].min()<0:
            data_cleaned['%s_log'%i]=np.log(data_cleaned[i]-data_cleaned[i].min()+0.000001)
        elif data_cleaned[i].min()==0:
            data_cleaned['%s_log'%i]=np.log(data_cleaned[i]+0.000001)
        else:
            data_cleaned['%s_log'%i]=np.log(data_cleaned[i])

#χρήση της συνάρτησης sqrt(x) για τον μετασχηματισμό των μεταβλητών με skewness>0.5
for i in num_cols:
    if abs(data_cleaned[i].skew())>0.5:
        if data_cleaned[i].min()<0:
            data_cleaned['%s_sqrt'%i]=np.sqrt(data_cleaned[i]-data_cleaned[i].min())
        else:
            data_cleaned['%s_sqrt'%i]=np.sqrt(data_cleaned[i])

#σύγκριση των 2 μετασχηματισμών και παραμονή στην ανάλυση του μετασχηματισμού με το μικρότερο skewness
skewness=data_cleaned.skew()
transformed_cols=[]
for i in num_cols:
    if abs(data_cleaned[i].skew())>0.5:
        if abs(skewness['%s_log'%i])<=abs(skewness['%s_sqrt'%i]):
            transformed_cols.append('%s_log'%i)
        else:
            transformed_cols.append('%s_sqrt'%i)
            
num_cols_final=transformed_cols.copy()
for i in num_cols:
    if abs(data_cleaned[i].skew())<=0.5:
        num_cols_final.append(i)

#supervised encoding για τις κατηγορικές μεταβλητές
cat_cols.remove('churn')
for i in cat_cols:
    odds=pd.DataFrame(data_cleaned[[i,'churn']].groupby(i).sum())
    odds['cat_count']=data_cleaned[[i,'churn']].groupby(i).count()['churn']
    odds['rate']=odds['churn']/odds['cat_count']
    odds['odds']=odds['rate']/(1-odds['rate'])
    for j in odds.index:
        data_cleaned[i]=data_cleaned[i].replace(j,odds.loc[j,'odds'])
    odds=pd.DataFrame()

data_cleaned[num_cols_final+cat_cols+['churn']].to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned_transformed.xlsx',index=False)
