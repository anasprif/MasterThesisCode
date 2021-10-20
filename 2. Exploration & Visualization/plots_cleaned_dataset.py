#φόρτωση βιβλιοθηκών
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
#φόρτωση δεδομένων
data=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\telecom_churn_cleaned.xlsx')

correct_dtypes=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\dtypes_old.xlsx',index_col=0)

#συνάρτηση για τα περιγραφικά στατιστικά των ποσοτικών μεταβλητών
def descriptive_stats1(x):
    mo=x.mean()
    mo5trimmed=stats.trim_mean(x, 0.05)
    diam=x.median()
    diakim=x.var()
    typ_apokl=x.std()
    elax=x.min()
    megisto=x.max()
    evros=megisto-elax
    inter_range=np.percentile(x, 75)-np.percentile(x, 25)
    loxotita=x.skew()
    kurtotita=x.kurt()
    y=[mo,mo5trimmed,diam,diakim,typ_apokl,elax,megisto,evros,inter_range,loxotita,kurtotita]
    z=pd.DataFrame(y,index=['Μέσος Όρος','5% trimmed Μέσος Όρος','Διάμεσος','Διακύμανση','Τυπική Απόκλιση','Ελάχιστο','Μέγιστο','Εύρος','Ενδοτεταρτημοριακό Εύρος','Ασυμμετρία','Κύρτωση'])
    return z

#συνάρτηση για τα περιγραφικά στατιστικά των κατηγορικών μεταβλητών
def descriptive_stats2(x):
    syxnotites=x.value_counts()
    pososta=syxnotites/syxnotites.sum()
    a8roistikes=pososta.cumsum()
    y={'Συχνότητες':syxnotites, 'Σχετικές Συχνότητες':pososta,'Αθροιστικές Συχνότητες':a8roistikes} 
    z=pd.DataFrame(y)
    return z

#προσδιορισμός των κατηγορικών,ποσοτικών,διατάξιμων μεταβλητών
num_cols=[]
cat_cols=[]
ord_cols=[]
for i in data.columns:
    if correct_dtypes.loc[i,'correct_dtypes'] == 'float64' or correct_dtypes.loc[i,'correct_dtypes'] == 'int64':
        num_cols.append(i)
    else:
        cat_cols.append(i)
    if correct_dtypes.loc[i,'ordinal'] == 1:
        ord_cols.append(i)

#εύρεση των περιγραφικών στατιστικών των ποσοτικών μεταβλητών
descriptive_num=pd.DataFrame()
for i in num_cols:
    descriptive_num[i]=descriptive_stats1(data[i])[0]
descriptive_num.to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Descriptive Num\descriptive_num.xlsx')

#εύρεση των περιγραφικών στατιστικών των ποιοτικών μεταβλητών
for i in cat_cols:
    descriptive_stats2(data[i]).to_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Descriptive Cat\%s.xlsx'%i)
    
        
#Δημιουργία του correlation matrix
num_cols.append('churn')
corr = data[num_cols].corr(method ='spearman')
num_cols.remove('churn')
#heatmap με τις συσχετίσεις των ποσοτικών μεταβλητών
plt.figure(figsize=(20, 20))
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=.2, cmap="GnBu")
plt.rcParams["font.weight"] = "bold" 
plt.rcParams["axes.labelweight"] = "bold"

#ιστογράμματα των numericals
data[num_cols].hist(bins=45, figsize=(20, 70), layout=(20, 4))

#barplots
cat_cols.remove('area')
cat_cols.remove('ethnic')
num_plots = len(cat_cols)
total_cols = 2
total_rows = num_plots//total_cols + 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(8*total_cols, 6*total_rows), constrained_layout=True)
for i, var in enumerate(cat_cols):
    row = i//total_cols
    pos = i % total_cols
    plt.rcParams["font.weight"] = "bold" 
    plt.rcParams["axes.labelweight"] = "bold"
    plot = sns.countplot(x=var, data=data, ax=axs[row][pos])
    total=100000
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        plot.annotate(percentage, (x, y),ha='center')
plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Barplots\barplots_without_churn.png')
cat_cols.append('area')
cat_cols.append('ethnic')

for i in ['area','ethnic']:
    plt.figure(figsize=(20, 15)) 
    plot=sns.countplot(x=i,data=data)
    if i == 'area':
        plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=0)
    plt.rcParams["font.weight"] = "bold" 
    plt.rcParams["axes.labelweight"] = "bold"
    total=100000
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        plot.annotate(percentage, (x, y),ha='center')
    plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Barplots\%s.png'%i)



#barplots με την churn ως label
for i in cat_cols:
    plt.figure(figsize=(20, 15)) 
    sns.countplot(x=i, hue='churn',data=data)
    if i == 'area':
        plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=0)
    plt.rcParams["font.weight"] = "bold" 
    plt.rcParams["axes.labelweight"] = "bold"
    plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Barplots\%s.png'%i)



num_plots = len(cat_cols)
total_cols = 2
total_rows = num_plots//total_cols + 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(8*total_cols, 6*total_rows), constrained_layout=True)
for i, var in enumerate(cat_cols):
    if var not in ['area','churn']:
        row = i//total_cols
        pos = i % total_cols
        plt.rcParams["font.weight"] = "bold" 
        plt.rcParams["axes.labelweight"] = "bold"
        plot = sns.countplot(x=var, data=data,hue='churn', ax=axs[row][pos])

#boxplots
num_plots = len(num_cols)
total_cols = 2
total_rows = num_plots//total_cols + 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(8*total_cols, 6*total_rows), constrained_layout=True)
for i, var in enumerate(num_cols):
    row = i//total_cols
    pos = i % total_cols
    plt.rcParams["font.weight"] = "bold" 
    plt.rcParams["axes.labelweight"] = "bold"
    plot = sns.boxplot(y=var,data=data,palette='GnBu',ax=axs[row][pos])
plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Boxplots\boxplots_without_churn.png')

#boxplots αριθμητικών με την churn ως label
for j in num_cols:
    f = plt.figure(figsize=[3,6])
    ax = f.add_subplot(111)
    sns.boxplot(x='churn',y=j,data=data,palette='GnBu')
    ax.set_ylabel(ylabel=j,fontsize=13)
    plt.tick_params(axis='y',which='both',labelleft='on',labelright='on')
    ax.yaxis.set_ticks_position('both')
    plt.setp(ax.get_xticklabels(), rotation=90)
    f.tight_layout()
    plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\Boxplots\%s.png'%j)

pp = sns.pairplot(data=data,
                  x_vars=num_cols,
                  y_vars=num_cols,
                  hue='churn')

ps = sns.pairplot(data=data,
                  x_vars=['age1','age2','eqpdays','hnd_price','totmrc_Mean'],
                  y_vars=['age1','age2','eqpdays','hnd_price','totmrc_Mean'],
                  hue='churn')

#kde plots
def kdeplot(feature, hist, kde,df):
    plt.figure(figsize=(9, 4))
    plt.title("Plot for {}".format(feature))
    ax0 = sns.distplot(df[df['churn'] == 0][feature].dropna(), hist=hist, kde=kde, 
             color = 'darkblue',  label='Churn: No',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3})
    ax1 = sns.distplot(df[df['churn'] == 1][feature].dropna(), hist=hist, kde=kde, 
             color = 'orange',  label='Churn: Yes',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3})
    plt.rcParams["font.weight"] = "bold" 
    plt.rcParams["axes.labelweight"] = "bold"
    plt.legend()
#kde plots
for i in num_cols:
    kdeplot(i, hist = False, kde = True,df=data)
    plt.savefig(r'C:\Users\Tasos\Desktop\Diplomatiki\Plots cleaned dataset\KDE Plots\%s.png'%i)

