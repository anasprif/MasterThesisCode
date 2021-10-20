import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
results=pd.read_excel(r'C:\Users\Tasos\Desktop\Diplomatiki\final_results.xlsx')


for i,col in enumerate(['AUC', 'Accuracy', 'Precision',  'Recall',  'F1']):
  ax = sns.barplot(x=col, y="Model", data=results.sort_values(by=col, ascending=False))
  ax.set_xlim(35,results[col].max()+1)
  plt.show()
