from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import plot_confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def get_result(model,model_name):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  # Evaluation based on a 10-fold cross-validation
  scores = cross_validate(model, X_train, y_train, 
                        scoring=['accuracy','precision','recall','f1','roc_auc'], cv=10,return_train_score=True)

  df_scores = pd.DataFrame(scores, index = range(1,11))
  cv_acc_train=df_scores['train_accuracy'].mean()
  cv_acc_test=df_scores['test_accuracy'].mean()
  cv_prec_test=df_scores['test_precision'].mean()
  cv_recall_test=df_scores['test_recall'].mean()
  cv_f1_test=df_scores['test_f1'].mean()
  cv_auc_test=df_scores['test_roc_auc'].mean()
  
  # accuracy scores
  print('Average (CV=10), Training Set: ', cv_acc_train)
  print('Average Accuracy (CV=10), Test Set:', cv_acc_test)
  print('Average Precisiom (CV=10), Test Set:', cv_prec_test)
  print('Average Recall (CV=10), Test Set:', cv_recall_test)
  print('Average F1 (CV=10), Test Set:', cv_f1_test)
  print('Average AUC (CV=10), Test Set:', cv_auc_test)
  train_preds=model.predict(X_train)
  test_preds=model.predict(X_test)
  train_acc = accuracy_score(y_train,train_preds)
  test_acc = accuracy_score(y_test,test_preds)
  print('Accuracy of train set',train_acc)
  print('Accuracy of test set',test_acc)
  probs=model.predict_proba(X_test)
  auc=roc_auc_score(y_test,probs[:,1])
  print('AUC of test set',auc)
  # Plot Confusion Matrix
  plot_confusion_matrix(model, X_test, y_test, values_format='d')
  plt.grid(False)
  plt.show()
  df_eval = pd.DataFrame(data={'model':[model_name], 
                              'cv_acc_train':[cv_acc_train],
                              'cv_acc_test':[cv_acc_test],
                              'cv_prec_test':[cv_prec_test],
                              'cv_recall_test':[cv_recall_test],
                              'cv_f1_test':[cv_f1_test],
                              'cv_auc_test':[cv_auc_test],
                              'acc_train':[train_acc],
                              'acc_test':[test_acc],
                              'auc_test':[auc]})
  return df_eval
  
  # print classification report
  print(classification_report(y_test, test_preds, zero_division=0))

def plot_roc(y_true, y_pred, title):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #print(fpr, tpr, prec, rec)
    plt.plot(fpr, tpr)
    plt.plot(fpr,tpr,linestyle = "dotted",
             color = "royalblue",linewidth = 2,
             label = "AUC = " + str(np.around(roc_auc_score(y_true,y_pred),3)))
    plt.legend(loc='best')
    plt.plot([0,1], [0,1])
    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid(b=True, which='both')
    plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
