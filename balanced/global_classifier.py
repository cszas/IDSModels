import os
import pandas                   as pd
import numpy                    as np
from sklearn.externals          import joblib
from tensorflow.keras.models    import load_model
import matplotlib.pyplot        as plt
import seaborn                  as sns
from itertools                  import cycle
from time                       import time
from sklearn.preprocessing      import label_binarize
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score


t_i = time()

# Plot ROC curves
def plt_roc():
    y_ts = label_binarize(df['real'].values, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_score = label_binarize(df['prediction'].values, classes=[0,1,2,3,4,5,6,7,8,9])
    n_classes = y_ts.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_ts[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_ts.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'violet', 'slategray', 'yellowgreen', 'navy', 'mistyrose', 'mediumseagreen', 'lightcoral'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("./img/ROC_global.png", dpi=300)

# Load data
X_test = pd.read_csv("./data/x_test_corr.csv")
y_test = pd.read_csv("./data/y_test.csv")
X_test.set_index('Unnamed: 0', inplace = True)
y_test = y_test['attack_cat']

# Create dataframes
x_6 = X_test[['sbytes', 'dbytes', 'sload', 'smean', 'dmean', 'ct_srv_dst']]
df_ensemble = pd.DataFrame(columns=['knn', 'dt', 'rf', 'xgb', 'mlp', 'lstm', 'prediction', 'real'])
df = pd.DataFrame(columns=['prediction', 'real'])

# KKN
knn = joblib.load('./saved_model/knn_model.pkl')
knn_pred = knn.predict_proba(x_6)
type_argmax= []
tprs = {
        'KNN':[0.56, 0.47, 0.7, 0.49, 0.73, 0.97, 0.84, 0.78, 0.86, 0.98],
        'DT': [0.53, 0.65, 0.86, 0.54, 0.79, 0.98, 0.90, 0.83, 0.99, 0.99],
        'RF': [0.58, 0.60, 0.89, 0.58, 0.83, 0.98, 0.91, 0.84, 0.99, 0.99],
        'XGB': [0.50, 0.63, 0.82, 0.52, 0.8, 0.98, 0.89, 0.82, 0.99, 0.99],
        'MLP': [0.28, 0.44, 0.69, 0.47, 0.76, 0.97, 0.76, 0.81, 0.90, 0.99],
        'LSTM': [0.56, 0.5, 0.55, 0.52, 0.80, 0.98, 0.78, 0.82, 0.96, 0.99],
}
for i in range(len(knn_pred)):
    if tprs['KNN'][np.argmax(knn_pred[i])] > 0.45:
        type_argmax.append(np.argmax(knn_pred[i]))
    else:
        type_argmax.append(np.nan)
df_ensemble['knn'] = type_argmax

# Decision Tree
dt = joblib.load('./saved_model/dt_model.pkl')
dt_pred = dt.predict_proba(X_test)
type_argmax= []
for i in range(len(dt_pred)):
    if tprs['DT'][np.argmax(dt_pred[i])] > 0.45:
        type_argmax.append(np.argmax(dt_pred[i]))
    else:
        type_argmax.append(np.nan)
df_ensemble['dt'] = type_argmax

# Random Forest
rf = joblib.load('./saved_model/rf_model.pkl')
rf_pred = rf.predict_proba(X_test)
type_argmax= []
for i in range(len(rf_pred)):
    if tprs['RF'][np.argmax(rf_pred[i])] > 0.45:
        type_argmax.append(np.argmax(rf_pred[i]))
    else:
       type_argmax.append(np.nan)
df_ensemble['rf'] = type_argmax

# XGBoost
xgb = joblib.load('./saved_model/xgb_model.pkl')
xgb_pred = xgb.predict_proba(X_test)
type_argmax= []
for i in range(len(xgb_pred)):
    if tprs['XGB'][np.argmax(xgb_pred[i])] > 0.45:
        type_argmax.append(np.argmax(xgb_pred[i]))
    else:
        type_argmax.append(np.nan)
df_ensemble['xgb'] = type_argmax

# MLP
mlp = load_model('./saved_model/mlp_model.h5')
mlp_pred = mlp.predict(X_test)
type_argmax= []
for i in range(len(mlp_pred)):
    if tprs['MLP'][np.argmax(mlp_pred[i])] > 0.45:
        type_argmax.append(np.argmax(mlp_pred[i]))
    else:
        type_argmax.append(np.nan)
df_ensemble['mlp'] = type_argmax

# LSTM
lstm = load_model('./saved_model/lstm_model.h5')
x = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
lstm_pred = lstm.predict(x)
type_argmax= []
for i in range(len(lstm_pred)):
    if tprs['LSTM'][np.argmax(lstm_pred[i])] > 0.45:
        type_argmax.append(np.argmax(lstm_pred[i]))
    else:
        type_argmax.append(np.nan)
df_ensemble['lstm'] = type_argmax

# Fill dataframes with predictions
df_ensemble['prediction'] = df_ensemble[['knn', 'dt', 'rf', 'xgb', 'mlp', 'lstm']].mode(axis=1)
df['prediction'] = df_ensemble['prediction']
df_ensemble['real'] = y_test
df['real'] = y_test

attack_cat = {0 :  "Analysis",
				1 :  "Backdoor",
				2 :  "DoS",
				3 :  "Exploits",
				4 :  "Fuzzers",
				5 :  "Generic",
				6 :  "Normal",
				7 :  "Reconnaissance",
				8 :  "Shellcode",
				9 :  "Worms"}

# Convert attack ID into string
df_ensemble.replace({'prediction': attack_cat},  inplace = True)
df_ensemble.replace({'real': attack_cat},  inplace = True)


t_f= time()

print('')
print('')
print("##########")
print('REMEMBER:')
print("##########")
print('')
print('Label assigned to each attack:')
print('0 :  Analysis')
print('1 :  Backdoor')
print('2 :  DoS')
print('3 :  Exploits')
print('4 :  Fuzzers')
print('5 :  Generic')
print('6 :  Normal')
print('7 :  Reconnaissance')
print('8 :  Shellcode')
print('9 :  Worms')
print('')
print(df_ensemble)
print('')

acc = accuracy_score(df['prediction'],df['real'])
print('')
print("Accuracy for Dynamic Real-Time Classifier: ", acc)
print('')

print('')
print("Confusion Matrix for Dynamic Real-Time Classifier saved.")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(df['real'], df['prediction'])
sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
plt.title("Confusion Matrix Dynamic Real-Time Classifier")
fig.savefig("./img/CM_Global.png", dpi=300)
#print(c_matrix)


# FPR y TPR

FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Recall or True Positive Rate
TPR = TP/(TP+FN)
print('')
print('TPR :')
for i in range (10):
    print('Class ', i, ' :', TPR[i])
# True negative rate
TNR = TN/(TN+FP)
print('') 
print('TNR :')
for i in range (10):
    print('Class ', i, ' :', TNR[i])
# False positive rate
FPR = FP/(FP+TN)
print('')
print('FPR :')
for i in range (10):
    print('Class ', i, ' :', FPR[i])
# False negative rate
FNR = FN/(TP+FN)
print('')
print('FNR :')
for i in range (10):
    print('Class ', i, ' :', FNR[i])


#print('')
#print('Clasification Report for Dynamic Real-Time Classifier:')
#print(classification_report(df_ensemble['real'], df_ensemble['prediction']))

print('')
print('F1-Score for Dynamic Real-Time Classifier:')
print(f1_score(df['real'], df['prediction'], average='weighted'))


print('')
print('ROC Curves for Dynamic Real-Time Classifier saved.')
plt_roc()

print('')
print('Execution time: ', t_f-t_i)

# Convert results into binary classification
df_binary = pd.DataFrame(columns=['prediction', 'real'])
df_binary['real'] = (df['real'] != 6) + 0
df_binary['prediction'] = (df['prediction'] != 6) + 0
print('')
print('Binary Classification: ')
print('0: Normal Traffic')
print('1: Attack')
print('')
print(df_binary)

bin_acc = accuracy_score(df_binary['prediction'],df_binary['real'])
print('')
print("Binary Accuracy for Dynamic Real-Time Classifier: ", bin_acc)
print('')


