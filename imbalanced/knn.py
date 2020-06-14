import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection        import GridSearchCV, cross_val_score
from sklearn.preprocessing          import label_binarize
from sklearn.feature_selection      import SelectKBest, mutual_info_classif
from scipy.stats                    import zscore
from sklearn.externals              import joblib
from time                           import time
from itertools                      import cycle

t_i = time()

# Load of the training set
X_train = pd.read_csv("./data/x_train_uncorr.csv")
X_test = pd.read_csv("./data/x_test_uncorr.csv")
y_test = pd.read_csv("./data/y_test.csv")
y_train = pd.read_csv("./data/y_train.csv")
X_train.set_index('id', inplace = True)
X_test.set_index('id', inplace = True)

print('')
print("###############################")
print('SELECTION OF FEATURES - KBEST:')
print("###############################")
print('')
# K-Best:
best=SelectKBest(mutual_info_classif,k=6)
X_best = best.fit_transform(X_train, y_train['attack_cat'])
selected = best.get_support(indices=True)
print('Selected features are: ', X_train.columns[selected])
X_train = X_train[X_train.columns[selected]]
y_train = y_train['attack_cat']
X_test = X_test[X_test.columns[selected]]
y_test = y_test['attack_cat']

# Plot ROC curves
def plt_roc():
    
    y_ts = label_binarize(df_pred['Real Class'].values, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_score = pred_prob
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
    plt.savefig("./img/KNN/ROC_knn.png", dpi=300)


# Set the n_neighbors parameter by cross-validation
def cross_v():
    print('')
    print("#######################################################")
    print('CROSS-VALIDATION FOR HYPERPARAMETER OPTIMIZATION:')
    print("#######################################################")
    print('')
    parameters = {'n_neighbors': np.arange(1, 30)}

    gs = GridSearchCV(KNeighborsClassifier(), parameters)

    gs.fit(X_train, y_train)

    # summarize the results of the grid search
    print("Best score: ", gs.best_estimator_)
    print("Best params: ", gs.best_params_)

#cross_v()


print('')
print("#################")
print('MODEL TRAINING:')
print("#################")
print('')
# Train the model for k = 27 (best result in cross validation)
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(X_train, y_train)

pred_train = knn.predict(X_train)
print('')
print('Accuracy for KNN - Train:')
print(accuracy_score(pred_train, y_train))

print('')
print("#################")
print('TEST RESULTS:')
print("#################")
print('')
#Make the prediction for the test
pred = knn.predict(X_test)


print('')
print('Accuracy for KNN - Test:')
print(accuracy_score(pred, y_test))


print('')
print("Confusion Matrix for KNN saved.")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test, pred)
sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
plt.title("Confusion Matrix KNN")
fig.savefig("./img/KNN/CM_KNN.png", dpi=300)
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
#print('Clasification Report for KNN:')
#print(classification_report(y_test, pred))

print('')
print('F1-Score for KNN:')
print(f1_score(y_test, pred, average='weighted'))

#Choose the highest probability and type for the prediction
pred_prob = knn.predict_proba(X_test)
df_pred = pd.DataFrame(columns=('Prediction', 'Probability', 'Real Class'))
df_pred['Real Class'] = y_test
type_argmax= []
prob_argmax = []

for i in range(len(pred_prob)):
    type_argmax.append(np.argmax(pred_prob[i]))
    prob_argmax.append(pred_prob[i][np.argmax(pred_prob[i])])
df_pred['Prediction'] = type_argmax
df_pred['Probability'] = prob_argmax

print('')
print('ROC Curves for KNN saved.')
plt_roc()

print('')
t_f= time()
print('Execution time: ', t_f-t_i)

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
print(df_pred)

# Convert results into binary classification
df_binary = pd.DataFrame(columns=['prediction', 'real'])
df_binary['real'] = (df_pred['Real Class'] != 6) + 0
df_binary['prediction'] = (type_argmax != 6) + 0
print('')
print('Binary Classification: ')
print('0: Normal Traffic')
print('1: Attack')
print('')
print(df_binary)

bin_acc = accuracy_score(df_binary['prediction'],df_binary['real'])
print('')
print("Binary Accuracy for KNN Classifier: ", bin_acc)
print('')

#Save the model
joblib.dump(knn, './saved_model/knn_model.pkl')
print('')
print('Model correctly saved.')

