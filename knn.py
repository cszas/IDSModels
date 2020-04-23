import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection        import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing          import label_binarize
from sklearn.feature_selection      import SelectKBest, mutual_info_classif
from scipy.stats                    import zscore
from sklearn.externals              import joblib
from time                           import time
from itertools                      import cycle

t_i = time()

# Load of the training set
df = pd.read_csv("./data/UNSW_uncorr.csv")
df.set_index('id', inplace = True)

print('')
print("###############################")
print('SELECTION OF FEATURES - KBEST:')
print("###############################")
print('')
# K-Best:
best=SelectKBest(mutual_info_classif,k=6)
X = df.drop('attack_cat',1)
y = df['attack_cat']
X_best = best.fit_transform(X, y)
selected = best.get_support(indices=True)
print('Selected features are: ', X.columns[selected])
X = df[X.columns[selected]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

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
    tuned_parameters = [{'n_neighbors': np.arange(1, 30)}]
                    
    scores = ['recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # cv = the fold of the cross-validation cv, defaulted to 5
        gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10, scoring='%s_weighted' % score)
        gs.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(gs.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']

        for mean_score, std_score, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, std_score * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, gs.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
#cross_v()


print('')
print("#################")
print('MODEL TRAINING:')
print("#################")
print('')
# Train the model for k = 9 (best result in cross validation)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

# Create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)
scores = cross_val_score(knn, X_train, y_train, cv=cv)
from scipy.stats import sem
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print('K-Fold result:')
print(mean_score(scores))

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

#Save the model
joblib.dump(knn, './saved_model/knn_model.pkl')
print('')
print('Model correctly saved.')
