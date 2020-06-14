import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models                   import Sequential
from keras.layers                   import Dense, Dropout
from keras.layers.embeddings        import Embedding
from tensorflow.keras.callbacks     import EarlyStopping
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection        import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection      import SelectKBest, mutual_info_classif
from sklearn.preprocessing          import label_binarize
from scipy.stats                    import zscore
from time                           import time
from itertools                      import cycle

t_i = time()

# Load of the training set
X_train = pd.read_csv("./data/x_train_corr.csv")
X_test = pd.read_csv("./data/x_test_corr.csv")
y_test = pd.read_csv("./data/y_test.csv")
y_train = pd.read_csv("./data/y_train.csv")
X_train.set_index('id', inplace = True)
X_test.set_index('id', inplace = True)


y_train = y_train['attack_cat']
y_test = y_test['attack_cat']

print('')
print("#################")
print('MODEL TRAINING:')
print("#################")
print('')


df_pred = pd.DataFrame(columns=('Prediction', 'Probability', 'Real Class'))
df_pred['Real Class'] = y_test
y_train_col = y_train
y_train = pd.get_dummies(y_train,prefix="cat")
y_test = pd.get_dummies(y_test,prefix="cat")

# Plot ROC curves
def plt_roc():
    
    y_ts = label_binarize(df_pred['Real Class'].values, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_score = pred
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
    plt.savefig("./img/MLP/ROC_mlp.png", dpi=300)

# Create model with the optimized values
mlp = Sequential()
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(42, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(25, activation='softsign', input_dim=X_train.shape[1]))
mlp.add(Dense(25, activation='softsign'))
mlp.add(Dense(y_train.shape[1], activation='softmax'))
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
mlp.fit(X_train, y_train, epochs=100, verbose=0,callbacks=[monitor], batch_size=64)

# Training accuracy
pred_train = mlp.predict(X_train)
training = []
for i in range(len(pred_train)):
    training.append(np.argmax(pred_train[i]))
print('Accuracy for MLP - Train:')
print(accuracy_score(training, y_train_col))

print('')
print("#################")
print('TEST RESULTS:')
print("#################")
print('')

#Make the prediction for the test and choose the highest probability and type
pred = mlp.predict(X_test)
type_argmax= []
prob_argmax = []
for i in range(len(pred)):
    type_argmax.append(np.argmax(pred[i]))
    prob_argmax.append(pred[i][np.argmax(pred[i])])

df_pred['Prediction'] = type_argmax
df_pred['Probability'] = prob_argmax


print('Accuracy for MLP - Test:')
print(accuracy_score(type_argmax, df_pred['Real Class']))
print('')


print("Confusion Matrix for MLP saved.")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(df_pred['Real Class'], type_argmax)
sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
plt.title("Confusion Matrix MLP")
fig.savefig("./img/MLP/CM_MLP.png", dpi=300)
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


#print('Clasification Report for MLP:')
#print(classification_report(df_pred['Real Class'], type_argmax))

print('')
print('F1-Score for MLP:')
print(f1_score(df_pred['Real Class'], type_argmax, average='weighted'))

print('')
print('ROC Curves for MLP saved.')
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
mlp.save('./saved_model/mlp_model.h5')
print('')
print('Model correctly saved.')
print('')

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
print("Binary Accuracy for MLP Classifier: ", bin_acc)
print('')