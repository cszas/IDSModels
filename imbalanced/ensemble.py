import os
import pandas 					as pd
import numpy 					as np
from sklearn.ensemble           import GradientBoostingClassifier
from sklearn.externals          import joblib
from tensorflow.keras.models    import load_model
from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.metrics            import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from time                       import time
from sklearn.preprocessing      import label_binarize
import matplotlib.pyplot        as plt
import seaborn                  as sns
from itertools                  import cycle

t_i = time()

# Plot ROC curves
def plt_roc(result):
    y_ts = label_binarize(result['Real'].values, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_score = label_binarize(result['Pred'].values, classes=[0,1,2,3,4,5,6,7,8,9])
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
    plt.savefig("./img/ROC_ensemble.png", dpi=300)

# Dataframe
df_p = pd.DataFrame(columns=('KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'LSTM'))
X = pd.read_csv("./data/x_test_corr.csv")
y = pd.read_csv("./data/y_test.csv")
X.set_index('id', inplace = True)
y = y['attack_cat']

# Create Ensemble Model
ensemble = GradientBoostingClassifier()

# Method to carry a test
def train_and_test():

    df_result = pd.DataFrame(columns=('KNN', 'DT', 'RF', 'XGB', 'MLP', 'LSTM'))
    result = pd.DataFrame(columns=('Pred', 'Real'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    x_6_train = X_train[['dur', 'proto', 'service', 'sbytes', 'dttl', 'smean']]
    x_6_test = X_test[['dur', 'proto', 'service', 'sbytes', 'dttl', 'smean']]

    result['Real'] = y_test

    print('')
    print('TRAIN')

    models(X_train, x_6_train, df_p)
    print('')
    print('Training the ensemble model.')
    ensemble.fit(df_p, y_train)

    print('')
    print('TEST')

    models(X_test, x_6_test, df_result)

    ensemble_pred = prob_sklearn(ensemble,df_result)
    result['Pred'] = ensemble_pred

    t_f= time()
    print('')
    print('Accuracy for Ensemble - Test:')
    print(accuracy_score(result['Pred'], result['Real']))

    print('')
    print("Confusion Matrix for Ensemble saved.")
    fig = plt.figure(figsize=(11,11))
    c_matrix = confusion_matrix(result['Real'], result['Pred'])
    sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
    plt.title("Confusion Matrix Ensemble")
    fig.savefig("./img/CM_Ensemble.png", dpi=300)
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
    #print(classification_report(result['Real'], result['Pred']))

    print('')
    print('F1-Score for Dynamic Real-Time Classifier:')
    print(f1_score(result['Real'], result['Pred'], average='weighted'))

    print('')
    print('ROC Curves for Dynamic Real-Time Classifier saved.')
    plt_roc(result)

    print('')
    print('Execution time: ', t_f-t_i)


    # Convert results into binary classification
    df_binary = pd.DataFrame(columns=['prediction', 'real'])
    df_binary['real'] = (result['Real'] != 6) + 0
    df_binary['prediction'] = (result['Pred'] != 6) + 0
    bin_acc = accuracy_score(df_binary['prediction'],df_binary['real'])
    print('')
    print("Binary Accuracy for Dynamic Real-Time Classifier: ", bin_acc)
    print('')

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

    result.replace({'Pred': attack_cat},  inplace = True)
    result.replace({'Real': attack_cat},  inplace = True)

    print(result)

# Method to obtain the prediction from the algorithms
def models(X, x6, df_save):
    
    # KKN
    knn = joblib.load('./saved_model/knn_model.pkl')
    knn_pred = prob_sklearn(knn, x6)
    df_save['KNN'] = knn_pred
    print('KNN')

    # SVM
    svm = joblib.load('./saved_model/svm_model.pkl')
    svm_pred = prob_sklearn(svm, x6)
    df_save['SVM'] = svm_pred
    print('SVM')

    # Decision Tree
    dt = joblib.load('./saved_model/dt_model.pkl')
    dt_pred = prob_sklearn(dt, X)
    df_save['DT'] = dt_pred
    print('DT')

    # Random Forest
    rf = joblib.load('./saved_model/rf_model.pkl')
    rf_pred = prob_sklearn(rf, X)
    df_save['RF'] = rf_pred
    print('RF')

    # XGBoost
    xgb = joblib.load('./saved_model/xgb_model.pkl')
    xgb_pred = prob_sklearn(xgb, X)
    df_save['XGB'] = xgb_pred
    print('XGB')

    # MLP
    mlp = load_model('./saved_model/mlp_model.h5')
    mlp_pred = prob_dl(mlp, X)
    df_save['MLP'] = mlp_pred
    print('MLP')

    # LSTM
    lstm = load_model('./saved_model/lstm_model.h5')
    X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))
    lstm_pred = prob_dl(lstm, X)
    df_save['LSTM'] = lstm_pred
    print('LSTM')

# Extract probabilities from SKLEARN Models
def prob_sklearn(model, data):
    pred_prob = model.predict_proba(data)
    type_argmax= []
    for i in range(len(pred_prob)):
        type_argmax.append(np.argmax(pred_prob[i]))
    return type_argmax

# Extract probabilities from Deep Learning Models
def prob_dl(model, data):
    pred = model.predict(data)
    type_argmax= []
    for i in range(len(pred)):
        type_argmax.append(np.argmax(pred[i]))
    return type_argmax

train_and_test()

joblib.dump(ensemble, './saved_model/ensemble_model.pkl')
print('')
print('Model correctly saved.')