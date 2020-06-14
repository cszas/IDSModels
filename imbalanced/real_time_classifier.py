import os
import pandas 					as pd
import numpy 					as np
from kafka                      import KafkaConsumer
from sklearn.externals          import joblib
from tensorflow.keras.models    import load_model
import matplotlib.pyplot        as plt
import seaborn                  as sns
from itertools                  import cycle
from sklearn.metrics            import accuracy_score, f1_score

# Set Kafka Consumer
consumer = KafkaConsumer('Streaming_TFM')

# Build results dataframe
df_pred = pd.DataFrame(columns=('Prediction', 'Probability', 'Model', 'Real Class'))

# Build final prediction dataframe
df_prediction = pd.DataFrame(columns=('Prediction', 'Real Class', 'IsAttack'))

# Load Real Classes
y = pd.read_csv("./data/y_test.csv")
y.set_index('id', inplace = True)
df_pred['Real Class'] =  y['attack_cat']
df_prediction['Real Class'] =  y['attack_cat']

# Metrics
received = []
real = []
probs = []
axis = 0

# Graphic configuration
plt.figure(figsize=(6*3.13,2*3.13))
plt.ion()
plt.show()

# Useful dictionaries and lists
attack_cat = {0 :  "Analysis", 1 :  "Backdoor", 2 :  "DoS", 3 :  "Exploits", 4 :  "Fuzzers", 
            5 :  "Generic", 6 :  "Normal", 7 :  "Reconnaissance", 8 :  "Shellcode", 9 :  "Worms"}

models = ['KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'LSTM']

# Array 10x10 to create confusion matrix
c_matrix = np.zeros((10, 10))

# Extract probability and attack type from sklearn models
def prob_sklearn(model, data):
    pred_prob = model.predict_proba(data)
    result = {'type': np.argmax(pred_prob), 'prob': pred_prob[0][np.argmax(pred_prob)]}
    return result

# Extract probability and attack type from deep learning models
def prob_dl(model, data):
    pred = model.predict(data)
    result = {'type': np.argmax(pred), 'prob': pred[0][np.argmax(pred)]}
    return result

# Dynamic classifier using ensemble method
def ensemble():
    # Load ensemble model
    ensemble = joblib.load('./saved_model/ensemble_model.pkl')
    
    # Create input data (individual predictions)
    df_results = pd.DataFrame(columns=['KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'LSTM'])
    ensemble_list = dict()
    for m in models:
        ensemble_list[m] = results[m]['type']
    df_results = df_results.append(ensemble_list, ignore_index=True)

    # Output the final classification and store it
    ensemble_pred = ensemble.predict(df_results)
    df_prediction.loc[df_prediction.index == id, "Prediction"] = ensemble_pred
    
    # Binary Classification (6: Normal Traffic)
    if ensemble_pred != 6:
        df_prediction.loc[df_prediction.index == id, "IsAttack"] = 1
    else:
        df_prediction.loc[df_prediction.index == id, "IsAttack"] = 0

    # Auxiliar lists to calculate metrics
    received.append(ensemble_pred)
    real.append(int(df_prediction.loc[df_prediction.index == id]['Real Class']))

    # Refresh confusion matrix
    real_class = df_prediction.loc[df_prediction.index == id]['Real Class'].values[0]
    c_matrix[ensemble_pred, real_class] += 1

    # Convert numeric index into Attack Label
    df_prediction.replace({'Prediction': attack_cat},  inplace = True)
    df_prediction.loc[df_prediction.index == id, 'Real Class'] = attack_cat[real_class]

    # Print prediction vs Real Class
    print(df_prediction.loc[df_prediction.index == id])

# Dynamic classifier using voter method
def voter():
    # List with individual predictions (weighted depending on TPR value)
    df_results = pd.DataFrame(columns=['KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'LSTM'])
    ensemble_list = dict()
    for m in models:
        ensemble_list[m] = results[m]['type']
    df_results = df_results.append(ensemble_list, ignore_index=True)

    # Calculate the final prediction (most frequent predicted type)
    df_prediction["Prediction"] = df_results.mode(axis=1)[0][0]

    # Binary Classification (6: Normal Traffic)
    if df_results.mode(axis=1)[0][0] != 6:
        df_prediction.loc[df_prediction.index == id, "IsAttack"] = 1
    else:
        df_prediction.loc[df_prediction.index == id, "IsAttack"] = 0

    # Auxiliar lists to calculate metrics
    received.append(df_results.mode(axis=1)[0][0])
    real.append(df_prediction.loc[df_prediction.index == id]['Real Class'].values[0])

    # Refresh confusion matrix
    real_class = df_prediction.loc[df_prediction.index == id]['Real Class'].values[0]
    c_matrix[df_results.mode(axis=1)[0][0], real_class] += 1

    # Convert numeric index into Attack Label
    df_prediction.replace({'Prediction': attack_cat},  inplace = True)
    df_prediction.loc[df_prediction.index == id, 'Real Class'] = attack_cat[real_class]

    # Print prediction vs Real Class
    print(df_prediction.loc[df_prediction.index == id])

# Receive each row and process it
for msg in consumer:
    # Create data-set with received data
    aux = msg.value.decode('utf-8').rstrip('\n').split(',')
    df = pd.DataFrame(columns=['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports'])
    df.loc[0] = aux
    for c in df.columns:
        df[c] = df[c].astype(float)
    df.set_index('id', inplace = True)
    df_6 = df[['dur', 'proto', 'service', 'sbytes', 'dttl', 'smean']]
    results = dict()
    id = int(df.index.values[0])

    # KKN
    knn = joblib.load('./saved_model/knn_model.pkl')
    knn_pred = prob_sklearn(knn, df_6)
    results['KNN'] = knn_pred

    # SVM
    svm = joblib.load('./saved_model/svm_model.pkl')
    svm_pred = prob_sklearn(svm, df_6)
    results['SVM'] = svm_pred
    
    # Decision Tree
    dt = joblib.load('./saved_model/dt_model.pkl')
    dt_pred = prob_sklearn(dt, df)
    results['DT'] = dt_pred

    # Random Forest
    rf = joblib.load('./saved_model/rf_model.pkl')
    rf_pred = prob_sklearn(rf, df)
    results['RF'] = rf_pred

    # XGBoost
    xgb = joblib.load('./saved_model/xgb_model.pkl')
    xgb_pred = prob_sklearn(xgb, df)
    results['XGB'] = xgb_pred

    # MLP
    mlp = load_model('./saved_model/mlp_model.h5')
    mlp_pred = prob_dl(mlp, df)
    results['MLP'] = mlp_pred

    # LSTM
    lstm = load_model('./saved_model/lstm_model.h5')
    df = np.reshape(df.values, (df.shape[0], 1, df.shape[1]))
    lstm_pred = prob_dl(lstm, df)
    results['LSTM'] = lstm_pred

    print('')
    print('PREDICTIONS: ', results)
    print('')

    # Ensemble
    #ensemble()
    
    # Voter method
    voter()

    # Classifier Accuracy
    r_acc = accuracy_score(received,real)
    print('')
    print("Accuracy ",r_acc)
    print('')
    plt.subplot(1,3,1)
    plt.plot(axis,r_acc,'o')
    plt.xlabel('N Rows')
    plt.ylabel('Accuracy')
    plt.title('Real-time Accuracy')
    plt.pause(0.001)
    print('')


    print('')
    print("Confusion Matrix")
    plt.subplot(1,3,2).remove()
    plt.subplot(1,3,2)
    sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
    plt.title("Confusion Matrix")
    plt.pause(0.001)

    print('')
    print('F1-Score:')
    f1 = f1_score(real, received, average='weighted')
    print(f1)
    plt.subplot(1,3,3)
    plt.plot(axis,f1,'o')
    plt.xlabel('N Rows')
    plt.ylabel('F1-Score')
    plt.title('Real-time F1-Score')
    axis += 1
    plt.pause(0.001)