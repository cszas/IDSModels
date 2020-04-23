import os
import pandas 					as pd
import numpy 					as np
from kafka                      import KafkaConsumer
from sklearn.externals          import joblib
from tensorflow.keras.models    import load_model


# Set Kafka Consumer
consumer = KafkaConsumer('Streaming_TFM')

# Build results dataframe
df_pred = pd.DataFrame(columns=('Prediction', 'Probability', 'Model', 'Real Class'))
y = pd.read_csv("./data/test_y.csv")
y.set_index('id', inplace = True)
df_pred['Real Class'] =  y['attack_cat']

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

# Find the best prediction from all the models studied
def choose():
    models = ['KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'LSTM']
    compared_prob = []
    for m in models:
        compared_prob.append(results[m]['prob'])

    max_prob = results[models[np.argmax(compared_prob)]]['prob']
    max_model = models[np.argmax(compared_prob)]
    max_class = results[models[np.argmax(compared_prob)]]['type']
    df_pred.loc[df_pred.index == id, "Probability"] = max_prob
    df_pred.loc[df_pred.index == id, "Model"] = max_model
    df_pred.loc[df_pred.index == id, "Prediction"] = max_class
    print(df_pred.loc[df_pred.index == id])

# Receive each row and process it
for msg in consumer:
    aux = msg.value.decode('utf-8').rstrip('\n').split(',')
    df = pd.DataFrame(columns=['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports'])
    df.loc[0] = aux
    for c in df.columns:
        df[c] = df[c].astype(float)
    df.set_index('id', inplace = True)
    df_6 = df[['dur', 'proto', 'service', 'sbytes', 'smean', 'ct_dst_sport_ltm']]
    results = dict()
    id = int(df.index.values[0])

    print('')
    print('MODELS')
    print('')
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

    print('PREDICTIONS: ', results)

    choose()

    print('')

