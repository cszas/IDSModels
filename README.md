# Adaptable Real-Time Machine Learning Intrusion Detection System based on attacks categorization

### Carmen Sanchez Zas

### MUIT - ETSI Telecomunicación - UPM

This repository includes the code and data-sets used for the development of the project.

Nowadays, Intrusion Detection Systems include neural networks trained to locate certain types of attacks, resulting into a generic operation, where the choice is between being accurate or exhaustive. 
However, the problem is simplified if this system has the ability to adapt to different attacks, using a machine learning algorithm depending on what it is receiving in real time.
The main objective of the project is to develop an adaptable system based on attack categorization, presenting an automatic selector for the training algorithm using different machine learning algorithms.


## Requirements

- TensorFlow
- Keras
- [Kafka](https://kafka.apache.org/downloads) 
- Conda3
- Pandas
- Scikit-learn

## Installation

To avoid Tensorflow incompatibility with some Python versions, the project was developed using a conda environment.
To create it, run:
 ```
conda create -y --name tensorflow python=3.6
 ```
 And, in order to activate it:
 ```
 conda activate tensorflow
 ```
## Repository Structure

### Data



On this folder, there are included the original UNSW-NB15 data-set used, and those generated from the process:

- **UNSW_NB15_training-set.csv**: Original [training set](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_training-set.csv).
- **UNSW_NB15_testing-set.csv**: Original [testing set](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_testing-set.csv).

In executing `pre-process.py`, the files obtained are:

- **UNSW_corr.csv**: Complete pre-processed training set, with all the features. Obt
- **UNSW_uncorr.csv**: Pre-processed training set, without correlated features.

Also, while streaming data, the Python file `data_stream.py` will generate the following:
- **test_x.csv**: Randomly mixed features from testing set.
- **test_y.csv**: Attack categorization column from testing set.

### Img

Here, all the graphs generated from the pre-processing and training phases are included.

### Saved_model

The trained models obtained are located here, with .pkl or .h5 extension.

### Main

- **pre-process.py**: Python script to deal training data.
- **data_stream.py**: Python script to send each testing row to the real-time classifier via Kafka.
- **real_time_classifier.py**: Python script to receive real-time data and evaluate it with all the models trained, choosing one.
- **knn.py**: Python script to develop and train KNN model.
- **svm.py**: Python script to develop and train SVM model.
- **dt.py**: Python script to develop and train Decision Tree Classifier model.
- **rf.py**: Python script to develop and train Random Forest Classifier model.
- **xgb.py**: Python script to develop and train XGBoost model.
- **MLP.py**: Python script to develop and train MLP model.
- **LSTM.py**: Python script to develop and train LSTM model.

## Execution

### Model Training
During this phase, the files from the saved_model folder are created. To assure the correct operation of the whole project, it is recommended to run them.

```
> python3 knn.py
> python3 svm.py
> python3 dt.py
> python3 rf.py
> python3 xgb.py
> python3 MLP.py
> python3 LSTM.py
```

### Real-time Classifier

To run this feature, the scripts used will be `data_stream.py` and `real_time_classifier.py`, and the Kafka Broker.


#### Kafka Broker

Kafka scripts to raise the Broker. Version 2.12-2.3.0 is used.
This unit is in charge of receiving the rows sended from `data_stream.py`, and send them to `real_time_classifier.py`.

*Terminal 1:* 
```
> cd kafka_2.12_2.3.0
> bin/zookeeper-server-start.sh config/zookeeper.properties 
```

*Terminal 2:*
```
> cd kafka_2.12_2.3.0
> bin/kafka-server-start.sh config/server.properties
```

#### Classifier

This script will wait the data to be streamed.

*Terminal 3:*
```
> python3 real_time_classifier.py
```
#### Data Stream

Run:

*Terminal 4:*
```
> python3 data_stream.py
```
### Results

As a result, the `real_time_classifier.py` script will show the attack type predicted, compared to the real value.
For this purpose,the label assigned to each attack is:
0 :  Analysis
1 :  Backdoor
2 :  DoS
3 :  Exploits
4 :  Fuzzers
5 :  Generic
6 :  Normal
7 :  Reconnaissance
8 :  Shellcode
9 :  Worms

