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
- Scikit-learn
- [Kafka](https://kafka.apache.org/downloads) 
- Conda3
- Pandas, numpy, matplotlib, seaborn, joblib

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

The repository is structured in two folders with the same organization, for balanced and original data-sets. Inside them, there are different folders for data, generated images (img) and trained models (saved_model), and the python classes created for this project.

### Data

On this folder, there must be included the original UNSW-NB15 data-set used, and those generated from the process:

- **UNSW_NB15_training-set.csv**: Original [training set](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_training-set.csv).
- **UNSW_NB15_testing-set.csv**: Original [testing set](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_testing-set.csv).

Executing `pre-process.py`, the files obtained are:

- **x_train_corr.csv**: Complete pre-processed training set, with all the features.
- **x_train_uncorr.csv**: Complete pre-processed training set, without the correlated features.
- **x_test_corr.csv**: Complete pre-processed testing set, with all the features.
- **x_test_uncorr.csv**: Complete pre-processed testing set, without the correlated features.
- **y_train.csv**: Attack category column from training set.
- **y_test.csv**: Attack category column from testing set.

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
- **decision_tree.py**: Python script to develop and train Decision Tree Classifier model.
- **random_forest.py**: Python script to develop and train Random Forest Classifier model.
- **xgboost.py**: Python script to develop and train XGBoost model.
- **mlp.py**: Python script to develop and train MLP model.
- **lstm.py**: Python script to develop and train LSTM model.
- **global_classifier.py**: Python script to develop voter method for dynamic classifier.
- **ensemble.py**: Python script to develop ensemble method for dynamic classifier.


## Execution

### Data Pre-Process

Having in 'data' folder the training and testing data-sets, when executing `> pyhton3.6 pre-process.py`, all the transformations over data occur.

### Model Training
During this phase, the files from the saved_model folder are created. To assure the correct operation of the whole project, it is recommended to run them.

```
> python3.6 knn.py
> python3.6 svm.py
> python3.6 decision_tree.py
> python3.6 random_forest.py
> python3.6 xgboost.py
> python3.6 mlp.py
> python3.6 lstm.py
```

Also, there are two classes to verify the overall performance of the dynamic classifier:

```
> python3.6 global_classifier.py

> python3.6 ensemble.py
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
> python3.6 real_time_classifier.py
```
#### Data Stream

Run:

*Terminal 4:*
```
> python3.6 data_stream.py
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

