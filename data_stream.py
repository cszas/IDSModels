import os
import pandas as pd
from kafka 						import KafkaProducer
import time
from sklearn.utils 				import shuffle
from sklearn.preprocessing 		import LabelEncoder
from scipy.stats 				import zscore

# Load data-set and divide into features to stream and result to compare
df = pd.read_csv("./data/UNSW_NB15_testing-set.csv")
df.set_index('id', inplace = True)

# Function to encode String features
def l_encoder(col):
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])
	if col == 'attack_cat':
		for i in range(len(le.classes_)):
			print(i, ': ', le.classes_[i])

# Apply label encoders and zscore function to numerical values
for c in df.columns: 
    if df[c].dtypes not in ('int64', 'float64'): 
        l_encoder(c)
    else: df[c] = zscore(df[c])


# Suffle the data-set
df = shuffle(df)

# Save Real Attack Category
df['attack_cat'].to_csv('./data/test_y.csv')
df.drop(['label','attack_cat'], axis=1, inplace = True)

# Save Features 
df.to_csv('./data/test_x.csv')

# Open features
file = open("./data/test_x.csv","r") #Solo lectura

# Send to Kafka Broker
producer = KafkaProducer(bootstrap_servers="localhost:9092")

for line in file:
	if not line.__contains__('id'):
		print(line)
		producer.send('Streaming_TFM',line.encode('utf-8')) #Topic: Streaming_TFM
		time.sleep(10)


