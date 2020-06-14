import os
import pandas as pd
from kafka 						import KafkaProducer
import time

# Open features
file = open("./data/x_test_corr.csv","r") #Solo lectura

# Send to Kafka Broker
producer = KafkaProducer(bootstrap_servers="localhost:9092")

for line in file:
	if not line.__contains__('id'):
		print(line)
		producer.send('Streaming_TFM',line.encode('utf-8')) #Topic: Streaming_TFM
		time.sleep(1)