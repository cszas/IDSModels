import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/UNSW_NB15_training-set.csv")

id = df['id']
df.set_index('id', inplace = True)

# Non-numerical Values:
print("")
print("################################")
print("ANALYSIS OF NON-NUMERICAL VALUES")
print("################################")
print("")
non_num = []
num = []
for c in df.columns: 
    if df[c].dtypes not in ('int64', 'float64'): 
        non_num.append(c)
    else: num.append(c)
print("Non-numerical Values: ", non_num)
print("")
# Unique values for each non_numerical:
for item in non_num:
    print("Number of distinct values for ", item, ": ",df[item].unique().size)

# For protocol feature, a numerical value is asigned for each distinct object with label encoder.
le_proto = LabelEncoder()
df['proto'] = le_proto.fit_transform(df['proto'])

le_service = LabelEncoder()
df['service'] = le_service.fit_transform(df['service'])

le_state = LabelEncoder()
df['state'] = le_state.fit_transform(df['state'])

print("")
print("Label assigned to each attack:")
le_cat = LabelEncoder()
df['attack_cat'] = le_cat.fit_transform(df['attack_cat'])
for i in range(len(le_cat.classes_)):
   print(i, ': ', le_cat.classes_[i])

# Numerical Values:
print("")
print("##############################")
print("ANALYSIS OF NUMERICAL VALUES")
print("##############################")
print("")

# Zscore for numerical features
print("Values transformed using zscore function")
for feature in num:
    df[feature] = zscore(df[feature])

# Pre-process saved
label = df['label']
df.drop('label', axis=1, inplace = True)
df.to_csv('./data/UNSW_corr.csv')

# Correlation Analysis:
print("")
print("#########################")
print("CORRELATION ANALYSIS")
print("#########################")
print("")

X = df.drop('attack_cat',1)
y = df['attack_cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# Create the correlation matrix using kendall coefficient
c_matrix = X_train.corr('kendall')

# Display and save a heatmap of the correlation matrix
fig = plt.figure(figsize=(11,11))
sns.heatmap(c_matrix)
c_features = set()
for i in range(len(c_matrix.columns)):
    for j in range(i):
        if abs(c_matrix.iloc[i, j]) > 0.8:
            col = c_matrix.columns[i]
            c_features.add(col)

print("The correlated features are deleted: ", c_features)
print('')          
df.drop(labels=c_features, axis=1, inplace=True)
#plt.show()
fig.savefig("./img/corr_matrix.png", dpi=300)

# Save dataset without correlated features
df.to_csv('./data/UNSW_uncorr.csv')
print('Data-set correctly saved.')