import os
import pandas                   as pd
import numpy                    as np
import matplotlib.pyplot        as plt
import seaborn                  as sns
from collections                import Counter
from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import train_test_split
from imblearn.over_sampling     import SMOTE
from scipy.stats                import zscore

df1 = pd.read_csv("./data/UNSW_NB15_testing-set.csv")
df2 = pd.read_csv("./data/UNSW_NB15_training-set.csv")

df = pd.concat([df1, df2])

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

# MinMaxScaler for numerical features
print("Values transformed using Zscore function")
for feature in num:
    df[feature] = zscore(df[feature])

# Load Balancer with SMOTE
print("")
print("")
print("Data-set NOT balanced with SMOTE class")
X = df.drop('attack_cat',1)
y = df['attack_cat']
counter = Counter(y)
print(counter)
print([(i, counter[i] / sum(counter.values())  * 100.0) for i in counter])

# Pre-process saved
label = X['label']
X.drop('label', axis=1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

X_train.to_csv('./data/x_train_corr.csv')
X_test.to_csv('./data/x_test_corr.csv')
y_train.to_csv('./data/y_train.csv')
y_test.to_csv('./data/y_test.csv')

# Correlation Analysis:
print("")
print("#########################")
print("CORRELATION ANALYSIS")
print("#########################")
print("")

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
X_test.drop(labels=c_features, axis=1, inplace=True)
X_train.drop(labels=c_features, axis=1, inplace=True)
#plt.show()
fig.savefig("./img/corr_matrix.png", dpi=300)

# Save dataset without correlated features
X_train.to_csv('./data/x_train_uncorr.csv')
X_test.to_csv('./data/x_test_uncorr.csv')
print('Data-set correctly saved.')