# Data Preprocessing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# Import dataset using pandas and convert to numpy array
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Deal with missing data by using the mean of other values in feature
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode categorical data (Countries and Purchased)
labelEncoderX = LabelEncoder()
X[:,0] = labelEncoderX.fit_transform(X[:,0])
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

## Use dummy encoding to stop bias (On countries)
# My solution
# oneHotEncoder = OneHotEncoder(handle_unknown='ignore')
# enc = oneHotEncoder.fit_transform(X[:,0].reshape(-1,1)).toarray()
# X = np.concatenate((enc,X[:,1:]),axis=1)

# Their solution
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset into the Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.transform(X_train) # Don't need to fit as already fitted to X_train
