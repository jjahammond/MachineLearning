# Data Preprocessing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as StatsModels

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode catagorial data
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Avoid dummy variable trap (we don't need to do this, python library would do it for us)
X = X[:,1:]

# Split the dataset into the Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling (Library takes care of this as well)
"""standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.transform(X_train)""" # Don't need to fit as already fitted to X_train

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

for i in range(len(y_pred)):
    print("Real Profit: ${}\t Predicted Profit: ${:.2f}".format(y_test[i], y_pred[i]))

# Building the optimal model using backward elimination
# X = np.append(arr=np.ones((X.shape[0],1),dtype=float), values=X, axis=1)
X = StatsModels.add_constant(X).astype(float)

X_opt = X[:,[0,1,2,3,4,5]]
regressorOLS = StatsModels.OLS(y, X_opt).fit()

print(regressorOLS.pvalues)
print(np.argmax(regressorOLS.pvalues))

# Watch adjusted R-squared here as well - while it increases we're good
# if it drops, maybe don't remove the independent variable.
while np.max(regressorOLS.pvalues) > 0.05:

    X_opt = X_opt[:,np.arange(X_opt.shape[1]) != np.argmax(regressorOLS.pvalues)]
    regressorOLS = StatsModels.OLS(y, X_opt).fit()

print(regressorOLS.summary())
print(X_opt)
