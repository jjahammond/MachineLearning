# Data Preprocessing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

# Split the dataset into the Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

# Feature scaling
"""standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.transform(X_train)""" # Don't need to fit as already fitted to X_train

# We must check assumptions of linear regression
# 1. Linearity
# 2. Homoscedasticity
# 3. Multivariate normality
# 4. Independence of errors
# 5. Lack of multicollinearity


# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print("Salary = {}*Years + {}".format(regressor.coef_[0], regressor.intercept_))

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
