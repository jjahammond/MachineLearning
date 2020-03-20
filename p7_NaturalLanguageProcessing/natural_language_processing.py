# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Clean text libraries
import re
import nltk
nltk.download('stopwords') # Words to remove from reviews
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Bag of words libraries
from sklearn.feature_extraction.text import CountVectorizer

# ML libraries
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the dataset
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating bag of words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting K-NN to the Training set - 61%
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set - 73%
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set - 73.5%
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set - 71%
# classifier = DecisionTreeClassifier(criterion = 'entropy')
# classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set - 71.5%
# classifier = RandomForestClassifier(n_estimators = 10000, criterion = 'entropy')
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Prediction Accuracy: {}%".format(np.trace(cm)/2))
