# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

lifts = []
for i in range(len(results)):
    lifts.append(results[i].ordered_statistics[0].lift)
order = np.argsort(lifts)

for i in order:
    ordered_statistics = results[i].ordered_statistics[0]
    print("{}\t{}\t{}".format(ordered_statistics.items_base, ordered_statistics.items_add, ordered_statistics.lift))
