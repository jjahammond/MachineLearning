# Upper Confidence Bound

# Importing the libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import math

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10

ads_selected = []
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
total_reward = 0

for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        # Only run algorithm once all ads have been shown once
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        # Otherwise, set upper_bound of ad to large value and use each ad one by one
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

print("Total Rewards: {}".format(total_reward))

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Visualising the results
plt.bar(range(10), sums_of_rewards)
plt.title('Histogram of ads rewards')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was rewarded')
plt.show()
