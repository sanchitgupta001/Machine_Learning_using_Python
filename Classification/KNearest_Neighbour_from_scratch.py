import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

# A Counter is a dict subclass for counting hashable objects. 
# It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values.

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
        
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.show()       

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]        
    vote_result = Counter(votes).most_common(1)[0][0] # Finds the single most common key from votes.
    # In order to get more than 1 most common pass n required as an argument to most_common() function.
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# Some of the data elements may be in quotes(as a string). So, we are converting each element to float datatype.
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]} # 2 : benign and 4 : malignant
# for more info consult breast-cancer-wisconsin.names file 
          
test_set = {2:[],4:[]} 
train_data = full_data[:-int(test_size*len(full_data))] 
test_data = full_data[-int(test_size*len(full_data)):]
 
# Filling the train_set with train_data
for i in train_data:
    train_set[i[-1]].append(i[:-1])  

# Filling the test_set with test_data
for i in test_data:
    test_set[i[-1]].append(i[:-1]) 

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total +=1    
    
print('Accuracy : ',correct/total)        