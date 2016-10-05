import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression   
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') # DataFrame
# print(df.head()) Displays first 5 rows of the dataset
# print(df.tail()) Displays last 5 rows of the dataset

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] # We are choosing only some of the coumns here

# High Low Percentage
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0

# Percent Change
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True) # Replacing NaN values with some value

forecast_out = int(math.ceil(0.01*len(df))) # Number of columns to be shifted up to make future predictions
# print(forecast_out) Currently it is 31

df['label'] = df[forecast_col].shift(-forecast_out) # To shift 'forecast_out' number of columns up

# Features
X = np.array(df.drop(['label'],1)) # 1 denotes we are deleting a column and not a row

# Scaling:
# Standardize a dataset along any axis (0 by default (row wise))
# Center to the mean and component wise scale to unit variance.
X = preprocessing.scale(X)

X = X[:-forecast_out] # Everything except the last forecast_out number of elements
X_lately = X[-forecast_out:] # To be used for prediction (Last forecast_out number of elements)

df.dropna(inplace=True) 
# Labels
y = np.array(df['label'])
y = np.array(df['label'])

# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # test_size here denotes we want to use only 20% data as the testing data 

clf = LinearRegression() # Initialize the classifier 
# Here, we can use 'n_jobs' as an argument and it denotes number of threads. 
# The number of jobs to use for the computation. If -1 all CPUs are used. 
# This will only provide speedup for n_targets > 1 and sufficient large problems. By default n_jobs = 1

# clf = svm.SVR() This can also be used as a classifier. SVR is Support Vector regression. In this case, its accuracy is less than LinearRegression()

clf.fit(X_train, y_train) # Fit or train the classifier X: Features, y: Label 
accuracy = clf.score(X_test,y_test)

# print(accuracy)

# Predicting for last forecast_out number of elements 
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

# loc works on labels in the index.
# iloc works on the positions in the index (so it only takes integers).
# ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
