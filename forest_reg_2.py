# test case 2 lets see how this works
# importing all the libraries
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

# read the Data set
Data_set = pd.read_csv("Regration\Data.csv")
x = Data_set.iloc[:, :-1].values
y = Data_set.iloc[:, -1].values

# make test and train set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=0)


# lets train the model

reg = RandomForestRegressor(n_estimators=10, random_state=0)
reg.fit(x_train, y_train)


# now lets predict
y_pred = reg.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# for finding the effiecncy
score = r2_score(y_test, y_pred)
print(score)
