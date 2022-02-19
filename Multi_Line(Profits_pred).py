# Importing all the libraries and classes we are going to use
from operator import le
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from math import remainder
from sklearn.linear_model import LinearRegression
from matplotlib import colors
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt


# Importing the data set
DataSet = pd.read_csv("50_Startups.csv")
x = DataSet.iloc[:, :-1].values
y = DataSet.iloc[:, -1].values


# Encoding the DataSet
# Here we One-Hot-Encode coz we have the state coloum which is a variable we should take in consideration while building the model
ct = ColumnTransformer(
    transformers=[("endcoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))


# Splitting them apart into Train Set ## Slipting them apart into Test Set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=0)


# Training our model

reg = LinearRegression()
reg.fit(x_train, y_train)


# Predict the results
y_pred = reg.predict(x_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# How to make singel predictions

f = round(float(reg.predict([[0, 1, 0, 160000, 130900, 30000]])), 2)
print(f"$ {f}")
