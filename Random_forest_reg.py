# Importing all the data set
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt


# import all the data sets
Data_set = pd.read_csv("Position_Salaries.csv")
x = Data_set.iloc[:, 1:-1].values
y = Data_set.iloc[:, -1].values

# lets now train our Data set with random forest model
reg = RandomForestRegressor(n_estimators=10, random_state=0)
reg.fit(x, y)

# now lets predict
val = float(input("enter the value you want to predict for :\n"))
print(f'The predicted salary is $ {round(float(reg.predict([[val]])),2)}')
