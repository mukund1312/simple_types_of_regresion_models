# Import all the libraries
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt


# import the data set
Data_set = pd.read_csv("Position_Salaries.csv")
x = Data_set.iloc[:, 1:-1].values
y = Data_set.iloc[:, -1].values


# Train the regresion Tree Data set Model
reg = DecisionTreeRegressor(random_state=0)
reg.fit(x, y)


# predict a new result
val = float(input("enter the value you want to predict the salary for :\n"))
p = round(float(reg.predict([[val]])), 2)
print(f'$ {p}')

# for a visulatizing a smoother curve
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
mlt.scatter(x, y, color='red')
mlt.plot(X_grid, reg.predict(X_grid), color='blue')
mlt.title('Truth or Bluff (Decision tree Regression)')
mlt.xlabel('Position level')
mlt.ylabel('Salary')
mlt.show()
