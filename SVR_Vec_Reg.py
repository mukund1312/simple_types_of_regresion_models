# Support vector regression

# importing all the libraries

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from typing import ValuesView
import numpy as np
import pandas as pd
import matplotlib.pylab as mlt
from sklearn.utils.extmath import svd_flip


# read the data set
Data_Set = pd.read_csv("Position_Salaries.csv")
x = Data_Set.iloc[:, 1:-1].values
y = Data_Set.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scalling
x_scale = StandardScaler()
y_scale = StandardScaler()
x = x_scale.fit_transform(x)
y = y_scale.fit_transform(y)


# SVR model

reg = SVR(kernel="rbf")
reg.fit(x, y)

# single value prediction
pred = float(input("enter the value :\n"))
pred_fin = round(float(y_scale.inverse_transform(
    reg.predict(x_scale.transform([[pred]])))), 2)

print(f' the predicted value is :\n$ {pred_fin}')


# now we visulize the Polynomial Regresion model
mlt.scatter(x_scale.inverse_transform
            (x), y_scale.inverse_transform(y), color="red")
mlt.plot(x_scale.inverse_transform(x),
         y_scale.inverse_transform(reg.predict(x)), color="blue")
mlt.title("Bluff or truth  (SVR regresion)")
mlt.ylabel("salary")
mlt.xlabel("position")


# for a smoother curve
x = x_scale.inverse_transform(x)
y = y_scale.inverse_transform(y)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
mlt.scatter(x, y, color='red')
mlt.plot(X_grid, y_scale.inverse_transform(
    reg.predict(x_scale.transform(X_grid))), color='blue')
mlt.title('Truth or Bluff (Support linear Regression)')
mlt.xlabel('Position level')
mlt.ylabel('Salary')
mlt.show()
