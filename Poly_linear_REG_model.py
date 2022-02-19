# importing all the libraries we will be using to build our model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

# Read our data SET
Data_Set = pd.read_csv("Position_Salaries.csv")
x = Data_Set.iloc[:, 1:-1].values
y = Data_Set.iloc[:, -1].values


# build a linear model to see the eficiency
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Now build a poly linear reg

poly_Fea = PolynomialFeatures(degree=3)
x_poly = poly_Fea.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)


# now we visulize the Linear Regresion model out put
mlt.scatter(x, y, color="red")
mlt.plot(x, lin_reg.predict(x), color="blue")
mlt.title("Bluff or truth  (linear regresion)")
mlt.ylabel("Salary")
mlt.xlabel("Position")
# mlt.show()

# now we visulize the Polynomial Regresion model
mlt.scatter(x, y, color="red")
mlt.plot(x, poly_reg.predict(x_poly), color="blue")
mlt.title("Bluff or truth  (Poly regresion)")
mlt.ylabel("salary")
mlt.xlabel("position")
# mlt.show()

# for a smoother curve
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
mlt.scatter(x, y, color='red')
mlt.plot(X_grid, poly_reg.predict(
    poly_Fea.fit_transform(X_grid)), color='blue')
mlt.title('Truth or Bluff (Polynomial Regression)')
mlt.xlabel('Position level')
mlt.ylabel('Salary')
mlt.show()

# for predicting singel values by linear Regration
print(
    f'this is the predicted salary using Linear Regresion model: $ {round(float(lin_reg.predict([[6.6]])), 2)}')

# for predicting singel values by Polynomial Regration

print(
    f'this is the predicted salary using Plynomial Regresion model: $ {round(float(poly_reg.predict(poly_Fea.fit_transform([[6.6]]))), 2)}')
