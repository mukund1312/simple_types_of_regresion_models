
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