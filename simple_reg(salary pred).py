# Step 1 is to import the libraries

from sklearn.linear_model import LinearRegression
from matplotlib import colors
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

# step 2 is to import the dataset

dataSet = pd.read_csv("Salary_Data.csv")
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# step 3 is to slipt the dataSet into train set and test set

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=0)

# print(f'This is the x train set :\n{x_train}')
# print(f'This is the x test set :\n{x_test}')
# print(f'This is the y train set :\n{y_train}')
# print(f'This is the y test set :\n{y_test}')


# Step 4 is to train the model in train Data set

reg = LinearRegression()
reg.fit(x_train, y_train)

# Step 5 is to predict the test case

y_pred = reg.predict(x_test)

# Step 6 is to visulize the train Data set

mlt.scatter(x_train, y_train, color="red")
mlt.plot(x_train, reg.predict(x_train), color="blue")
mlt.title("salary VS experience (Train DataSet)")
mlt.ylabel("Salary")
mlt.xlabel("Years or experience")
mlt.show()

# Step 7 is to visualize the Test Data SET

mlt.scatter(x_test, y_test, color="red")
mlt.plot(x_train, reg.predict(x_train), color="blue")
mlt.title("salary VS experience (Test DataSet)")
mlt.ylabel("Salary")
mlt.xlabel("Years of experience")
mlt.show()


# Making a single prediction (for example the salary of an employee with 12 years of experience)

years = int(input(
    "enter the salary you want to predict for the Number Years of experience:\n "))
pred_sal = reg.predict([[years]])
pred_sal = round(float(pred_sal), 2)
print(f"The Employee would be paid : ${pred_sal}")


# Getting the final linear regression equation with the values of the coefficients
# Salary=9345.94×YearsExperience+26816.19
print(reg.coef_)
print(reg.intercept_)
