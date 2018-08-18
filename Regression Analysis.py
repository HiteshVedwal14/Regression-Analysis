# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('car data.csv')


Name = dataset['Car_Name']
#Converting the Name to DataFrame
Name = pd.DataFrame(Name)
#Calculating the Name of the most frquent car 
Name.Car_Name.value_counts()

df = dataset.loc[dataset['Car_Name'] == "city"].iloc[:, 1:5]
X = df.iloc[:, 0:1].values
y1 = df.iloc[:, 1:2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y1)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y1, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Selling Price (lakhs)')
plt.show()

y1_pred = lin_reg_2.predict(poly_reg.fit_transform(X))




#For Present Price

y2 = df.iloc[:, 2:3].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
regressor.fit(X, y2)

# Predicting a new result
y2_pred = regressor.predict(X)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y2, color = 'magenta')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Year')
plt.ylabel('Present Price(lakhs)')
plt.show()



