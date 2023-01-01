#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# df = pd.read_csv("health_data.csv")
#
#
# X = df.drop("Age", axis=1)
# y = df["Age"]
#
# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Build the model
# model = RandomForestClassifier()
#
# # Train the model on the training data
# model.fit(X_train, y_train)
#
#
# # Use the model to make predictions about the future
# future_predictions = model.predict(X_test)
#
# print(future_predictions)
# output_table = pd.DataFrame({"prediction": future_predictions})
# print(output_table)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read in data
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('health_data.csv')

# Pre-process data
# ...
#
# # Split data into training and test sets
# X = df.drop('BMI', axis=1)
# y = df['BMI']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Fit linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on test set
# y_pred = model.predict(X_test)
#
# # Evaluate results
# #Compare ML results
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
# print(f'MAE: {mae:.2f}')
# print(f'MSE: {mse:.2f}')
# print(f'RMSE: {rmse:.2f}')
# print(f'R2: {r2:.2f}')

X = df.drop('BMI', axis=1)
y = df['BMI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Fit decision tree model
model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

# Make predictions on test set
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

# Evaluate results
mae1 = mean_absolute_error(y_test, y_pred1)
mse1 = mean_squared_error(y_test, y_pred1)
rmse1 = np.sqrt(mse1)
r2_1 = r2_score(y_test, y_pred1)

mae2 = mean_absolute_error(y_test, y_pred2)
mse2 = mean_squared_error(y_test, y_pred2)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y_test, y_pred2)

print(f'Linear Regression:')
print(f'MAE: {mae1:.2f}')
print(f'MSE: {mse1:.2f}')
print(f'RMSE: {rmse1:.2f}')
print(f'R2: {r2_1:.2f}')

print(f'Decision Tree:')
print(f'MAE: {mae2:.2f}')
print(f'MSE: {mse2:.2f}')
print(f'RMSE: {rmse2:.2f}')
print(f'R2: {r2_2:.2f}')