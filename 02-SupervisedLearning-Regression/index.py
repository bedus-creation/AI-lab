import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd

# read Data using pandas
data = pd.read_csv('data/linear.txt', sep=",", header=None)
data.columns = ["x", "y"]

# Training and testing samples
training_samples = int(0.6 * len(data))
testing_samples = len(data) - training_samples


# Separate Training and testing data
train_data = data[:training_samples]
test_data = data[-training_samples:]


# Create a linear regressor object
reg_linear = linear_model.LinearRegression()
# print(train_data[["x"]])
reg_linear.fit(train_data[["x"]], train_data[["y"]])
y_test_pred = reg_linear.predict(test_data[["x"]])
plt.scatter(test_data['x'], test_data['y'], color='red')
plt.plot(test_data['x'], y_test_pred, color='black', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()
