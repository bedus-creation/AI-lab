import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd

# read Data using pandas
data = pd.read_csv('data/linear.txt', sep=",", header=None)
data.columns = ["x", "y"]

# Training and testing samples
training_samples = int(0.6 * data.size)
testing_samples = data.size - training_samples

# Separate Training and testing data
X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]

# reg_linear = linear_model.LinearRegression()
# reg_linear.fit(X_train, y_train)
# y_test_pred = reg_linear.predict(X_test)
# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_test, y_test_pred, color='black', linewidth=2)
# plt.xticks(())
# plt.yticks(())
# plt.show()
