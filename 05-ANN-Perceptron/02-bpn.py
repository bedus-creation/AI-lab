from src.BackPropagation import BackPropagation
import numpy as np

# Inputs
training_inputs = np.array([0.5, 0.1])
# Desired output
target = np.array([1, 0, 0, 0])


# Executing the main class of back Propagation
backprogation = BackPropagation(2, 2, 2, 2)
backprogation.train(training_inputs, target)
