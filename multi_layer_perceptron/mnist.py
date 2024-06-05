import pickle

import numpy as np
from keras.datasets import mnist
from model import MultiLayerPerceptron
from sklearn.metrics import accuracy_score

# Path to file to save model parameters after training
SAVE_MODEL_FILEPATH = "models/mlp.pkl"

# Training hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.001

# Model hyperparamters
N_HIDDEN_LAYERS = 2
HIDDEN_SIZE = 128

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to NumPy arrays
x_train = np.array(x_train.reshape(60000, 28 * 28))
y_train = np.array(y_train.reshape(60000))
x_test = np.array(x_test.reshape(10000, 28 * 28))
y_test = np.array(y_test.reshape(10000))

# Initialize and train the MLP
mlp = MultiLayerPerceptron(
    n_features=28 * 28,
    n_hidden_layers=N_HIDDEN_LAYERS,
    hidden_size=N_HIDDEN_LAYERS,
    n_classes=10,
)

# Train Multi-Layer Perceptron
mlp.train(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Predict and evaluate on test data
y_pred = mlp.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {test_accuracy * 100:.2f}")

# Save trained Multi-Layer Perceptron
with open(SAVE_MODEL_FILEPATH, "wb") as outp:
    pickle.dump(mlp, outp, pickle.HIGHEST_PROTOCOL)
