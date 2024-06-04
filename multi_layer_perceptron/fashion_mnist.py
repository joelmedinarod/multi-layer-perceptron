from model import MultiLayerPerceptron
import pickle
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Convert to NumPy arrays
x_train = np.array(x_train.reshape(60000, 28 * 28))
y_train = np.array(y_train.reshape(60000))
x_test = np.array(x_test.reshape(10000, 28 * 28))
y_test = np.array(y_test.reshape(10000))

# Initialize and train the MLP
mlp = MultiLayerPerceptron(
    n_features=28 * 28, n_hidden_layers=2, hidden_size=128, n_classes=10
)

# Train Multi-Layer Perceptron
mlp.train(x_train, y_train, epochs=500, learning_rate=0.0002)

# Predict and evaluate on test data
y_pred = mlp.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {test_accuracy * 100:.2f}")

# Save trained Multi-Layer Perceptron
with open("fashion_mlp.pkl", "wb") as outp:
    pickle.dump(mlp, outp, pickle.HIGHEST_PROTOCOL)
