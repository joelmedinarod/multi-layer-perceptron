import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

# Path to saved model for classifying numbers (mnist dataset)
PATH_TO_SAVED_MODEL = "mlp.pkl"

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to NumPy arrays
x_train = np.array(x_train.reshape(60000, 28 * 28))
y_train = np.array(y_train.reshape(60000))
x_test = np.array(x_test.reshape(10000, 28 * 28))
y_test = np.array(y_test.reshape(10000))


# Save trained Multi-Layer Perceptron
with open(PATH_TO_SAVED_MODEL, "rb") as inp_file:
    mlp = pickle.load(inp_file)


# Plot the first 20 images
plt.figure(figsize=(10, 5))
for i in range(80):
    y_pred = mlp.predict(x_test[1000 + i])
    y = y_test[1000 + i]
    if y_pred != y:
        plt.subplot(8, 10, i + 1)
        plt.imshow(
            x_test[i + 1000].reshape(28, 28), cmap="gray"
        )  # Reshape each image to 28x28
        plt.title(f"y: {y}, pred: {y_pred}")
        plt.axis("off")
plt.tight_layout()
plt.show()
