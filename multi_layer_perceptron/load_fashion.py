import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# Convert to NumPy arrays
x_train = np.array(x_train.reshape(60000, 28 * 28))
y_train = np.array(y_train.reshape(60000))
x_test = np.array(x_test.reshape(10000, 28 * 28))
y_test = np.array(y_test.reshape(10000))


# Save trained Multi-Layer Perceptron
with open("models/fashion_mlp.pkl", "rb") as inp_file:
    mlp = pickle.load(inp_file)

# Predict and evaluate on test data
y_pred = mlp.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {test_accuracy * 100:.2f}")

# Plot false predictions from the first 40 images (after image #1000)
plt.figure(figsize=(10, 5))
for i in range(40):
    y_pred = fashion_mnist_labels[int(mlp.predict(x_test[1000 + i]))]
    y = fashion_mnist_labels[int(y_test[1000 + i])]
    if y_pred != y:
        plt.subplot(5, 8, i + 1)
        plt.imshow(
            x_test[i + 1000].reshape(28, 28), cmap="gray"
        )  # Reshape each image to 28x28
        plt.title(f"y: {y}, pred: {y_pred}")
        plt.axis("off")
plt.tight_layout()
plt.show()
