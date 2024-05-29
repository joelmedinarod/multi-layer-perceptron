from model import MultiLayerPerceptron
import pickle
from sklearn.metrics import accuracy_score
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Convert to NumPy arrays
x_train = np.array(x_train.reshape(60000,28*28))
y_train = np.array(y_train.reshape(60000))
x_test = np.array(x_test.reshape(10000,28*28))
y_test = np.array(y_test.reshape(10000))

# Initialize and train the MLP
mlp = MultiLayerPerceptron(n_features=28*28, n_hidden_layers=2, hidden_size=128, n_classes=10)

# Train Multi-Layer Perceptron
mlp.train(x_train, y_train, epochs=1000, learning_rate=0.001)

# Predict and evaluate on test data
y_pred = mlp.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data: {test_accuracy * 100:.2f}')

# Save trained Multi-Layer Perceptron
with open('mlp.pkl', 'wb') as outp:
    pickle.dump(mlp, outp, pickle.HIGHEST_PROTOCOL)


# Plot the first 20 images
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_test[i].reshape(8, 8), cmap='gray')  # Reshape each image to 28x28
    plt.title(f"Label: {y_test[i]}, Prediction: {mlp.predict(x_test[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()