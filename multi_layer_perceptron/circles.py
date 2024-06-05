"""
Create datasets with two classes of datapoints dispersed
in a 2D-Space as circles and classify them using MLP
"""

from helper_functions import accuracy_fn
from model import MultiLayerPerceptron, plot_decision_boundaries
from sklearn.datasets import make_circles

# Define model hyperparameters
N_HIDDEN_LAYERS = 2
HIDDEN_SIZE = 8  # number of neurons per hidden layer

# Define training hyperparameters
N_EPOCHS = 5000
LEARNING_RATE = 0.03
TRAINING_SPLIT = 0.8  # 80% of data is training data

# Create data points of two concentric circles
X, y = make_circles(
    n_samples=1000,
    shuffle=True,
    noise=0.03,
    random_state=42,
)

# Separate data set into train and test sets
num_train = int(TRAINING_SPLIT * len(X))
X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# Initialize a Multi Layer Perceptron with 2 input
# neurons, 8 hidden units and 2 output neurons.
mlp = MultiLayerPerceptron(
    n_features=2, n_hidden_layers=N_HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE, n_classes=2
)

# Train the Multi Layer Perceptron
mlp.train(
    X_train,
    y_train,
    epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    print_every_n_epochs=100,
)

# Predict on the test data and print accuracy
y_pred = mlp.predict(X_test)
print("Accuracy on test data:", accuracy_fn(y_test, y_pred))

# Plot dataset and decision boundary trained by model
plot_decision_boundaries(mlp, X, y, grid_resolution=(101, 101))
