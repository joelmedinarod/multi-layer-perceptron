from sklearn.datasets import make_circles
from model import MultiLayerPerceptron, plot_decision_boundaries
from helper_functions import accuracy_fn

# Create data points of two concentric circles
X, y = make_circles(
    n_samples=1000,
    shuffle=True,
    noise=0.03,
    random_state=42,
)

# Separate data set into train and test sets
num_train = int(0.8 * len(X))
X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# Initialize a Multi Layer Perceptron with 2 input
# neurons, 8 hidden units and 2 output neurons.
mlp = MultiLayerPerceptron(n_features=2, n_hidden_layers=1, hidden_size=8, n_classes=2)

# Train the Multi Layer Perceptron
mlp.train(X_train, y_train, epochs=10000, learning_rate=0.03)

# Predict on the test data and print accuracy
y_pred = mlp.predict(X_test)
print("Accuracy on test data:", accuracy_fn(y_test, y_pred))

plot_decision_boundaries(mlp, X, y, grid_resolution=(101, 101))
