from sklearn.datasets import load_iris
from model import MultiLayerPerceptron
from helper_functions import accuracy_fn

# Get IRIS Dataset, where X contains the features of the flowers,
# and y the types of flowers
iris = load_iris()
X, y = iris.data, iris.target
num_classes = len(set(y))
num_datapoints = len(X)
num_features = len(X[0])


# Separate data set into train and test sets
num_train = int(0.8 * len(X))
X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# Initialize a Multi Layer Perceptron with 16 hidden units
mlp = MultiLayerPerceptron(
    input_size=num_features, hidden_size=8, output_size=num_classes
)

# Train the Multi Layer Perceptron
mlp.train(X_train, y_train, epochs=100000, learning_rate=0.01)

# Predict on the test data and print accuracy
y_pred = mlp.predict(X_test)
print("Accuracy on test data:", accuracy_fn(y_test, y_pred))
