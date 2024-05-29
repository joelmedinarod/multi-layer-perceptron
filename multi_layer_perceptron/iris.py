from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model import MultiLayerPerceptron
from helper_functions import accuracy_fn

# Get IRIS Dataset, where X contains the features of the flowers,
# and y the types of flowers
iris = load_iris()
X, y = iris.data, iris.target
num_classes = len(set(y))
num_datapoints = len(X)
num_features = len(X[0])

# Split the dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Initialize a Multi Layer Perceptron with 16 hidden units
mlp = MultiLayerPerceptron(
    n_features=num_features, n_hidden_layers=1, hidden_size=16, n_classes=num_classes
)

# Train the Multi Layer Perceptron
mlp.train(X_train, y_train, epochs=10000, learning_rate=0.01)

# Predict on the test data and print accuracy
y_pred = mlp.predict(X_test)
print("Accuracy on test data:", accuracy_fn(y_test, y_pred))
