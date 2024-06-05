from helper_functions import accuracy_fn
from model import MultiLayerPerceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define model hyperparameters
N_HIDDEN_LAYERS = 2
HIDDEN_SIZE = 16  # number of neurons per hidden layer

# Define training hyperparameters
N_EPOCHS = 1000
LEARNING_RATE = 0.01
TEST_SPLIT = 0.3  # 30% of the data is used as test data

# Get IRIS Dataset, where X contains the features of the flowers,
# and y the types of flowers
iris = load_iris()
X, y = iris.data, iris.target
num_classes = len(set(y))
num_datapoints = len(X)
num_features = len(X[0])

# Split the dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=0
)

# Initialize a Multi Layer Perceptron with 16 hidden units
mlp = MultiLayerPerceptron(
    n_features=num_features,
    n_hidden_layers=N_HIDDEN_LAYERS,
    hidden_size=HIDDEN_SIZE,
    n_classes=num_classes,
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
