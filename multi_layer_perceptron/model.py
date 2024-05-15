import numpy as np
from helper_functions import (
    accuracy_fn,
    softmax,
    ReLU,
    ReLU_derivative,
    cross_entropy_derivative,
    cross_entropy_loss,
)
import matplotlib.pyplot as plt


class LinearLayer:
    """Perform a linear transformation on input data"""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """
        Initialize a linear layer with num_inputs input neurons
        and num_ouputs output neurons.
        """
        self.weights = np.random.random((num_inputs, num_outputs))
        self.biases = np.random.random((1, num_outputs))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform linear transformation on input data x"""
        return np.dot(x, self.weights) + self.biases


class MultiLayerPerceptron:
    """Multi Layer Perceptron / Feed Fordward Neural Network"""

    def __init__(self, input_size, hidden_size, output_size) -> None:
        """
        Initialize neural network with input_size input neurons,
        hidden_size hidden_units and output_size output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.input_layer = LinearLayer(input_size, hidden_size)
        self.hidden_layer = LinearLayer(hidden_size, hidden_size)
        self.output_layer = LinearLayer(hidden_size, output_size)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Pass input data through the neural network and get output"""
        self.input_sum = self.input_layer.forward(X)
        self.input_activation = ReLU(self.input_sum)
        self.hidden_sum = self.hidden_layer.forward(self.input_activation)
        self.hidden_activation = ReLU(self.hidden_sum)
        self.output_sum = self.output_layer.forward(self.hidden_activation)
        self.output_activation = softmax(self.output_sum)
        return self.output_activation

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        Propagate error of the model backwards and update the
        parameters using gradient descent

        Arguments:
        X: training data to do backward propagation with
        y: labels of training data, used to determine the error of the model
        learning_rate: step size for gradient descent
        """
        # Backward pass
        output_error = cross_entropy_derivative(y, self.output_activation)
        hidden_error = np.dot(
            output_error, self.output_layer.weights.T
        ) * ReLU_derivative(self.hidden_activation)
        input_error = np.dot(
            hidden_error, self.hidden_layer.weights.T
        ) * ReLU_derivative(self.input_activation)

        # Update weights and biases
        self.output_layer.weights -= learning_rate * np.dot(
            self.hidden_activation.T, output_error
        )
        self.output_layer.biases -= learning_rate * np.sum(
            output_error, axis=0, keepdims=True
        )
        self.hidden_layer.weights -= learning_rate * np.dot(
            self.input_activation.T, hidden_error
        )
        self.hidden_layer.biases -= learning_rate * np.sum(
            hidden_error, axis=0, keepdims=True
        )
        self.input_layer.weights -= learning_rate * np.dot(X.T, input_error)
        self.input_layer.biases -= learning_rate * np.sum(
            input_error, axis=0, keepdims=True
        )

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float
    ) -> None:
        """
        Train neural network

        Arguments:
        X: input data (features)
        y: output data (labels / classes)
        epochs: number of training iterations
        learning_rate: step size for gradient descent
        """
        for epoch in range(epochs):
            y_preds = self.predict(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 100 == 0:
                loss = cross_entropy_loss(y, self.forward(X))
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {loss}, Accuracy: {accuracy_fn(y, y_preds)}"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify input data"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def plot_decision_boundaries(
        self, X: np.ndarray, y: np.ndarray, grid_resolution=(501, 501)
    ) -> None:
        """
        Plot decision boundaries of the model 
        
        Create a grid of data points and make predictions at those
        data points. Create a plot with the predictions of the model
        and add the evaluated data set to the graph.
        """
        # Setup prediction boundaries and grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution[0]),
            np.linspace(y_min, y_max, grid_resolution[1]),
        )

        # Make features
        X_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))

        # Make predictions
        y_pred = self.predict(X_to_pred_on)

        # Reshape preds and plot
        y_pred = np.reshape(y_pred, xx.shape)
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)

        # Add data set to the plot
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example data (4 samples, 3 features, 3 classes)
    X = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.8, 0.2, 0.1], [0.5, 0.5, 0.5]])
    y = np.array([0, 1, 2, 1])

    # Create a Multi Layer Perceptron with 3 input neurons
    # for 3 features, 4 hidden neurons and 3 output neurons
    # for the 3 classes
    mlp = MultiLayerPerceptron(input_size=3, hidden_size=4, output_size=3)

    # Train the model
    mlp.train(X, y, epochs=1000, learning_rate=0.1)

    # Predict on new data
    predictions = mlp.predict(X)
    print("Predictions:", predictions)
