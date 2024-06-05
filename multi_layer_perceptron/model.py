from typing import List

import matplotlib.pyplot as plt
import numpy as np
from helper_functions import (
    ReLU,
    ReLU_derivative,
    accuracy_fn,
    cross_entropy_derivative,
    cross_entropy_loss,
    softmax,
    xavier_initialization,
)


class LinearLayer:
    """Perform a linear transformation on input data"""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """
        Initialize a linear layer with num_inputs input neurons
        and num_ouputs output neurons.
        """
        # Use Xavier initialization for initializing weights
        self.weights = xavier_initialization(num_inputs, num_outputs)
        self.biases = np.random.random((1, num_outputs))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform linear transformation on input data x"""
        self.inputs = x
        self.outputs = np.dot(x, self.weights) + self.biases
        return self.outputs

    def backward(self, loss: np.ndarray, learning_rate: float) -> np.ndarray:
        d_loss_d_weights = np.dot(self.inputs.T, loss)
        d_loss_d_biases = np.sum(loss, axis=0, keepdims=True)

        # Gradient clipping: Ensure numerical stability
        np.clip(d_loss_d_weights, -1, 1, out=d_loss_d_weights)
        np.clip(d_loss_d_biases, -1, 1, out=d_loss_d_biases)

        self.weights -= learning_rate * d_loss_d_weights
        self.biases -= learning_rate * d_loss_d_biases

        d_loss_d_inputs = np.dot(loss, self.weights.T)

        return d_loss_d_inputs


class MultiLayerPerceptron:
    """Multi Layer Perceptron / Feed Fordward Neural Network"""

    def __init__(self, n_features, n_hidden_layers, hidden_size, n_classes) -> None:
        """
        Initialize neural network with n_features input neurons,
        hidden_size hidden_units and output_size output neurons.
        """
        self.layers: List[LinearLayer] = []

        # Initialize the input layer
        self.layers.append(LinearLayer(n_features, hidden_size))

        # Initialize hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(LinearLayer(hidden_size, hidden_size))

        # Initialize output layer
        self.output_layer = LinearLayer(hidden_size, n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pass input data through the neural network and get prediction probabilities"""
        for layer in self.layers:
            x = layer.forward(x)
            x = ReLU(x)
        self.outputs = softmax(self.output_layer.forward(x))
        return self.outputs

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        Propagate error of the model backwards and update the
        parameters using gradient descent

        Arguments:
        x: training data to do backward propagation with
        y: labels of training data, used to determine the error of the model
        learning_rate: step size for gradient descent
        """
        loss = cross_entropy_derivative(y, self.forward(x))

        loss = self.output_layer.backward(loss, learning_rate)
        for layer in reversed(self.layers):
            loss *= ReLU_derivative(layer.outputs)
            loss = layer.backward(loss, learning_rate)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        print_every_n_epochs: int = 0,
    ) -> None:
        """
        Train neural network

        Arguments:
        X: input data (features)
        y: output data (labels / classes)
        epochs: number of training iterations
        learning_rate: step size for gradient descent
        print_every_n_epochs: n defines how often loss and
            accuracy are printed while training the model.
            Default: 0 (never print)
        """
        for epoch in range(epochs):  #
            # Predict on training data
            y_preds = self.predict(x)
            # Propagate training loss
            self.backward(x, y, learning_rate)

            if print_every_n_epochs != 0:
                if (epoch + 1) % print_every_n_epochs == 0:
                    # Calculate Loss
                    loss = cross_entropy_loss(y, self.forward(x))
                    # Print actual training accuracy and loss
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {loss}, Accuracy: {accuracy_fn(y, y_preds)}"
                    )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Classify input data"""
        output = self.forward(x)
        return np.argmax(output, axis=1)


def plot_decision_boundaries(
    model: MultiLayerPerceptron,
    x: np.ndarray,
    y: np.ndarray,
    grid_resolution=(501, 501),
) -> None:
    """
    Plot decision boundaries of the model

    Create a grid of data points and make predictions at those
    data points. Create a plot with the predictions of the model
    and add the evaluated data set to the graph.
    """
    # Setup prediction boundaries and grid
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution[0]),
        np.linspace(y_min, y_max, grid_resolution[1]),
    )

    # Make features
    x_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))

    # Make predictions
    y_pred = model.predict(x_to_pred_on)

    # Reshape preds and plot
    y_pred = np.reshape(y_pred, xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)

    # Add data set to the plot
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
