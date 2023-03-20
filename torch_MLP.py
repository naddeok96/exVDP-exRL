import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) with ReLU activation in the
    hidden layer and Softmax activation in the output layer.

    Args:
        input_dim (int): Size of the input layer.
        hidden_dim (int): Size of the hidden layer.
        output_dim (int): Number of classes for classification.

    Attributes:
        fc1 (Linear): The first fully-connected layer.
        relu (ReLU): The ReLU activation function.
        fc2 (Linear): The second fully-connected layer.
        softmax (Softmax): The Softmax activation function.

    Methods:
        forward(x): Performs a forward pass through the network.

    """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).

        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
