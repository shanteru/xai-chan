import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        
        # First fully connected layer: 784 input features and 128 output features
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Second fully connected layer: 128 input features and 10 output features
        self.fc2 = nn.Linear(in_features=128, out_features=10)
    def forward(self, x):
        x = self.fc1(x)  # Pass input through the first fully connected layer
        x = self.relu(x) # Apply ReLU activation function
        x = self.fc2(x)  # Pass result through the second fully connected layer
        return x
# Create an instance of the network
network = MyNeuralNetwork()

# Print the structure of the network
print(network)
