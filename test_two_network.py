import torch
import torch.nn as nn
import torch.optim as optim


# Define the first MLP network
class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the second MLP network
class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the networks
input_size = 10
hidden_size = 20
output_size = 5

mlp1 = MLP1(input_size, hidden_size, output_size)
mlp2 = MLP2(output_size, hidden_size, output_size)

# Generate random input data
input_data = torch.randn(1, input_size)

# Forward pass through the first network
output_mlp1 = mlp1(input_data)

# Forward pass through the second network using the output of the first network as input
output_mlp2 = mlp2(output_mlp1)

# Compute the gradients
mlp1.zero_grad()
mlp2.zero_grad()

loss = torch.sum(output_mlp2)  # Example loss
loss.backward()

# Access gradients
gradients_mlp1 = {name: param.grad for name, param in mlp1.named_parameters()}
gradients_mlp2 = {name: param.grad for name, param in mlp2.named_parameters()}

print("Gradients of MLP1:")
print(gradients_mlp1)

print("\nGradients of MLP2:")
print(gradients_mlp2)