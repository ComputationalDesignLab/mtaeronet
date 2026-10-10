import torch
import torch.nn as nn
from dataclasses import dataclass, asdict

def train_net(model, train_x, train_y, epochs, learning_rate):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()

    return model

@dataclass
class MLPHyperparameters:
    hidden_layers: list
    input_dim: int
    output_dim: int
    activation: callable
    
class MLP(nn.Module):

    def __init__(self, hyperparameters):
        super().__init__()
        self.network = self.create_network(**asdict(hyperparameters))
        self.hyperparameters = hyperparameters

    def create_network(self, input_dim, hidden_layers, output_dim, activation):
        network_def = []
        network_def.append(nn.Linear(input_dim, hidden_layers[0]))
        network_def.append(activation)
        for i in range(1, len(hidden_layers)):
            network_def.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            network_def.append(activation)
        network_def.append(nn.Linear(hidden_layers[-1], output_dim))
        return nn.Sequential(*network_def)

    def forward(self, x):
        return self.network(x)