import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_of_layers, input_dim, hidden_dim, output_dim):
        '''
        num_layers: Number of layers in the MLP (excluding the input layer). If num_of_layers=1, this reduces to linear layer
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of all hidden layers
        output_dimension: number of classes for prediction
        '''
        super(MLP, self).__init__()

        self.num_of_layers = num_of_layers

        if num_of_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_of_layers == 1:
            self.linear == nn.Linear(input_dim, output_dim)
        else:
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            # Input layer
            self.linears.append(nn.Linear(input_dim, hidden_dim))

            # Hidden layer
            for layer in range(num_of_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Output layer
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_of_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        
    def forward(self, x):
        if self.num_of_layers == 1:
            return self.linear(x)
        
        h = x
        for layer in range(self.num_of_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_of_layers - 1](h)
        
