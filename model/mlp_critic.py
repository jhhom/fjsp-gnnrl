import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPCritic(nn.Module):
    def __init__(self, num_of_layers, input_dim, hidden_dim, output_dim):
        '''
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        '''
        super(MLPCritic, self).__init__()

        self.num_of_layers = num_of_layers

        if self.num_of_layers < 1:
            raise ValueError("number of layers should be positive")
        elif num_of_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_of_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
        
    
    def forward(self, x):
        if self.num_of_layers == 1:
            return self.linear(x)

        h = x
        for layer in range(self.num_of_layers - 1):
            h = self.linears[layer](h)
            h = torch.tanh(h)

            # h = F.relu(h)
        return self.linears[self.num_of_layers - 1](h)
