import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPActor(nn.Module):
    def __init__(self, num_of_layers, input_dim, hidden_dim, output_dim):
        '''
        num_of_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_of_layers = 1, this reduces to a linear layer.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden layers
        output_dim: number of classes for prediction
        '''
        super(MLPActor, self).__init__()

        self.num_of_layers = num_of_layers
    
        if num_of_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_of_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_of_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
        
        # 128 (input_dim) * 32 (hidden_dim) is the weights size
        # 60 * 132 is the input size
        # 60 candidates * 132 feat dimensions

    def forward(self, x):
        if self.num_of_layers == 1:
            return self.linear(x)
        
        h = x
        # X SHAPE SHOULD BE (1, candidate_size, 128)
        # candidate size = num_of_jobs * num_of_machines (alternatives)
        for layer in range(self.num_of_layers - 1):
            h = self.linears[layer](h)
            # h = F.relu(h)
            h = torch.tanh(h)
        
        h = self.linears[self.num_of_layers - 1](h)
        return h

    
    
    