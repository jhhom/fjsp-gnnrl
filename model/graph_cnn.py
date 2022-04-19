import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import GIN

from .mlp import MLP

class GraphCNN(nn.Module):
    def __init__(
            self,
            num_of_layers,
            num_of_mlp_layers,
            input_dim,
            hidden_dim,
            use_learn_epsilon):
        '''
        num_of_layers: Number of layers in neural networks (including the input layer)
        num_of_mlp_layers: Number of layers in MLPs (excluding the input layer)        
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        use_learn_epsilon: If True, learn epsilon to distinguish center nodes from neighbouring nodes
        '''

        super(GraphCNN, self).__init__()

        self.num_of_layers = num_of_layers
        self.use_learn_epsilon = use_learn_epsilon

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        # self.gin = GIN(input_dim, hidden_dim, self.num_of_layers, hidden_dim)

        # List of batch norms apllied to output of MLP
        # input of the final prediction linear layer
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_of_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_of_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_of_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def forward(self, x, adj_matrix, graph_pool):
        h = x

        # h = self.gin(x, adj_matrix)

        for layer in range(self.num_of_layers-1):
            h = self.next_layer(h, layer, adj_matrix=adj_matrix)

        h_nodes = h.clone()
        
        # pooling over entire graph
        pooled_h = torch.sparse.mm(graph_pool, h.double())

        return pooled_h, h_nodes



    def next_layer(self, h, layer, adj_matrix):
        pooled = torch.mm(adj_matrix, h)    # 1. aggregate neighbour nodes (SUM)
        if self.use_learn_epsilon:          # Optional: Reweight center node when aggregating with neighbours
            pooled = pooled + (1 + self.eps[layer]) * h
        h = self.mlps[layer](pooled)             # 2. pass to MLP
        h = self.batch_norms[layer](h)      # 3. normalize
        h = F.relu(h)                       # 4. activation
        return h
    


    
    

        
        
