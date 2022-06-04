import torch

from params import device

def aggregate_observations(observation_minibatch, n_node):
    # observation_minibatch is [m, n_nodes_each_state, fea_dim]
    # m is the number of nodes in batch
    n_node = observation_minibatch.shape[0]
    indexes = observation_minibatch.coalesce().indices()
    values = observation_minibatch.coalesce().values()
    new_index_row = indexes[1] + indexes[0] * n_node
    new_index_col = indexes[2] + indexes[0] * n_node
    index_minibatch = torch.stack((new_index_row, new_index_col))
    adjacency_batch = torch.sparse.FloatTensor(
        indices=index_minibatch,
        values=values,
        size=torch.Size([observation_minibatch.shape[0] * n_node,
                         observation_minibatch.shape[0] * n_node])
    ).to(device)
    return adjacency_batch
