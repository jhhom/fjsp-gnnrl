import torch

from params import config, device

num_of_nodes = config.num_of_training_operations

element = torch.full(size=(1 * num_of_nodes, 1), fill_value=(1/num_of_nodes), dtype=torch.float64).view(-1)
index_0 = torch.arange(start=0, end=1, dtype=torch.long).repeat(num_of_nodes, 1).t().reshape((1*num_of_nodes, 1)).squeeze()
index_1 = torch.arange(start=0, end=num_of_nodes*1, dtype=torch.long)
index = torch.stack((index_0, index_1))

graph_pool_step = torch.sparse.FloatTensor(index, element, torch.Size([1, num_of_nodes * 1])).to(device)

def get_graph_pool_mb(batch_size):
    element = torch.full(size=(batch_size[0] * num_of_nodes, 1), fill_value=(1/num_of_nodes), dtype=torch.float64).view(-1)
    index_0 = torch.arange(start=0, end=batch_size[0], device=device, dtype=torch.long) \
        .repeat(num_of_nodes, 1).t() \
        .reshape((batch_size[0]*num_of_nodes, 1)) \
        .squeeze()
    index_1 = torch.arange(start=0, end=num_of_nodes * batch_size[0], dtype=torch.long)
    index = torch.stack((index_0, index_1))

    return torch.sparse.FloatTensor(index, element, torch.Size([batch_size[0], num_of_nodes * batch_size[0]])).to(device)
