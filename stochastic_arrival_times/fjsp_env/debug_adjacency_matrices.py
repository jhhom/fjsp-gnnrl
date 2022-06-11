import math
import networkx as nx
import numpy as np
from .fjsp_env import FJSP

import matplotlib.pyplot as plt

def matrix_to_list(matrix, op_id_to_job_info):
    list = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                source = op_id_to_job_info[i]
                target = op_id_to_job_info[j]
                source = f'{source[0]+1}{source[1]+1}'
                target = f'{target[0]+1}{target[1]+1}'
                list.append((source, target))
    return list


def node_name_to_position(name, highest_index):
    x = highest_index - (int(name[0]) - 1)
    y = int(name[1]) - 1
    return [y, x]


def get_op_dispatch_statuses(feat):
    return [i*1000 > 0 for i in feat[:, -1]]


def draw_graph(adj_matrix, op_id_to_job_info, op_dispatch_statuses):
    G = nx.DiGraph()

    V = [f'{i+1}{j+1}' for (i, j) in op_id_to_job_info]
    E = matrix_to_list(adj_matrix, op_id_to_job_info)

    G.add_nodes_from(V)
    G.add_edges_from(E)
    
    node_positions = { i: node_name_to_position(i, 2) for i in V}

    color_map = ['#BBF7D0' if dispatch else '#E2E8F0' for dispatch in op_dispatch_statuses]

    nx.draw(
        G,
        node_positions,
        with_labels=True,
        font_size=15,
        node_size=1000,
        edge_color='gray',
        arrowsize=20,
        node_color=color_map
    )


if __name__ == '__main__':
    G = nx.DiGraph()

    instance = np.array([[
        [0, 0, 3],
        [1, 0, 0],
        [0, 0, 0]],

       [[0, 0, 2],
        [0, 0, 1],
        [1, 0, 0]],

       [[3, 0, 0],
        [0, 0, 2],
        [0, 3, 0]]], dtype=np.int32)
    V = ['11', '12', '21', '22', '23', '31', '32', '33']

    env = FJSP(3, 3, 3)

    adj, feat, omega, mask, machine_feat = env.reset(instance, 3)
    adj, feat, reward, done, omega, mask, machine_feat = env.step((0, 2))
    adj, feat, reward, done, omega, mask, machine_feat = env.step((2, 2))
    dispatch_statuses = [i*1000 > 0 for i in feat[:, -1]]
    draw_graph(env.adj_matrix, env.op_id_to_job_info, dispatch_statuses)
    plt.show()
    '''
    E = matrix_to_list(env.adj_matrix, env.op_id_to_job_info)

    G.add_nodes_from(V)
    G.add_edges_from(E)
    
    node_positions = { i: node_name_to_position(i, 2) for i in V}
    print(node_positions)
    nx.draw(G, node_positions, with_labels=True, font_size=15, node_size=400, edge_color='gray', arrowsize=30, node_color='#E2E8F0')
    plt.show()
    '''