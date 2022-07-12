import torch
import numpy as np

from validate import validate
from uniform_instance_gen import datasetConfigs

from ppo import PPO

from params import config


# problem_size is only 1 to 10
def new_model(problem_size, weight_id):
    if problem_size < 1 or problem_size > 10:
        raise ValueError('problem_size must be an integer between 1 and 10')

    problem_size = '{:02d}'.format(problem_size)
    problem_config = datasetConfigs[f'MK{problem_size}']
    config.n_j = problem_config.num_of_jobs
    config.n_m = problem_config.num_of_machines
    config.input_dim = config.n_m + 1

    ppo = PPO(
        lr=config.learning_rate,
        gamma=config.gamma,
        k_epochs=config.k_epochs,
        eps_clip=config.epsilon_clip,
        n_j=config.n_j,
        n_m=config.n_m,
        num_of_layers=config.num_of_layers,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_of_mlp_layers_feature_extract=config.num_of_mlp_layers_feature_extract,
        num_of_mlp_layers_actor=config.num_of_mlp_layers_actor,
        hidden_dim_actor=config.num_of_hidden_dim_actor,
        num_of_mlp_layers_critic=config.num_of_mlp_layers_critic,
        hidden_dim_critic=config.num_of_hidden_dim_critic
    )
    path_to_weight = f'./records/MK{problem_size}/ID_{weight_id}/best_weight.pth'
    ppo.policy.load_state_dict(torch.load(path_to_weight, map_location=torch.device('cpu')))
    return ppo.policy


if __name__ == '__main__':
    import json

    # weight id for MK01 to MK10
    weight_ids = {
        1: 2,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 2,
    }

    results = {}
    for problem_size in range(1, 2):
        weight_id = weight_ids[problem_size]

        model = new_model(problem_size, weight_id)
        problems = np.load('./evaluations/standard/boxplot/dataset/MK{:02d}_12.npy'.format(problem_size))
        makespans = validate(validation_set=problems, model=model, ub_num_of_operations_per_job=config.num_of_operations_ub_per_job, release_times=None)
        results['MK{:02d}'.format(problem_size)] = (makespans * -1).tolist()
    json.dump(results, open('./evaluations/standard/boxplot/gnn_results_2.json', 'w'))

