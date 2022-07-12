import numpy as np
import torch
import json

from ppo import PPO
from stochastic_arrival_times.fjsp_env.fjsp_env import StochasticFJSP
from uniform_instance_gen import datasetConfigs
from validate import validate

from params import config

'''
Results

MK01 73
MK02 64
MK03 368
MK04 110
MK05 272
MK06 171
MK07 348
MK08 671
MK09 569
MK10 441
'''


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
    path_to_weight = f'./stochastic_records/MK{problem_size}/ID_{weight_id}/best_weight.pth'
    ppo.policy.load_state_dict(torch.load(path_to_weight, map_location=torch.device('cpu')))
    return ppo.policy


if __name__ == '__main__':
    problem_size = 10
    problem = np.load('./evaluations/standard/brandimarte/brandimarte_dataset_numpy/Mk{:02d}.fjs.npy'.format(problem_size))

    weights_config = json.load(open('./evaluations/weights.json'))

    env = StochasticFJSP(
        n_j=problem.shape[0],
        n_m=problem.shape[2],
        num_of_operations_ub_per_job=problem.shape[1])

    release_times = open('./stochastic_arrival_times/brandimarte/MK{:02d}_AR.txt'.format(problem_size)).read().split(' ')
    release_times = [x.replace('\n', '') for x in release_times]
    release_times = [int(x) for x in release_times]
    env.reset(problem, ub_num_of_operations_per_job=problem.shape[1], release_times=release_times)
    id = weights_config['stochastic']['MK{:02d}'.format(problem_size)]

    model = new_model(problem_size, id)
    problem = np.array( [problem] , dtype=np.int32)

    makespan = validate(problem, model, problem.shape[1], [release_times])
    print(makespan)