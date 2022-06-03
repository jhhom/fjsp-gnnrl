import torch
import numpy as np

from validate import validate

from ppo import PPO

from params import config

# problem_size is only 01 to 10
problem_size = '04'
weight_id = 1

test_set_path = f'./evaluations/benchmark/brandimarte/Mk{problem_size}.fjs.npy'
weights_path = f'./records/MK{problem_size}/ID_{weight_id}/best_weight.pth'

test_set = np.load(test_set_path)
test_set = np.array([test_set], dtype=np.int32)


torch.manual_seed(config.torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.torch_seed)

np.random.seed(200)

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

ppo.policy.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

makespans = validate(validation_set=test_set, model=ppo.policy, ub_num_of_operations_per_job=config.num_of_operations_ub_per_job)
print(makespans)

# mk02 47
# mk03 264
# mk04 109
# mk05 205
# mk06 115
# mk08 782 (can't be used)
# mk09 470