import torch
import numpy as np

from actor_critic import ActorCritic

from validate import validate

from ppo import PPO

from params import config

validation_data = np.array([
    [
  [
    [5, 0, 4, 0, 0, 0],
    [0, 1, 5, 0, 3, 0],
    [0, 0, 4, 0, 0, 2],
    [1, 6, 0, 0, 0, 5],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 6, 3, 0, 6]
  ],
  [
    [0, 6, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [2, 0, 0, 0, 0, 0],
    [0, 6, 0, 6, 0, 0],
    [1, 6, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0]
  ],
  [
    [0, 6, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 2],
    [1, 6, 0, 0, 0, 5],
    [0, 6, 4, 0, 0, 6],
    [1, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0]
  ],
  [
    [1, 6, 0, 0, 0, 5],
    [0, 6, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 5, 0, 3, 0],
    [0, 0, 4, 0, 0, 2],
    [0, 0, 0, 0, 0, 0]
  ],
  [
    [0, 1, 5, 0, 3, 0],
    [1, 6, 0, 0, 0, 5],
    [0, 6, 0, 0, 0, 0],
    [5, 0, 4, 0, 0, 0],
    [0, 6, 0, 6, 0, 0],
    [0, 6, 4, 0, 0, 6]
  ],
  [
    [0, 0, 4, 0, 0, 2],
    [2, 0, 0, 0, 0, 0],
    [0, 6, 4, 0, 0, 6],
    [0, 6, 0, 0, 0, 0],
    [1, 6, 0, 0, 0, 5],
    [3, 0, 0, 2, 0, 0]
  ],
  [
    [0, 0, 0, 0, 0, 1],
    [3, 0, 0, 2, 0, 0],
    [0, 6, 4, 0, 0, 6],
    [6, 6, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ],
  [
    [0, 0, 4, 0, 0, 2],
    [0, 6, 4, 0, 0, 6],
    [1, 6, 0, 0, 0, 5],
    [0, 6, 0, 0, 0, 0],
    [0, 6, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ],
  [
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 5, 0],
    [0, 0, 6, 3, 0, 6],
    [2, 0, 0, 0, 0, 0],
    [0, 6, 4, 0, 0, 6],
    [0, 6, 0, 6, 0, 0]
  ],
  [
    [0, 0, 4, 0, 0, 2],
    [0, 6, 4, 0, 0, 6],
    [0, 1, 5, 0, 3, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 6, 0, 6, 0, 0],
    [3, 0, 0, 2, 0, 0]
  ]
]
], dtype=np.int32)

'''x
data_loaded = np.load(f'./validation/mk01_validation_set_4.npy')
validation_data = []

for i in range(data_loaded.shape[0]):
    validation_data.append(data_loaded[i])
'''


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

ppo.policy.load_state_dict(torch.load('./weights/10_6_1_6_55.pth'))

makespans = validate(validation_set=validation_data, model=ppo.policy, ub_num_of_operations_per_job=config.num_of_operations_ub_per_job)
print(makespans)

