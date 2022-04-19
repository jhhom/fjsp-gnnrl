import torch
from uniform_instance_gen import DatasetConfig, datasetConfigs

TRAINING_RESUME = 'RESUME'
TRAINING_SAVE = 'SAVE'
TRAINING_ONESHOT = 'ONESHOT'

class ProgressConfig:
    # training mode: TRAINING_RESUME, TRAINING_SAVE, OR TRAINING_ONESHOT
    training_mode: str
    path_to_save_progress: str

class Config:
    num_of_envs: int
    size: str

    dataset_config: DatasetConfig

    n_j: int
    n_m: int
    num_of_operations_ub_per_job: int
    num_of_training_operations: int
    torch_seed: int

    learning_rate: float
    gamma: float        # discount factor
    k_epochs: int
    epsilon_clip: float

    num_of_layers: int
    input_dim: int
    hidden_dim: int
    num_of_mlp_layers_feature_extract: int
    num_of_mlp_layers_actor: int
    num_of_hidden_dim_actor: int
    num_of_mlp_layers_critic: int
    num_of_hidden_dim_critic: int
    
    max_updates: int

    duration_low: int
    duration_high: int

    device: str

    progress_config: ProgressConfig



config = Config()

# just change this
config.size = 'MK01'

datasetConfig = datasetConfigs[config.size]

config.dataset_config = datasetConfig

config.num_of_envs = 4
config.n_j = datasetConfig.num_of_jobs
config.n_m = datasetConfig.num_of_machines
config.num_of_operations_ub_per_job = datasetConfig.highest_num_of_operations_per_job
config.num_of_operations_lb_per_job = datasetConfig.lowest_num_of_operations_per_job
# used for graph pool
config.num_of_training_operations = datasetConfig.get_total_num_of_operations()
config.num_of_alternatives_lb = datasetConfig.num_of_alternative_bounds[0]
config.num_of_alternatives_ub = datasetConfig.num_of_alternative_bounds[1]

config.torch_seed = 600

config.learning_rate = 2e-5
config.gamma = 1
config.k_epochs = 1
config.epsilon_clip = 0.2

config.num_of_layers = 3
config.input_dim = config.n_m + 1
config.hidden_dim = 64
config.num_of_mlp_layers_feature_extract = 2
config.num_of_mlp_layers_actor = 2
config.num_of_hidden_dim_actor = 32
config.num_of_mlp_layers_critic = 2
config.num_of_hidden_dim_critic = 32

config.max_updates = 10_000

config.duration_low = datasetConfig.duration_bounds[0]
config.duration_high = datasetConfig.duration_bounds[1]
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(config.device)

config.progress_config.training_mode = TRAINING_SAVE
config.progress_config.path_to_save_progress = f'./records/{config.size}/ID_1'


'''
Notes:

The folder name will be MK01 / MK02
The subfolder name you can determine yourself


keep a fixed format

Suggestion:

<Experiment_ID>

Things to save

1. a serialized config object in the folder.
2. training log
3. validation log
4. best_weight
5. last_weight
6. last_optimizer_weights


'''