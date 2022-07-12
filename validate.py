from params import config
import numpy as np
import torch

from ppo import PPO

def validate(validation_set, model, ub_num_of_operations_per_job, release_times):
    N_JOBS = validation_set[0].shape[0]

    N_MACHINES = validation_set[0].shape[2]

    import numpy as np
    import torch

    from fjsp_env.fjsp_env import FJSP
    from agent_utils import greedy_select_action
    from graph_pool import get_graph_pool_step
    from params import device

    if release_times != None:
        from stochastic_arrival_times.fjsp_env.fjsp_env import StochasticFJSP
        FJSP = StochasticFJSP

    env = FJSP(n_j=N_JOBS, n_m=N_MACHINES, num_of_operations_ub_per_job=ub_num_of_operations_per_job)

    makespans = []

    for i, data in enumerate(validation_set):
        if release_times != None:
            adj, fea, candidate, mask, machine_feat = env.reset(data, ub_num_of_operations_per_job, release_times[i])
        else:
            adj, fea, candidate, mask, machine_feat = env.reset(data, ub_num_of_operations_per_job)
        graph_pool_step = get_graph_pool_step(env.num_of_operations)
        rewards = -env.initial_quality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            machine_feat_tensor = torch.from_numpy(np.copy(machine_feat)).to(device)
        
            with torch.no_grad():
                pi, _ = model(
                    x=fea_tensor,
                    adj_matrix=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0),
                    graph_pool=graph_pool_step,
                    machine_feat=machine_feat_tensor.unsqueeze(0)
                )
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask, machine_feat = env.step(action)
            rewards += reward
            if done:
                break
        makespans.append(rewards - env.positive_rewards)
    
    return np.array(makespans)


def validate_and_get_environment(validation, model, ub_num_of_operations_per_job, release_times):
    N_JOBS = validation.shape[0]

    N_MACHINES = validation.shape[2]

    import numpy as np
    import torch

    from fjsp_env.fjsp_env import FJSP
    from agent_utils import greedy_select_action
    from graph_pool import get_graph_pool_step
    from params import device

    if release_times != None:
        from stochastic_arrival_times.fjsp_env.fjsp_env import StochasticFJSP
        FJSP = StochasticFJSP

    env = FJSP(n_j=N_JOBS, n_m=N_MACHINES, num_of_operations_ub_per_job=ub_num_of_operations_per_job)

    makespan = 0

    if release_times != None:
        adj, fea, candidate, mask, machine_feat = env.reset(validation, ub_num_of_operations_per_job, release_times)
    else:
        adj, fea, candidate, mask, machine_feat = env.reset(validation, ub_num_of_operations_per_job)
    graph_pool_step = get_graph_pool_step(env.num_of_operations)
    rewards = -env.initial_quality
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        machine_feat_tensor = torch.from_numpy(np.copy(machine_feat)).to(device)
    
        with torch.no_grad():
            pi, _ = model(
                x=fea_tensor,
                adj_matrix=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0),
                graph_pool=graph_pool_step,
                machine_feat=machine_feat_tensor.unsqueeze(0)
            )
        action = greedy_select_action(pi, candidate)
        adj, fea, reward, done, candidate, mask, machine_feat = env.step(action)
        rewards += reward
        if done:
            break
    makespan += (rewards - env.positive_rewards)
    
    return makespan, env





if __name__ == '__main__':
    data_loaded = np.load(f'./validation/mk01_validation_set_4.npy')
    validation_data = []

    for i in range(data_loaded.shape[0]):
        validation_data.append(data_loaded[i])


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

    validate(validation_set=validation_data, model=ppo.policy, ub_num_of_operations_per_job=config.num_of_operations_ub_per_job)
