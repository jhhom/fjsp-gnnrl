from platform import release
from agent_utils import select_action
from fjsp_env.fjsp_env import FJSP
from stochastic_arrival_times.fjsp_env.stochastic_arrival_times import calculate_problem_release_times
from params import config, device
from ppo import PPO
from memory import Memory
from graph_pool import get_graph_pool_step
from save_progress import save_progress

from uniform_instance_gen import uniform_instance_gen
from validate import validate

import copy
import torch
import numpy as np
import datetime
import time
import os

if config.stochastic:
    from stochastic_arrival_times.fjsp_env.fjsp_env import StochasticFJSP
    FJSP = StochasticFJSP

def train():
    envs = [FJSP(n_j=config.n_j, n_m=config.n_m, num_of_operations_ub_per_job=config.num_of_operations_ub_per_job) for _ in range(config.num_of_envs)]
    memories = [Memory() for _ in range(config.num_of_envs)]

    validation_data_path = f'./validation/{config.size}_validation_set_4.npy'
    if config.stochastic:
        validation_data_path = f'./stochastic_arrival_times/validation/job_durations/{config.size}_validation_set_4.npy'


    data_loaded = np.load(validation_data_path)
    release_times = None
    if config.stochastic:
        release_times = np.loadtxt(f'./stochastic_arrival_times/validation/job_release_times/{config.size}_0.95.txt').astype(np.int32)

    validation_data = []

    if config.progress_config.save_training:
        if not os.path.isdir(config.progress_config.path_to_save_progress):
            os.makedirs(os.path.dirname(f'{config.progress_config.path_to_save_progress}/'), exist_ok=True)
    elif os.path.isdir(config.progress_config.path_to_save_progress):
        if len(os.listdir(f'{config.progress_config.path_to_save_progress}/')) != 0:
            print(f"ERROR: {os.path.dirname(config.progress_config.path_to_save_progress)} is not empty")
            quit()
    else:
        os.makedirs(os.path.dirname(f'{config.progress_config.path_to_save_progress}/'), exist_ok=True)

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

    training_log = []
    validation_log = []
    record = 100_000
    training_iteration = 0

    if config.progress_config.save_training:
        if len(os.listdir(config.progress_config.path_to_save_progress)) != 0:
            checkpoint = torch.load(f'{config.progress_config.path_to_save_progress}/saved.pth')
            training_log = checkpoint['training_log']
            validation_log = checkpoint['validation_log']
            record = checkpoint['best_record']
            training_iteration = len(checkpoint['training_log'])
            ppo.policy.load_state_dict(checkpoint['model_state_dict'])
            ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_training = time.time()
    for i_update in range(training_iteration, config.max_updates):
        ep_rewards = [0 for _ in range(config.num_of_envs)]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        machine_feat_envs = []
        env_indexes_envs = []
        graph_pool_step_envs = []

        # INITIALIZE ALL ENVIRONMENTS
        for i, env in enumerate(envs):
            problem = uniform_instance_gen(
                num_of_jobs=config.n_j,
                num_of_machines=config.n_m,
                lowest_num_of_operation_per_job=config.num_of_operations_lb_per_job,
                highest_num_of_operation_per_job=config.num_of_operations_ub_per_job,
                lowest_num_of_alternatives_per_op=config.num_of_alternatives_lb,
                highest_num_of_alternatives_per_op=config.num_of_alternatives_ub,
                duration_lb=config.duration_low,
                duration_ub=config.duration_high
            )
            if config.stochastic:
                adj, fea, candidate, mask, machine_feat = env.reset(
                    problem,
                    config.num_of_operations_ub_per_job,
                    calculate_problem_release_times(problem, config.machine_utilisation, config.n_j, config.n_m)
                )
            else:
                adj, fea, candidate, mask, machine_feat = env.reset(problem, config.num_of_operations_ub_per_job)

            
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            machine_feat_envs.append(machine_feat)
            graph_pool_step_envs.append(get_graph_pool_step(env.num_of_operations))
            env_indexes_envs.append(i)
            ep_rewards[i] = - env.initial_quality
        
        # COLLECT EXPERIENCES FOR ENTIRES EPISODE
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            machine_feat_tensor_envs = [torch.from_numpy(np.copy(machine_feat)).to(device) for machine_feat in machine_feat_envs]
            env_indexes_before_envs = copy.deepcopy(env_indexes_envs)

            with torch.no_grad():
                action_envs = []
                action_index_envs = []

                for i in range(len(fea_tensor_envs)):
                    pi, _ = ppo.policy_old(
                        x=fea_tensor_envs[i],
                        adj_matrix=adj_tensor_envs[i],
                        candidate=candidate_tensor_envs[i].unsqueeze(0),
                        mask=mask_tensor_envs[i].unsqueeze(0),
                        graph_pool=graph_pool_step_envs[env_indexes_before_envs[i]],
                        machine_feat=machine_feat_tensor_envs[i].unsqueeze(0),
                    )
                    action, action_index = select_action(pi, candidate_envs[i], memories[env_indexes_before_envs[i]])
                    action_envs.append(action)
                    action_index_envs.append(action_index)
            
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            machine_feat_envs = []
            env_indexes_envs = []

            # QUESTION: if let's say that environment 3 is done, but environment 1, 2, 4 is not done yet
            # there will be 3 elements in the array
            # because we iterate over 4 environments, hence on the 4th element, we get index out of bounds
            # but even if we don't get index out of bounds, how do we know, which environment's memories to insert
            # because if environment 3 is done, the fourth environment's content will be in index 3
            # so Index no longer maps or implies directly the environment

            # One solution is to have a parallel array, where every element in the array indicates the environment of the content
            # Example: environments = [0, 1, 3]
            # Other contents: adj = [adj0, adj1, adj3]
            # so we can iterate over this environment, and insert into its corresponding memories
            # we then need to clear this environment every time

            # INSERT EVERY EXPERIENCE FROM EVERY ENVIRONMENT INTO THE MEMORIES
            for i in range(len(fea_tensor_envs)):
                i_memory = env_indexes_before_envs[i]
                memories[i_memory].adj_mb.append(adj_tensor_envs[i])
                memories[i_memory].fea_mb.append(fea_tensor_envs[i])
                memories[i_memory].candidate_mb.append(candidate_tensor_envs[i])
                memories[i_memory].mask_mb.append(mask_tensor_envs[i])
                memories[i_memory].a_mb.append(action_index_envs[i])
                memories[i_memory].machine_feat_mb.append(machine_feat_tensor_envs[i])
            
                adj, fea, reward, done, candidate, mask, machine_feat = envs[i_memory].step(action_envs[i])
                if not done:
                    adj_envs.append(adj)
                    fea_envs.append(fea)
                    candidate_envs.append(candidate)
                    mask_envs.append(mask)
                    machine_feat_envs.append(machine_feat)
                    env_indexes_envs.append(i_memory)

                ep_rewards[i] += reward

                memories[i_memory].r_mb.append(reward)
                memories[i_memory].done_mb.append(done)
            
            # FINISH EPISODE
            if all([env.done() for env in envs]):
                break
        
        for j in range(config.num_of_envs):
            ep_rewards[j] -= envs[j].positive_rewards

        num_of_training_operations = [env.num_of_operations for env in envs]
        _, v_loss = ppo.update(memories, num_of_training_operations)
        for memory in memories: memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)

        training_log.append([i_update, mean_rewards_all_env])
        print(f'Episode {i_update+1} \t Last reward: {mean_rewards_all_env:.2f} \t Mean V Loss: {v_loss:.8f}')

        if (i_update + 1) % 100 == 0:
            validation_result = - validate(
                    validation_set=validation_data, 
                    model=ppo.policy,
                    ub_num_of_operations_per_job=config.num_of_operations_ub_per_job,
                    release_times=release_times).mean()

            print(f'The validation quality is: {validation_result}')

            save_progress(
                training_log=training_log,
                validation_log=validation_log,
                validation_result=validation_result,
                record=record,
                model=ppo
            )

            if validation_result < record:
                record = validation_result
    end_training = time.time()

    with open(f'{config.progress_config.path_to_save_progress}/training_duration.txt', 'w') as logfile:
        str = f'''Start training
Timestamp: {datetime.datetime.fromtimestamp(start_training)}
Time: {start_training}

End training
Timestamp: {datetime.datetime.fromtimestamp(end_training)}
Time: {end_training}

Training duration: {end_training - start_training} seconds
        '''
        logfile.write(str)

if __name__ == '__main__':
    train()
