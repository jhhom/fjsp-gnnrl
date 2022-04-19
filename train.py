from agent_utils import select_action
from fjsp_env.fjsp_env import FJSP
from params import config, device, TRAINING_ONESHOT, TRAINING_RESUME, TRAINING_SAVE
from ppo import PPO
from memory import Memory
from graph_pool import graph_pool_step
from save_progress import save_progress

from uniform_instance_gen import uniform_instance_gen_with_fixed_num_of_operations
from validate import validate


import torch
import numpy as np
import datetime
import os

def train():
    envs = [FJSP(n_j=config.n_j, n_m=config.n_m, num_of_operations_ub_per_job=config.num_of_operations_ub_per_job) for _ in range(config.num_of_envs)]
    memories = [Memory() for _ in range(config.num_of_envs)]

    data_loaded = np.load(f'./validation/{config.size}_validation_set_4.npy')
    validation_data = []

    current_time = f'{datetime.datetime.now()}'

    if config.progress_config.training_mode == TRAINING_SAVE:
        os.makedirs(os.path.dirname(config.progress_config.path_to_save_progress))

    for i in range(data_loaded.shape[0]):
        validation_data.append(data_loaded[i])

    torch.manual_seed(config.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.torch_seed)

    np.random.seed(200)

    # TODO: change this, instead initialize like this, load from paused
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
    if config.progress_config.training_mode == TRAINING_RESUME or config.progress_config.training_mode == TRAINING_SAVE:
        checkpoint = torch.load(config.progress_config.path_to_save_progress)
        ppo.policy.load_state_dict(checkpoint['model_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    log = []
    validation_log = []
    record = 100_000

    # FOR THE NUMBER OF MAX EPISODES
    # TODO: change this - i_update in range(start, config.max_updates)
    for i_update in range(config.max_updates):
        ep_rewards = [0 for _ in range(config.num_of_envs)]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        machine_feat_envs = []

        # INITIALIZE ALL ENVIRONMENTS
        for i, env in enumerate(envs):
            adj, fea, candidate, mask, machine_feat = env.reset(uniform_instance_gen_with_fixed_num_of_operations(
                num_of_machines=config.n_m,
                durations_bounds=(config.duration_low, config.duration_high),
                highest_num_of_operation_per_job=config.num_of_operations_ub_per_job,
                num_of_alternatives_bounds=(config.num_of_alternatives_lb, config.num_of_alternatives_ub),
                num_of_jobs=config.n_j,
                num_of_operations_to_num_of_jobs=config.dataset_config.num_of_operations_to_num_of_jobs
            ), config.num_of_operations_ub_per_job)
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            machine_feat_envs.append(machine_feat)
            ep_rewards[i] = - env.initial_quality
        
        # COLLECT EXPERIENCES FOR ENTIRES EPISODE
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            machine_feat_tensor_envs = [torch.from_numpy(np.copy(machine_feat)).to(device) for machine_feat in machine_feat_envs]
            with torch.no_grad():
                action_envs = []
                action_index_envs = []

                for i in range(config.num_of_envs):
                    pi, _ = ppo.policy_old(
                        x=fea_tensor_envs[i],
                        adj_matrix=adj_tensor_envs[i],
                        candidate=candidate_tensor_envs[i].unsqueeze(0),
                        mask=mask_tensor_envs[i].unsqueeze(0),
                        graph_pool=graph_pool_step,
                        machine_feat=machine_feat_tensor_envs[i].unsqueeze(0),
                    )
                    action, action_index = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    action_index_envs.append(action_index)
            
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            machine_feat_envs = []

            # INSERT EVERY EXPERIENCE FROM EVERY ENVIRONMENT INTO THE MEMORIES
            for i in range(config.num_of_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(action_index_envs[i])
                memories[i].machine_feat_mb.append(machine_feat_tensor_envs[i])
            
                adj, fea, reward, done, candidate, mask, machine_feat = envs[i].step(action_envs[i])
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                machine_feat_envs.append(machine_feat)

                ep_rewards[i] += reward

                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
            
            # FINISH EPISODE
            if envs[0].done():
                break
        
        for j in range(config.num_of_envs):
            ep_rewards[j] -= envs[j].positive_rewards

        _, v_loss = ppo.update(memories, config.num_of_training_operations)
        for memory in memories: memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)

        log.append([i_update, mean_rewards_all_env])
        print(f'Episode {i_update+1} \t Last reward: {mean_rewards_all_env:.2f} \t Mean V Loss: {v_loss:.8f}')

        if (i_update + 1) % 100 == 0:
            validation_result = - validate(
                validation_set=validation_data, 
                model=ppo.policy,
                ub_num_of_operations_per_job=config.num_of_operations_ub_per_job).mean()
            save_progress(
                training_log=log,
                validation_log=validation_log,
                validation_result=validation_result,
                record=record,
                model=ppo
            )
    

if __name__ == '__main__':
    train()
