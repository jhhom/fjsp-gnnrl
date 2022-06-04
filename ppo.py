from matplotlib.pyplot import get
import torch
import torch.nn as nn
from copy import deepcopy


from actor_critic import ActorCritic
from aggregate_observation import aggregate_observations
from memory import Memory
from agent_utils import eval_actions
from graph_pool import get_graph_pool_mb
from params import device


class PPO:
    def __init__(
        self,
        lr,
        gamma,
        k_epochs,
        eps_clip,
        n_j,
        n_m,
        num_of_layers,
        input_dim,
        hidden_dim,
        num_of_mlp_layers_feature_extract,
        num_of_mlp_layers_actor,
        hidden_dim_actor,
        num_of_mlp_layers_critic,
        hidden_dim_critic,
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(
            n_j=n_j,
            n_m=n_m,
            num_of_layers=num_of_layers,
            use_learn_epsilon=False,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_of_mlp_layers_for_feature_extract=num_of_mlp_layers_feature_extract,
            num_of_mlp_layers_actor=num_of_mlp_layers_actor,
            num_of_mlp_layers_critic=num_of_mlp_layers_critic,
            hidden_dim_actor=hidden_dim_actor,
            hidden_dim_critic=hidden_dim_critic,
        )
        self.policy = self.policy.float()

        self.decay_step_size = 2000
        self.decay_ratio = 0.9

        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=self.decay_step_size,
                                                        gamma=self.decay_ratio)
        self.v_loss = nn.MSELoss()

        self.critic_loss_coefficient = 1.0
        self.policy_loss_coefficient = 2.0
        self.entropy_loss_coefficient = 0.01
        self.lr_decay_flag = False

    
    def update(self, memories: 'list[Memory]', n_operations):
        # array of minibatches over all environments
        # each minibatch contains the feature at the timestep for all environment
        # for example: Minibatch for action may contains actions at time step 2 for all environments
        all_env_mb_rewards = []
        all_env_mb_adj_matrices = []
        all_env_mb_features = []
        all_env_mb_candidate_features = []
        all_env_mb_masks = []
        all_env_mb_actions = []
        all_env_mb_old_logprobs = []
        all_env_mb_machine_feats = []

        # store data for all environments
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(
                reversed(memories[i].r_mb),
                reversed(memories[i].done_mb)
            ):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            all_env_mb_rewards.append(rewards)

            # process each env data
            all_env_mb_adj_matrices.append(
                aggregate_observations(torch.stack(memories[i].adj_mb).to(device), n_operations)
            )
            feature_minibatch_t = torch.stack(memories[i].fea_mb).to(device)
            feature_minibatch_t = feature_minibatch_t.reshape(-1, feature_minibatch_t.size(-1))
            all_env_mb_features.append(feature_minibatch_t)
            all_env_mb_candidate_features.append(
                torch.stack(memories[i].candidate_mb).to(device).squeeze()
            )
            all_env_mb_masks.append(
                torch.stack(memories[i].mask_mb).to(device).squeeze()
            )
            all_env_mb_actions.append(
                torch.stack(memories[i].a_mb).to(device).squeeze()
            )
            all_env_mb_machine_feats.append(
                torch.stack(memories[i].machine_feat_mb).to(device).squeeze()
            )
            all_env_mb_old_logprobs.append(
                torch.stack(memories[i].logprobs).to(device).squeeze().detach()
            )


        graph_pool_mbs = [get_graph_pool_mb(torch.stack(memories[k].adj_mb).to(device).shape, n_operations[k]) for k in range(len(memories))]
        for _ in range(self.k_epochs):
            loss_sum = 0
            v_loss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(
                    x=all_env_mb_features[i],
                    adj_matrix=all_env_mb_adj_matrices[i],
                    candidate=all_env_mb_candidate_features[i],
                    mask=all_env_mb_masks[i],
                    graph_pool=graph_pool_mbs[i],
                    machine_feat=all_env_mb_machine_feats[i],
                )
                logprobs, entropy_loss = eval_actions(pis.squeeze(), all_env_mb_actions[i])
                ratios = torch.exp(logprobs - all_env_mb_old_logprobs[i].detach())
                advantages = all_env_mb_rewards[i] - vals.view(-1).detach()
                surrogate_1 = ratios * advantages
                surrogate_2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.v_loss(vals.squeeze(), all_env_mb_rewards[i])
                p_loss = -torch.min(surrogate_1, surrogate_2).mean()
                entropy_loss = -entropy_loss.clone()
                loss = \
                    self.critic_loss_coefficient * v_loss + \
                    self.policy_loss_coefficient * p_loss + \
                    self.entropy_loss_coefficient * entropy_loss
                loss_sum += loss
                v_loss_sum += v_loss
            
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()
        
        # copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        if self.lr_decay_flag:
            self.scheduler.step()
        return loss_sum.mean().item(), v_loss_sum.mean().item()

    




