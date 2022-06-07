import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from model.graph_cnn import GraphCNN

from params import device, config


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_j,
        n_m,
        # feature extraction net attributes
        num_of_layers,
        use_learn_epsilon,
        input_dim,
        hidden_dim,
        # feature extraction net MLP attributes
        num_of_mlp_layers_for_feature_extract,
        # actor net attributes
        num_of_mlp_layers_actor,
        hidden_dim_actor,
        # critic net attributes
        num_of_mlp_layers_critic,
        hidden_dim_critic,
    ):
        super(ActorCritic, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        
        self.feature_extract = GraphCNN(
            num_of_layers=num_of_layers,
            num_of_mlp_layers=num_of_mlp_layers_for_feature_extract,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_learn_epsilon=use_learn_epsilon,
        ).to(device)
        self.actor = MLPActor(
            num_of_layers=num_of_mlp_layers_actor,
            # 64 + 64 + 4
            # pooled node dimension + node dimension from GNN + length of machine features
            input_dim=(config.hidden_dim * 2) + 4,
            hidden_dim=hidden_dim_actor,
            output_dim=1,
        ).to(device)
        self.critic = MLPCritic(
            num_of_layers=num_of_mlp_layers_critic,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim_critic,
            output_dim=1,
        ).to(device)

    def forward(
        self,
        x,
        adj_matrix,
        graph_pool,
        candidate,
        mask,
        machine_feat,
    ):
        h_pooled, h_nodes = self.feature_extract(
            x=x.float(),
            adj_matrix=adj_matrix,
            graph_pool=graph_pool,
        )

        candidate_ops = torch.unique(candidate[:, :, 0], dim=1)
        dummy = candidate_ops.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))

        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy.type(torch.int64))
        candidate_feature = torch.repeat_interleave(candidate_feature, config.n_m, dim=1)

        candidate_and_machine_feature = torch.cat((candidate_feature, machine_feat), dim=2)

        h_pooled_repeated = h_pooled.unsqueeze(1)
        h_pooled_repeated = h_pooled_repeated.expand((
            candidate_and_machine_feature.shape[0],
            candidate_and_machine_feature.shape[1],
            h_pooled_repeated.shape[2]
        ))

        # concatenate feature
        concat_feat = torch.cat((candidate_and_machine_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concat_feat.float())

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled.float())
        
        return pi, v

