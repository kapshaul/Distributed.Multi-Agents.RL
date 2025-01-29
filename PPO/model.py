import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# Feature Scaler
class FeatureScaler(nn.Module):
    def __init__(self, hidden_size, adjacency_matrix=0):
        super(FeatureScaler, self).__init__()

        # GNN matrix
        #F = torch.FloatTensor(self.gnn_normalize(adjacency_matrix))

        # Register constant vector or matrix into the buffer
        self.register_buffer("F", F)

    def gnn_normalize(self, adjacency_matrix):
        # Add self-loops (optional, common in GNNs)
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        # Compute the degree matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # Compute D^(-1/2)
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        # Compute the normalized adjacency matrix
        normalized_adj = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
        return normalized_adj

    def forward(self, x):
        return x * self.F


# PPO Network
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PPONetwork, self).__init__()
        # Common layer
        self.common = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            #nn.Softmax(dim=-1),
            nn.Sigmoid(),
        )

        # Policy layer
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, action_dim)
        )
        # Value layer
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        nn.init.zeros_(self.policy[0].weight)
        nn.init.zeros_(self.policy[0].bias)
        nn.init.zeros_(self.value[0].weight)
        nn.init.zeros_(self.value[0].bias)

    def forward(self, x):
        """
        Returns policy logits and state value
        """
        x = x.view(-1, x.size(-1))

        features = self.common(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value

    def get_action(self, state):
        """
        Given a single state,
        Sample an action
        Return action, log_prob, and value
        """

        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_actions(self, states, actions):
        """
        Given states and actions,
        Return log_probs, entropy, and value
        """

        logits, values = self.forward(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)