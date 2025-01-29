import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda import device

from model import PPONetwork


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size, lr, gamma, lam, ppo_clip_eps, value_coef, entropy_coef,
                 device):
        self.model = PPONetwork(state_dim, action_dim, hidden_size).to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.ppo_clip_eps = ppo_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state)
        return action, log_prob, value

    def compute_advantages_returns(self, rollout_buffer, last_value):
        advantages, returns = rollout_buffer.compute_advantages_returns(
            last_value, self.gamma, self.lam
        )
        return advantages, returns

    def update(self, rollout_buffer, advantages, returns, ppo_epochs):
        b_states = torch.FloatTensor(np.array(rollout_buffer.states)).to(self.device)
        b_actions = torch.LongTensor(np.array(rollout_buffer.actions)).to(self.device)
        b_log_probs = torch.FloatTensor(np.array(rollout_buffer.log_probs)).to(self.device)

        # Normalize advantages
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(self.device)

        for _ in range(ppo_epochs):
            new_log_probs, entropy, values = self.model.evaluate_actions(b_states, b_actions)
            ratio = (new_log_probs - b_log_probs).exp().to(self.device)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values, returns.to(self.device))

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
