import torch.nn as nn
from torch.distributions import Categorical
from utils.model import CustomLinear, CustomConv2D, FeatureScaler


# PPO Network
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()

        # Common layer
        self.common = nn.Sequential(
            #nn.Linear(state_dim[0], hidden_size),
            CustomLinear(state_dim[0], hidden_size, 0.9),
            FeatureScaler(hidden_size),
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


# PPO Network with CNN
class PPONetwork_CNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        """
        Output size formula: 
        O = 1 + (I - K + 2P) // S
        where,
        - I = Input size (height/width)
        - K = Kernel size
        - P = Padding
        - S = Stride
        """

        self.conv1 = nn.Sequential(
            # [N, 32, 20, 20]
            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=True),
            #nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1),
            #nn.AvgPool2d(2, 2),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            # [N, 64, 9, 9]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            # [N, 64, 9, 9]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
        )

        # Common layer
        self.common = nn.Sequential(
            #nn.Linear(9*9*64, hidden_size),
            CustomLinear(7*7*64, hidden_size, 0.9),
            FeatureScaler(hidden_size),
            nn.ReLU(),
        )

        # Policy layer
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
        )
        # Value layer
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )

        nn.init.zeros_(self.policy[0].weight)
        nn.init.zeros_(self.policy[0].bias)
        nn.init.zeros_(self.value[0].weight)
        nn.init.zeros_(self.value[0].bias)

    def forward(self, x):
        """
        Returns policy logits and state value
        """

        x = x.view(-1, 4, 84, 84)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)

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