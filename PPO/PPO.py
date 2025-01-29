import numpy as np
import torch

import configparser

from train_evaluate import PPOTrainer
from utils.plot import plot_curves

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Environment
ENV_ID = config['ENVIRONMENT']['env_id']
RENDER = config.getboolean('ENVIRONMENT', 'render')

# Hyper-parameters
hyperparameters = {
    "hidden_size": int(config['MODEL']['hidden_size']),
    "learning_rate": float(config['TRAINING']['learning_rate']),
    "gamma": float(config['TRAINING']['gamma']),
    "lambda": float(config['TRAINING']['lambda']),
    "ppo_clip_eps": float(config['TRAINING']['ppo_clip_eps']),
    "value_coef": float(config['TRAINING']['value_coef']),
    "entropy_coef": float(config['TRAINING']['entropy_coef']),
}

# Seed
seed = int(config['ENVIRONMENT']['seed'])
torch.manual_seed(seed)
np.random.seed(seed)

# Training configuration
max_train_steps = int(config['TRAINING']['max_train_steps'])
max_episode_steps = int(config['TRAINING']['max_episode_steps'])
batch_size = int(config['TRAINING']['batch_size'])
ppo_epochs = int(config['TRAINING']['ppo_epochs'])
evaluation_episodes = int(config['TRAINING']['evaluation_episodes'])

def main():
    PPO = PPOTrainer(ENV_ID, RENDER, hyperparameters)

    train_reward_history, eval_reward_history = [], []
    num_iterations = 5
    train_steps = max_train_steps // num_iterations
    for _ in range(num_iterations):
        # Train the agent
         PPO.train(train_steps, max_episode_steps, batch_size, ppo_epochs)
        # Evaluate the agent
         PPO.evaluate(max_episode_steps, episodes=evaluation_episodes)

    # Plot the results
    plot_curves(PPO.train_reward_log, PPO.eval_reward_log, PPO.train_step_log, train_steps)


if __name__ == "__main__":
    main()
