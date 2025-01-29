import gym
import torch
from agent import PPOAgent
from memory import RolloutBuffer


class PPOTrainer:
    def __init__(self, env_id, render, hyperparameters):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyper-parameters
        self.env_id = env_id
        self.hidden_size = hyperparameters["hidden_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.gamma = hyperparameters["gamma"]
        self.lam = hyperparameters["lambda"]
        self.ppo_clip_eps = hyperparameters["ppo_clip_eps"]
        self.value_coef = hyperparameters["value_coef"]
        self.entropy_coef = hyperparameters["entropy_coef"]

        # Create environment
        if render:
            self.env = gym.make(env_id, render_mode="human")
        else:
            self.env = gym.make(env_id)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Initialize PPO agent
        self.agent = PPOAgent(
            self.state_dim, self.action_dim, self.hidden_size,
            self.learning_rate, self.gamma, self.lam,
            self.ppo_clip_eps, self.value_coef, self.entropy_coef,
            self.device
        )

        self.rollout_buffer = RolloutBuffer()

        self.episode, self.total_steps = 0, 0
        self.train_reward_log, self.eval_reward_log = [], []
        self.train_step_log = []

    def train(self, max_train_steps, max_episode_steps, batch_size, ppo_epochs):
        """
        Train
        """

        state, info = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        total_reward = 0
        total_steps, steps = 0, 0
        while total_steps < max_train_steps:
            # Collect data for one batch
            self.rollout_buffer.clear()
            for _ in range(batch_size):
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, info1, info2 = self.env.step(action)

                total_reward += reward
                self.rollout_buffer.add(
                    state.to(torch.device('cpu')).numpy(),
                    action,
                    log_prob.item(),
                    reward,
                    done,
                    value.item()
                )

                state = torch.FloatTensor(next_state).to(self.device)
                steps += 1
                total_steps += 1
                self.total_steps += 1

                if done or steps >= max_episode_steps:
                    state, info = self.env.reset()
                    state = torch.FloatTensor(state).to(self.device)

                    print(f"Episode: {self.episode + 1}      Rewards: {total_reward}")
                    self.train_reward_log.append(total_reward)
                    self.train_step_log.append(self.total_steps)
                    self.episode += 1
                    total_reward, steps = 0, 0

            # Compute advantages and returns
            with torch.no_grad():
                _, next_value = self.agent.model.forward(state)
                last_value = next_value.item()

            advantages, returns = self.agent.compute_advantages_returns(
                self.rollout_buffer, last_value
            )
            # PPO update
            self.agent.update(self.rollout_buffer, advantages, returns, ppo_epochs)

    def evaluate(self, max_episode_steps, episodes=10):
        """
        Evaluation
        """

        #self.env = gym.make(self.env_id, render_mode="human")

        total_rewards = []
        for episode in range(episodes):
            state, info = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < max_episode_steps:
                action, _, _ = self.agent.select_action(state)
                next_state, reward, done, info1, info2 = self.env.step(action)
                episode_reward += reward
                steps += 1
                state = torch.FloatTensor(next_state).to(self.device)

            total_rewards.append(episode_reward)
            print(f"Evaluation {episode + 1},      Reward = {episode_reward}")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward: {avg_reward}\n")
        self.eval_reward_log.append(avg_reward)
