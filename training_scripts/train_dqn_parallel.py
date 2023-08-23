import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import wandb
import gym
import gym.vector
import os
import yaml
from scipy.stats import linregress
from collections import namedtuple

def check_trend(history, threshold):
    # Create an array of time indices
    time_indices = np.arange(len(history))

    # Perform linear regression to calculate the slope
    slope, _, _, _, _ = linregress(time_indices, history)

    # Check if the absolute value of the slope is less than the threshold
    return abs(slope) < threshold

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def select_action(state, policy_net, n_actions, epsilon):
    sample = np.random.rand()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).item()
    else:
        return np.random.randint(n_actions)

def optimize_model(memory, policy_net, target_net, optimizer, cfg):
    if len(memory) < cfg.BATCH_SIZE:
        return None
    transitions = np.array(memory.sample(cfg.BATCH_SIZE))
    batch = np.array(transitions, dtype=object).T
    state_batch = torch.cat(batch[0]).to(device)
    action_batch = torch.cat(batch[1]).to(device)
    reward_batch = torch.cat(batch[2]).to(device)
    next_state_batch = torch.cat(batch[3]).to(device)
    done_batch = torch.cat(batch[4]).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (1 - done_batch) * (cfg.GAMMA * next_state_values)

    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transitions):
        self.memory.extend(transitions)

    def sample(self, batch_size):
        return np.array(np.random.choice(self.memory, batch_size, replace=False))

    def __len__(self):
        return len(self.memory)

def train(env, policy_net, target_net, optimizer, memory, cfg):
    reward_plateaued = False
    loss_plateaued = False

    reward_history = []
    loss_history = []
    best_reward_model_path = None
    best_loss_model_path = None

    best_reward = float('-inf')  
    worst_reward = float('inf')
    best_loss = float('inf')

    steps_done = 0
    episode_rewards = []
    for i_episode in range(cfg.NUM_EPS):
        obs = env.reset()
        total_reward = cfg.NUM_ENVS * [0]

        epsilon = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * np.exp(-1. * i_episode / cfg.EPS_DECAY)

        wandb.log({"epsilon": epsilon})

        # Check for reward plateau
        if len(reward_history) > cfg.PLATEAU_WINDOW:
            reward_history.pop(0)
            if check_trend(reward_history[-cfg.PLATEAU_WINDOW:], cfg.REWARD_PLATEAU_THRESHOLD) and total_reward != worst_reward:
                print(f"Reward has not changed over the last {cfg.PLATEAU_WINDOW} episodes.")
                reward_plateaued = True
            else: 
                reward_plateaued = False

        # Check for loss plateau
        if len(loss_history) > cfg.PLATEAU_WINDOW:
            loss_history.pop(0)
            if check_trend(loss_history[-cfg.PLATEAU_WINDOW:], cfg.LOSS_PLATEAU_THRESHOLD): # Note the negative sign
                print(f"Loss has not changed over the last {cfg.PLATEAU_WINDOW} episodes.")
                loss_plateaued = True
            else: 
                loss_plateaued = False

        if loss_plateaued and reward_plateaued and i_episode > cfg.MIN_NUM_EPISODES:
            break

        # Track envs that have reset during the loop
        reset_envs = torch.zeros(len(done), dtype=torch.float32).to(device)
        for t in range(cfg.MAX_STEPS):
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            action = select_action(obs, policy_net, env.action_space.n, epsilon)
            next_obs, reward, done, _ = env.step([action])

            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32)

            transitions = list(zip(obs, action, reward, next_obs, done))
            memory.push(transitions)

            loss = optimize_model(memory, policy_net, target_net, optimizer, cfg)
            steps_done += 1

            if steps_done % cfg.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            obs = next_obs

            reward_history.extend(total_reward)
            if loss is not None:
                loss_history.extend(loss)

            # Clear all rewards for envs that have ended
            episode_rewards.extend(total_reward[done])
            total_reward = total_reward * (1 - done)
            reset_envs |= done # Bitwise OR operation


        episode_rewards.extend(total_reward[~reset_envs])

        # Log rewards and loss to Weights & Biases
        wandb.log({'Episode Reward': total_reward, 'Loss': loss})

        print(f"Episode {i_episode} Reward: {total_reward}")

        # Check if the loss for this step is better than the best so far
        if loss is not None:
            if (loss < best_loss).any():
                best_loss = torch.min(loss).item()
                print(f"New best loss: {best_loss}. Saving model...")
                if best_loss_model_path and os.path.exists(best_loss_model_path):
                    os.remove(best_loss_model_path)  # Delete the previous best model file
                best_loss_model_path = f"saved_models/dqn/{wandb.config.env_name}_{wandb.run.project}_{wandb.run.name}_dqn_best_loss_at_{i_episode}_eps.pt"
                torch.save(best_loss_model_path)

        # Check if the total reward for this episode is better than the best so far
        if (total_reward > best_reward.any()):
            best_reward = torch.max(total_reward).item()
            print(f"New best score: {best_reward}. Saving model...")
            if best_reward_model_path and os.path.exists(best_reward_model_path):
                os.remove(best_reward_model_path)  # Delete the previous best model file
            best_reward_model_path = f"saved_models/dqn/{wandb.config.env_name}_{wandb.run.project}_{wandb.run.name}_dqn_best_reward_at_{i_episode}_eps.pt"
            torch.save(best_reward_model_path)

        if (total_reward < worst_reward).any():
            worst_reward = torch.min(total_reward).item()

    env.close()

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "6"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load hyperparameters from YAML file
    env_name = "Pong-v4"
    with open('yamls/dqn/' + env_name + '-async.yaml', 'r') as file:
        cfg_dict = yaml.safe_load(file)
        config = namedtuple('Config', cfg_dict.keys())
        cfg = config(**cfg_dict)

    wandb.init(project="DQN", entity="naddeok", config=cfg_dict)

    env = gym.vector.AsyncVectorEnv([lambda: gym.make('Pong-v4') for _ in range(cfg.NUM_ENVS)])
    obs_dim = env.observation_space[0].shape
    n_actions = env.action_space[0].n

    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.LR)
    memory = ReplayMemory(cfg.MEMORY_SIZE)

    train(env, policy_net, target_net, optimizer, memory, cfg)
