import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import wandb
import envpool
import cv2
import os
import yaml
from scipy.stats import linregress
from collections import namedtuple
import random
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def check_trend(history, threshold):
    # Create an array of time indices
    time_indices = np.arange(len(history))

    # Perform linear regression to calculate the slope
    slope, _, _, _, _ = linregress(time_indices, history)

    # Check if the absolute value of the slope is less than the threshold
    return abs(slope) < threshold

class DQN(nn.Module):
    def __init__(self, input_shape, action_size,
                 conv1_out_channels=32, conv1_kernel_size=8, conv1_stride=4,
                 conv2_out_channels=64, conv2_kernel_size=4, conv2_stride=2,
                 conv3_out_channels=64, conv3_kernel_size=3, conv3_stride=1,
                 fc1_size=512):
        super(DQN, self).__init__()

        # Assuming input shape is in the format: (height, width, channels)
        self.input_shape = input_shape
        self.action_size = action_size

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], conv1_out_channels, kernel_size=conv1_kernel_size, stride=conv1_stride),  # input_shape[2] is the channel dimension
            nn.ReLU(),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=conv2_kernel_size, stride=conv2_stride),
            nn.ReLU(),
            nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=conv3_kernel_size, stride=conv3_stride),
            nn.ReLU()
        )
        
        self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
        # return self.features(torch.zeros(1, *self.input_shape[::-1])).view(1, -1).size(1)  # Reverse the input shape to (channels, height, width) for PyTorch

    
def select_action(state, policy_net, n_actions, epsilon):
    num_envs = state.shape[0]
    sample = np.random.rand()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(-1).cpu().numpy()
    else:
        return np.random.randint(0, n_actions, num_envs)

def optimize_model(memory, policy_net, target_net, optimizer, cfg):
    if len(memory) < cfg.BATCH_SIZE:
        return None
    
    # Sample from memory
    transitions = memory.sample(cfg.BATCH_SIZE)

    # Unpack transitions
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.from_numpy(np.array(states)).float().to(device)
    action_batch = torch.from_numpy(np.array(actions)).long().to(device)  # Assuming actions are discrete/integers
    reward_batch = torch.from_numpy(np.array(rewards)).float().to(device)
    next_state_batch = torch.from_numpy(np.array(next_states)).float().to(device)
    done_batch = torch.from_numpy(np.array(dones)).float().to(device)  # Assuming done is a float for compatibility with neural network operations

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (1 - done_batch) * (cfg.GAMMA * next_state_values)

    loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        if torch.is_tensor(action):
            action = action.squeeze().cpu().numpy()

        # Use list comprehension to create the transitions
        transitions = [(obs[i], action[i], reward[i], next_obs[i], done[i]) for i in range(obs.shape[0])]

        # Remember
        self.memory.extend(transitions)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

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

    epsilon = cfg.EPS_START
    for i in tqdm(range(cfg.NUM_EPS // cfg.NUM_ENVS)):
        i_episode = i * cfg.NUM_ENVS

        env.async_reset()              
        obs, _, _, _, info = env.recv()
        env_id = info["env_id"]

        total_reward = cfg.NUM_ENVS * [0]

        if epsilon > cfg.EPS_END:
            epsilon *= cfg.EPS_DECAY

        wandb.log({"Episode": i_episode, "epsilon": epsilon})

        # Check for reward plateau
        if len(reward_history) > cfg.PLATEAU_WINDOW:
            reward_history.pop(0)
            if check_trend(reward_history[-cfg.PLATEAU_WINDOW:], cfg.REWARD_PLATEAU_THRESHOLD) and all(value == worst_reward for value in total_reward):
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

        # if loss_plateaued and reward_plateaued and i_episode > cfg.MIN_NUM_EPISODES:
        #     break

        # Track envs that have reset during the loop
        reset_envs = np.zeros(cfg.NUM_ENVS, dtype=bool)
        for t in range(cfg.MAX_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            # action = select_action(obs_tensor, policy_net, env.action_space.n, 0)
            action = select_action(obs_tensor, policy_net, env.action_space.n, epsilon)
            env.send(action, env_id)
            next_obs, reward, done, _, info = env.recv()
            env_id = info["env_id"]

            total_reward += reward

            memory.push(obs, action, reward, next_obs, done)


            loss = optimize_model(memory, policy_net, target_net, optimizer, cfg)
            steps_done += 1

            wandb.log({"Optim Steps": steps_done, "Loss": loss, "Memory Size": len(memory)})

            if steps_done % cfg.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            obs = next_obs

            reward_history.extend(list(total_reward))
            if loss is not None:
                loss_history.append(loss)

            # Clear all rewards for envs that have ended
            if done.any():
                episode_rewards.extend(total_reward[done])
                total_reward = total_reward * (1 - done)
                reset_envs |= done # Bitwise OR operation

            if reset_envs.all():
                break

        episode_rewards.extend(total_reward[~reset_envs])

        wandb.log({"Episode Batch": i, 
                   "Mean Episode Reward": np.mean(episode_rewards),
                   "STD Episode Reward": np.std(episode_rewards)})
        
        for ep_num, ep_reward in enumerate(episode_rewards):
            wandb.log({"Episode Number" : ep_num + i_episode, "Episode Reward": ep_reward})

        # Check if the loss for this step is better than the best so far
        if loss is not None:
            if loss < best_loss:
                best_loss = loss
                print(f"New best loss: {best_loss}. Saving model...")
                if best_loss_model_path and os.path.exists(best_loss_model_path):
                    os.remove(best_loss_model_path)  # Delete the previous best model file
                best_loss_model_path = f"saved_models/dqn/{wandb.config.ENV_NAME}_{wandb.run.project}_{wandb.run.name}_dqn_best_loss_at_{i_episode}_eps.pt"
                torch.save(policy_net.state_dict(), best_loss_model_path)

        # Check if the total reward for this episode is better than the best so far
        if (total_reward > best_reward).any():
            best_reward = np.max(total_reward)
            print(f"New best score: {best_reward}. Saving model...")
            if best_reward_model_path and os.path.exists(best_reward_model_path):
                os.remove(best_reward_model_path)  # Delete the previous best model file
            best_reward_model_path = f"saved_models/dqn/{wandb.config.ENV_NAME}_{wandb.run.project}_{wandb.run.name}_dqn_best_reward_at_{i_episode}_eps.pt"
            torch.save(policy_net.state_dict(), best_reward_model_path)

        if (total_reward < worst_reward).any():
            worst_reward = min(total_reward)

if __name__ == "__main__":

    # Initialize GPU usage
    device = torch.device("cuda")

    # Load hyperparameters from YAML file
    with open('yamls/dqn/Pong-v5-async.yaml', 'r') as file:
        cfg_dict = yaml.safe_load(file)
        config = namedtuple('Config', cfg_dict.keys())
        cfg = config(**cfg_dict)
    
    # Update wandb.config with your cfg
    wandb.init()
    cfg = wandb.config

    env = envpool.make(task_id = cfg.ENV_NAME,
                       env_type = "gym",
                       num_envs = cfg.NUM_ENVS,
                       stack_num = cfg.STACK_NUM
                       )
    obs_dim = env.observation_space.shape
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions,
                    conv1_out_channels=cfg.conv1_out_channels,
                    conv1_kernel_size=cfg.conv1_kernel_size,
                    conv1_stride=cfg.conv1_stride,
                    conv2_out_channels=cfg.conv2_out_channels,
                    conv2_kernel_size=cfg.conv2_kernel_size,
                    conv2_stride=cfg.conv2_stride,
                    conv3_out_channels=cfg.conv3_out_channels,
                    conv3_kernel_size=cfg.conv3_kernel_size,
                    conv3_stride=cfg.conv3_stride,
                    fc1_size=cfg.fc1_size).to(device)

    target_net = DQN(obs_dim, n_actions,
                    conv1_out_channels=cfg.conv1_out_channels,
                    conv1_kernel_size=cfg.conv1_kernel_size,
                    conv1_stride=cfg.conv1_stride,
                    conv2_out_channels=cfg.conv2_out_channels,
                    conv2_kernel_size=cfg.conv2_kernel_size,
                    conv2_stride=cfg.conv2_stride,
                    conv3_out_channels=cfg.conv3_out_channels,
                    conv3_kernel_size=cfg.conv3_kernel_size,
                    conv3_stride=cfg.conv3_stride,
                    fc1_size=cfg.fc1_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.LR)
    memory = ReplayMemory(cfg.MEMORY_SIZE)

    train(env, policy_net, target_net, optimizer, memory, cfg)
