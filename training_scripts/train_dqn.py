import sys
sys.path.append(sys.path[0] + '/..')  # Add parent directory to the path
from model_classes.dqn import DQNAgent

import random
import gym
import yaml
import torch
import os
import wandb
import numpy as np
from scipy.stats import linregress

def check_trend(history, threshold):
    # Create an array of time indices
    time_indices = np.arange(len(history))

    # Perform linear regression to calculate the slope
    slope, _, _, _, _ = linregress(time_indices, history)

    # Check if the absolute value of the slope is less than the threshold
    return abs(slope) < threshold

def print_params(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            print(name, param[0][0])
        else:
             print(name, param[0])

def train(env, agent, batch_size=32, episodes=500, max_steps=200):
    reward_plateau_threshold = wandb.config.reward_plateau_threshold
    loss_plateau_threshold = wandb.config.loss_plateau_threshold
    plateau_window = wandb.config.plateau_window
    reward_plateaued = False
    loss_plateaued = False

    reward_history = []
    loss_history = []
    best_reward_model_path = None
    best_loss_model_path = None

    best_reward = float('-inf')  # Initialize the best reward as negative infinity
    best_loss = float('inf')  # Initialize the best loss as positive infinity

    worst_reward = float('inf')
    total_reward = 0
    for e in range(episodes):
        

        # Check for reward plateau
        if len(reward_history) > plateau_window:
            reward_history.pop(0)
            if check_trend(reward_history[-plateau_window:], reward_plateau_threshold) and total_reward != worst_reward:
                print(f"Reward has not changed over the last {plateau_window} episodes.")
                reward_plateaued = True
            else: 
                reward_plateaued = False

        # Check for loss plateau
        if len(loss_history) > plateau_window:
            loss_history.pop(0)
            if check_trend(loss_history[-plateau_window:], loss_plateau_threshold): # Note the negative sign
                print(f"Loss has not changed over the last {plateau_window} episodes.")
                loss_plateaued = True
            else: 
                loss_plateaued = False

        if loss_plateaued and reward_plateaued and e > 200:
            break

        state, _ = env.reset()
        total_reward = 0    

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                reward = -10

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            loss, prev_epsilon = agent.replay(batch_size)

            # Check if the loss for this step is better than the best so far
            if loss is not None:
                if loss < best_loss:
                    best_loss = loss
                    print(f"New best loss: {best_loss}. Saving model...")
                    if best_loss_model_path and os.path.exists(best_loss_model_path):
                        os.remove(best_loss_model_path)  # Delete the previous best model file
                    best_loss_model_path = f"saved_models/dqn/{wandb.config.env_name}_{wandb.run.project}_{wandb.run.name}_dqn_best_loss_at_{e}_eps.pt"
                    agent.save(best_loss_model_path)

            # Log
            wandb.log({
                "step": step,
                "total_reward": total_reward,
                "loss": loss,
                "prev_epsilon": prev_epsilon,
                "action": action
            })

            if done or step == max_steps - 1:
                reward_history.append(total_reward)
                if loss is not None:
                    loss_history.append(loss.item())

                wandb.log({
                    "total_reward": total_reward,
                    "i_episode": e
                })
                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break


        # Check if the total reward for this episode is better than the best so far
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"New best score: {best_reward}. Saving model...")
            if best_reward_model_path and os.path.exists(best_reward_model_path):
                os.remove(best_reward_model_path)  # Delete the previous best model file
            best_reward_model_path = f"saved_models/dqn/{wandb.config.env_name}_{wandb.run.project}_{wandb.run.name}_dqn_best_reward_at_{e}_eps.pt"
            agent.save(best_reward_model_path)

        if total_reward <= worst_reward:
            worst_reward = total_reward

    # Save the final model
    agent.save(f"saved_models/dqn/{wandb.config.env_name}_{wandb.run.project}_{wandb.run.name}_dqn_final_at_{e}_eps.pt")


if __name__ == "__main__":

    # List of environments
    environments = [
        # "CartPole-v1", 
        # "MountainCar-v0", 
        # "Acrobot-v1",
        # "LunarLander-v2",  
        "Breakout-v4", 
        # "Pong-v4", 
        # "SpaceInvaders-v4"
    ] 
    random.shuffle(environments)

    # Initialize GPU usage
    gpu_number = "6"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Iterate through environments
    for env_name in environments:
        # Load hyperparameters from YAML file
        with open('yamls/dqn/' + env_name + '.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Initialize WandB
        wandb.init(project="DQN", entity="naddeok", config=config) #, mode="disabled")
        wandb.config.env_name = env_name
        
        # Access hyperparameters from WandB config
        fc1_size = wandb.config.fc1_size
        fc2_size = wandb.config.fc2_size
        gamma = wandb.config.gamma
        epsilon = wandb.config.epsilon
        epsilon_min = wandb.config.epsilon_min
        epsilon_decay = wandb.config.epsilon_decay
        learning_rate = wandb.config.learning_rate
        memory_size = wandb.config.memory_size
        batch_size = wandb.config.batch_size
        episodes = wandb.config.episodes
        max_steps = wandb.config.max_steps
        use_conv = wandb.config.use_conv

        # Extract conv-specific settings if use_conv=True
        if use_conv:
            conv1_out_channels = wandb.config.conv1_out_channels
            conv1_kernel_size = wandb.config.conv1_kernel_size
            conv1_stride = wandb.config.conv1_stride
            conv2_out_channels = wandb.config.conv2_out_channels
            conv2_kernel_size = wandb.config.conv2_kernel_size
            conv2_stride = wandb.config.conv2_stride
            conv3_out_channels = wandb.config.conv3_out_channels
            conv3_kernel_size = wandb.config.conv3_kernel_size
            conv3_stride = wandb.config.conv3_stride


        env = gym.make(env_name)
        
        cont = isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) > 1
        state_size = env.observation_space.shape if cont else env.observation_space.shape[0]

        action_size = env.action_space.n
        
        if use_conv:
            agent = DQNAgent(state_size, 
                            action_size, 
                            fc1_size=fc1_size, 
                            fc2_size=fc2_size,
                            conv1_out_channels=conv1_out_channels,
                            conv1_kernel_size=conv1_kernel_size,
                            conv1_stride=conv1_stride,
                            conv2_out_channels=conv2_out_channels,
                            conv2_kernel_size=conv2_kernel_size,
                            conv2_stride=conv2_stride,
                            conv3_out_channels=conv3_out_channels,
                            conv3_kernel_size=conv3_kernel_size,
                            conv3_stride=conv3_stride,
                            device=device, 
                            gamma=gamma,
                            epsilon=epsilon, 
                            epsilon_min=epsilon_min, 
                            epsilon_decay=epsilon_decay, 
                            learning_rate=learning_rate, 
                            memory_size=memory_size,
                            use_conv=use_conv)
        else:
            agent = DQNAgent(state_size, 
                            action_size, 
                            fc1_size=fc1_size, 
                            fc2_size=fc2_size,
                            device=device, 
                            gamma=gamma,
                            epsilon=epsilon, 
                            epsilon_min=epsilon_min, 
                            epsilon_decay=epsilon_decay, 
                            learning_rate=learning_rate, 
                            memory_size=memory_size,
                            use_conv=use_conv)

        # Standard train
        train(env, agent, batch_size, episodes, max_steps)

        wandb.finish()

