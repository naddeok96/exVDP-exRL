
import sys
sys.path.append("..")
import kyus_gym.gym as gym

import torch
import os
import wandb
import numpy as np

from dqn import DQNAgent

def add_noise(state, snr):
    
    signal_power = np.mean(np.square(state))
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), state.shape)
    return state + noise

def test(env, agent, episodes=500, max_steps=200, snr=None):
    rewards = []  # List to collect total rewards
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            
            adv_state = add_noise(state, snr)
            action = agent.act(adv_state)
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                reward = -100
                
            state = next_state

            if done or step == max_steps - 1:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}, SNR: {snr}")
                
                # Log total reward to the list
                rewards.append(total_reward)

                break

    # Calculate and log the mean total reward
    mean_reward = np.mean(rewards)
    print(f"Mean Total Reward: {mean_reward}")
    
    # Log mean reward
    wandb.log({
        "mean_reward": mean_reward
    })       
    
if __name__ == "__main__":

    # Initialize GPU usage
    gpu_number = "5"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize WandB
    
    fc1_size = fc1_size = 128
    fc2_size = fc2_size = 128  

    gamma           = 0.99
    epsilon         = 0
    epsilon_min     = 0.01
    epsilon_decay   = 0.995
    learning_rate   = 0.0001
    memory_size     = 10000

    batch_size = batch_size    = 32
    episodes = episodes        = 100
    max_steps = max_steps      = 300

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)
    agent.load_model("saved_models/Robust DQN CartPole_golden-deluge-3_dqn.pt")
    
    # Loop through SNR values from 0 to 1 in 10 increments
    snr_values = [0.01,  0.1, 1.0, 2.5, 5.0, 7.5, 10.0, 100]
    for snr in snr_values:
        
        wandb.init(project="Guassian DQN CartPole", entity="naddeok") # , mode="disabled")
        wandb.config.snr = snr
        test(env, agent, episodes, max_steps, snr)
        wandb.finish()

