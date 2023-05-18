
import sys
sys.path.append("..")
import kyus_gym.gym as gym

import torch
import os
import wandb
import numpy as np

from vdp_dqn import VDPDQNAgent  
    
if __name__ == "__main__":

    # Initialize GPU usage
    gpu_number = "0"
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

    agent = VDPDQNAgent(state_size, action_size, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)
    agent.load_model("saved_models/Robust DQN CartPole_elated-energy-4_vdp_dqn_at_100_successes.pt")
    
    agent.get_covariance_matrices()
    
    
