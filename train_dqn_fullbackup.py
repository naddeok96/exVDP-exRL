import torch
import sys
sys.path.append("..")
import kyus_gym.gym as gym
import os
import wandb
from copy import deepcopy

from dqn import DQNAgent
from vdp_dqn import VDPDQNAgent

def train(env, agent, batch_size=32, episodes=500, max_steps=200):
    num_success = 0
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):

            # Rollout one step for every action from the initial state
            next_states = torch.zeros(env.action_space.n, env.observation_space.shape[0])
            rewards     = torch.zeros(env.action_space.n)
            dones       = torch.zeros(env.action_space.n)
            for action_i in range(env.action_space.n):
                env.reset(state = state)
                next_state_i, reward_i, done_i, _, _ = env.step(action_i)

                next_states[action_i,:] = torch.tensor(next_state_i, dtype=torch.float32)
                rewards[action_i] = -100 if done_i else reward_i
                dones[action_i] = done_i 

            # Take actual step
            action = agent.act(state)
            env.reset(state = state)
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward

            agent.remember(state, action, rewards, next_states, dones)
            state = next_state

            if agent.kl_factor == 0:
                loss, prev_epsilon = agent.replay(batch_size)
            else:
                total_loss, loss, current_kl_losses, prev_epsilon  = agent.replay(batch_size)

            # Log
            if agent.kl_factor == 0:
                wandb.log({
                    "epsilon":agent.epsilon,
                    "i_episode": e,
                    "step": step,
                    "total_reward": total_reward,
                    "loss": loss,
                    "prev_epsilon": prev_epsilon,
                })
            else:
                wandb.log({
                    "epsilon":agent.epsilon,
                    "i_episode": e,
                    "step": step,
                    "total_reward": total_reward,
                    "loss": loss,
                    "total_loss" : total_loss, 
                    "kl1" : current_kl_losses[0],
                    "kl2" : current_kl_losses[1],
                    "kl3" : current_kl_losses[2],
                    "prev_epsilon": prev_epsilon,
                })

            if done or step == max_steps - 1:
                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if total_reward >= max_steps:
            num_success += 1
            if num_success >= 100:
                print(f"CartPole solved in {e+1} episodes!")
                break

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
    wandb.init(project="DQN CartPole", entity="naddeok")#, mode="disabled")
    wandb.config.fc1_size = fc1_size = 128
    wandb.config.fc2_size = fc2_size = 128  
    wandb.config.kl_factor = kl_factor = 0.001

    wandb.config.gamma = gamma                  = 0.99
    wandb.config.epsilon = epsilon              = 1.0
    wandb.config.epsilon_min = epsilon_min      = 0.01
    wandb.config.epsilon_decay = epsilon_decay  = 0.99995 # 0.995
    wandb.config.learning_rate = learning_rate  = 0.0001
    wandb.config.memory_size = memory_size      = 10000

    wandb.config.batch_size = batch_size    = 128
    wandb.config.episodes = episodes        = 10000
    wandb.config.max_steps = max_steps      = 300

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if kl_factor == 0:
        agent = DQNAgent(state_size, action_size, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)
    else:
        agent = VDPDQNAgent(state_size, action_size, fc1_size=fc1_size, kl_factor = kl_factor, fc2_size=fc2_size, device=device, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)

    train(env, agent, batch_size, episodes, max_steps)

