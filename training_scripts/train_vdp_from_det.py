
import gym

import torch
import os
import wandb
from copy import deepcopy

from vdp_dqn import VDPDQNAgent
from model_classes.dqn import DQNAgent

def train(env, vdp_env, agent, vdp_agent, batch_size=32, episodes=500, max_steps=200, target_reward=-100, target_successes=100):
    num_success = 0
    for e in range(episodes):
        state, _ = env.reset()
        internal_state = env.state

        total_reward = 0

        for step in range(max_steps):

            # Rollout one step for every action from the initial state
            next_states = torch.zeros(env.action_space.n, env.observation_space.shape[0])
            rewards     = torch.zeros(env.action_space.n)
            dones       = torch.zeros(env.action_space.n)
            for action_i in range(env.action_space.n):
                vdp_env.reset()
                vdp_env.state = internal_state

                next_state_i, reward_i, done_i, _, _ = vdp_env.step(action_i)

                next_states[action_i,:] = torch.tensor(next_state_i, dtype=torch.float32)
                rewards[action_i] = -10 if done_i else reward_i
                dones[action_i] = done_i 

            # Take actual step
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            _, q_sigma = vdp_agent.act(state, return_sigma=True)

            vdp_agent.remember(state, action, rewards, next_states, dones)
            state = next_state
            internal_state = env.state

            total_loss, nll_loss, weighted_w_kl_loss, weighted_b_kl_loss, weighted_predictive_sigmas, current_kl_losses, model_sigmas, predictive_sigmas, error_over_sigma, log_determinant, num_experience  = vdp_agent.replay(batch_size, return_uncertainty_values = True)
            
            # Log
            wandb.log({
                "action": action,
                "epsilon": agent.epsilon,
                "det sigma": torch.linalg.det(q_sigma),
                "total_reward": total_reward,
                "nll_loss": nll_loss,
                "weighted_w_kl_loss": weighted_w_kl_loss,
                "weighted_b_kl_loss": weighted_b_kl_loss,
                "weighted_predictive_sigmas" : weighted_predictive_sigmas,
                "total_loss" : total_loss, 
                "kl_losses" : current_kl_losses,
                "model_sigmas" :  model_sigmas,
                "predictive_sigmas" : predictive_sigmas,
                "error_over_sigma": error_over_sigma,
                "log_determinant": log_determinant,
                "num_experience" : num_experience
            })

            if done or step == max_steps - 1:
                # Log
                wandb.log({
                    "i_episode": e,
                    "total_reward": total_reward,
                })

                vdp_agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if total_reward >= target_reward:
            num_success += 1
        else:
            num_success = 0

        if num_success == target_successes:
            print(f"Solved in {e+1} episodes!")
                
            vdp_agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn_at_100_successes.pt")

        if e % 50 == 0:
            vdp_agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn_at_" + str(e) + "_episodes.pt")
    
    vdp_agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn.pt")

if __name__ == "__main__":

    # Initialize GPU usage
    gpu_number = "1"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize WandB
    wandb.init(project="VDP DQN Acrobot - True Off Policy", entity="naddeok") # mode="disabled")
    wandb.config.fc1_size = fc1_size = 256
    wandb.config.fc2_size = fc2_size = 128

    wandb.config.kl_w_factor = kl_w_factor = 0.01
    wandb.config.kl1_w_factor = kl1_w_factor = 1/18
    wandb.config.kl2_w_factor = kl2_w_factor = 1/118
    wandb.config.kl3_w_factor = kl3_w_factor = 1/3.435

    wandb.config.kl_b_factor = kl_b_factor = 1.0
    wandb.config.kl1_b_factor = kl1_b_factor = 1/290
    wandb.config.kl2_b_factor = kl2_b_factor = 1/55
    wandb.config.kl3_b_factor = kl3_b_factor = 1/9.365

    wandb.config.gamma= gamma                   = 0.99
    wandb.config.k = k                          = 3
    wandb.config.learning_rate = learning_rate  = 0.0001
    wandb.config.memory_size = memory_size      = 10000

    wandb.config.batch_size = batch_size    = 128
    wandb.config.episodes = episodes        = 10000
    wandb.config.max_steps = max_steps      = 300
    wandb.config.target_reward = target_reward = -100
    wandb.config.target_successes = target_successes = 1000

    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, fc1_size=128, fc2_size=128, device=device, gamma=gamma, epsilon=-1)
    agent.load_model("saved_models/DQN Acrobot_olive-pyramid-9_dqn.pt")

    vdp_agent = VDPDQNAgent(state_size, action_size, kl_w_factor = kl_w_factor, kl1_w_factor = kl1_w_factor, kl2_w_factor = kl2_w_factor, kl3_w_factor = kl3_w_factor, kl_b_factor = kl_b_factor, kl1_b_factor = kl1_b_factor, kl2_b_factor = kl2_b_factor, kl3_b_factor = kl3_b_factor, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma, k = k, learning_rate=learning_rate, memory_size=memory_size)
    # vdp_agent.load_model("saved_models/VDP DQN Acrobot - True Off Policy_worldly-serenity-7_vdp_dqn_at_650_episodes.pt")
    vdp_env = gym.make("Acrobot-v1")

    train(env, vdp_env, agent, vdp_agent, batch_size, episodes, max_steps, target_reward, target_successes)
