import gym
import torch
import os
import wandb
from copy import deepcopy

from vdp_dqn_var_only import VarOnlyVDPDQNAgent

def train(env, agent, batch_size=32, episodes=500, max_steps=200, target_reward=-100, target_successes=100):
    num_success = 0
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):

            # Take actual step
            action, _, q_sigma = agent.act(state, return_sigma=True)
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            total_loss, nll_loss, weighted_w_kl_loss, weighted_b_kl_loss, weighted_predictive_sigmas, current_kl_losses, model_sigmas, predictive_sigmas, mse, error_over_sigma, log_determinant, num_experiences  = agent.replay(batch_size, return_uncertainty_values = True)
            
            # Log
            wandb.log({
                "action": action,
                "epsilon": agent.epsilon,
                "det sigma": torch.prod(q_sigma),
                "mse": mse,
                "num_experiences" : num_experiences,
                "nll_loss": nll_loss,
                "weighted_w_kl_loss": weighted_w_kl_loss,
                "weighted_b_kl_loss": weighted_b_kl_loss,
                "weighted_predictive_sigmas" : weighted_predictive_sigmas,
                "total_loss" : total_loss, 
                "kl_losses" : current_kl_losses,
                "model_sigmas" :  model_sigmas,
                "predictive_sigmas" : predictive_sigmas,
                "error_over_sigma": error_over_sigma,
                "log_determinant": log_determinant
            })

            if done or step == max_steps - 1:
                # Log
                wandb.log({
                    "i_episode": e,
                    "total_reward": total_reward,
                })

                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if total_reward >= target_reward:
            num_success += 1
        else:
            num_success = 0

        if num_success == target_successes:
            print(f"Solved in {e+1} episodes!")
                
            agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn_at_100_successes.pt")

        if e % 1000 == 0:
            agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn_at_" + str(e) + "_episodes.pt")

    agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_vdp_dqn.pt")

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
    wandb.init(project="VDP DQN Acrobot - Var Only", entity="naddeok") # , mode="disabled")
    wandb.config.fc1_size = fc1_size = 256
    wandb.config.fc2_size = fc2_size = 128

    wandb.config.kl_w_factor = kl_w_factor = 0.00001
    wandb.config.kl1_w_factor = kl1_w_factor = 1/18
    wandb.config.kl2_w_factor = kl2_w_factor = 1/118
    wandb.config.kl3_w_factor = kl3_w_factor = 1/3.435

    wandb.config.kl_b_factor = kl_b_factor = 0.001
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
    wandb.config.target_successes = target_successes = 100

    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = VarOnlyVDPDQNAgent(state_size, action_size,
                        kl_w_factor     = kl_w_factor,
                        kl1_w_factor    = kl1_w_factor,
                        kl2_w_factor    = kl2_w_factor,
                        kl3_w_factor    = kl3_w_factor,
                        kl_b_factor     = kl_b_factor,
                        kl1_b_factor    = kl1_b_factor,
                        kl2_b_factor    = kl2_b_factor,
                        kl3_b_factor    = kl3_b_factor,
                        fc1_size        = fc1_size,
                        fc2_size        = fc2_size,
                        device          = device,
                        gamma           = gamma,
                        k               = k,
                        learning_rate   = learning_rate,
                        memory_size     = memory_size)


    train(env, agent, batch_size, episodes, max_steps, target_reward, target_successes)
