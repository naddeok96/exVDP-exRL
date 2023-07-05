import gym
import torch
import os
import wandb
import cv2
import numpy as np
import random
import imageio

from dqn import DQNAgent
from vdp_dqn import VDPDQNAgent
from noise_box import NoiseGenerator


def test(env, agent, noise_box, episodes=100, max_steps=300, model_weights=None):
    rewards = []  # List to collect total rewards
    reg_uncerts = []
    adv_uncerts = []
    # render_points = random.sample(range(episodes), 10)  # Randomly sample 10 render points
    # frames = []
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            
            adv_state = noise_box.add_noise(state)
            action, q_values, q_sigmas = agent.act(adv_state, return_sigma=True)
            reg_action, _, reg_q_sigmas = agent.act(state, return_sigma=True)
            
            reg_uncert = reg_q_sigmas[:,action, action]
            adv_uncert = q_sigmas[:,action, action]
            
            reg_uncerts.append(reg_uncert.item())
            adv_uncerts.append(adv_uncert.item())
            
            # print("State/Adv_state: ", np.stack((state, adv_state)), " Action/Adv_action:", agent.act(state), "/", action)
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
                
            state = next_state
            
            # if e in render_points:  # Render the environment at selected render points
            #     render_frame = env.render()
            #     frames.append(cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB))

            if done or step == max_steps - 1:
                # Log total reward to the list
                rewards.append(total_reward)

                break
    
    # output_path = 'results/normal.mp4'
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with imageio.get_writer(output_path, fps=30) as writer:
    #     for frame in frames:
    #         writer.append_data(frame)

    # Calculate and log the mean total reward
    mean_reward = np.mean(rewards)
    reg_uncert = np.mean(reg_uncerts)
    adv_uncert = np.mean(adv_uncerts)
    print("Reg Avg Uncert: ", reg_uncert, " Adv Avg Uncert: ", adv_uncert)
    print(f"Mean Total Reward: {mean_reward}, Noise Type: {noise_box.noise_type}, SNR: {noise_box.snr}, Model Weights: {model_weights}")
    print(f"")
    
    # Log mean reward
    wandb.log({
        "mean_reward": mean_reward
    })       
    
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
    episodes = 10
    max_steps = 300

    env = gym.make("Acrobot-v1", render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    noise_box = NoiseGenerator()

    models = ["saved_models/VDP DQN Acrobot - Distill_stilted-rain-11_vdp_dqn.pt"] # "saved_models/DQN Acrobot_olive-pyramid-9_dqn.pt", 
    noises = ["gaussian"] # , "uniform", "salt_and_pepper", "quantization", "sparse_coding"] #, "perlin"]
    snr_values = [1000]
    
    for model_weights in models:
        
        if "VDP" in model_weights:
            agent = VDPDQNAgent(state_size, action_size, fc1_size=256, fc2_size=128, device=device,explore=False)
        else:
            agent = DQNAgent(state_size, action_size, fc1_size=128, fc2_size=128, device=device, explore=False)

        agent.load_model(model_weights)
        
        for noise_type in noises:
            noise_box.noise_type = noise_type
            for snr in snr_values:
                noise_box.snr = snr

                wandb.init(project="Acrobot Robustness Test Extreme", entity="naddeok") # , mode="disabled")
                wandb.config.model_weights = model_weights
                wandb.config.noise_type = noise_type
                wandb.config.snr = snr

                test(env, agent, noise_box, episodes, max_steps, model_weights)
                wandb.finish()

