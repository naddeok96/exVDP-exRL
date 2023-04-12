import gym
import torch
import cv2
import os
import imageio
from tqdm import tqdm
import numpy as np
import wandb

from a2c import A2CAgent
from utils import calculate_novelty_exploration_efficiency, calculate_entropy_exploration_efficiency, calculate_coverage_exploration_efficiency

gpu_number = "2"
if gpu_number:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1', render_mode='rgb_array')
if isinstance(env.observation_space, gym.spaces.Box):
    # Continous space
    state_space_size = 1 # Note state space size is inf but we use 1 for division in calculate_coverage_exploration_efficiency
else:
    # Discrete space
    state_space_size = env.observation_space.n # total number of possible states
agent = A2CAgent(env.observation_space.shape[0], env.action_space.n, device=device)

wandb.init(project="A2C", entity="naddeok")
config = wandb.config
episodes = wandb.config.episodes = 100
max_time_steps = config.max_time_steps = 500

render_freq = 50 # render every 50 episodes
frames = []
visited_states = set() # keep track of visited states

episode_rewards = []

for i_episode in tqdm(range(episodes),desc="Episodes"):
    state, info = env.reset()
    episode_reward = 0
    done = False
    frame_number = 0
    time_step = 0
    while not done or time_step < max_time_steps:
        action, logits = agent.act(state, return_logits=True)
        next_state, reward, done, _, _ = env.step(action)

        # calculate exploration efficiency
        visited_states.add(tuple(state))
        novelty_based_exploration_efficiency = calculate_novelty_exploration_efficiency(next_state, visited_states)
        entropy_based_exploration_efficiency = calculate_entropy_exploration_efficiency(logits)
        coverage_based_exploration_efficiency = calculate_coverage_exploration_efficiency(visited_states, state_space_size)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        time_step += 1
        
        if done or time_step >= max_time_steps:
            episode_rewards.append(episode_reward)
            moving_avg = np.mean(episode_rewards[-int(config.episodes*0.1):])
            

            wandb.log({
                "i_episode": i_episode,
                "novelty_based_exploration_efficiency": novelty_based_exploration_efficiency,
                "entropy_based_exploration_efficiency": entropy_based_exploration_efficiency,
                "coverage_based_exploration_efficiency": coverage_based_exploration_efficiency,
                "episode_reward": episode_reward,
                "moving_average": moving_avg
            })

        if i_episode % render_freq == 0:
            frame = env.render()  

            # Add text to the frame
            frame = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), f"Episode: {i_episode}, Frame: {frame_number}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_number += 1

            if done:
                print(f"Episode {i_episode}: {episode_reward}")
                print(f"Moving average of last 10% of episodes: {moving_avg}")

            
env.close()

# Save the trained model
model_path = "saved_models/model.pt"
torch.save(agent.model.state_dict(), model_path)
print(f"Trained model saved at {model_path}")

# Create video of episodes
if len(frames) > 0:
    output_path = 'results/rendered.mp4'
    fps = 5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Rendered {len(frames)} frames as {output_path}")
else:
    print("No frames rendered")

