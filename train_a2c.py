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

# Hyperparameters
episodes = int(1e4)
max_time_steps = 300 
save_freq = int(1e3) # save model every 50 episodes
render_freq = int(1e2) # render every 50 episodes
model_path = "saved_models/a2c_model.pt"
output_path = 'results/a2c_model.mp4'

# Initialize GPU usage
gpu_number = "2"
if gpu_number:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Get state space size to calculate coverage
if isinstance(env.observation_space, gym.spaces.Box):
    # Continous space
    state_space_size = 1 # Note state space size is inf but we use 1 for division in calculate_coverage_exploration_efficiency
else:
    # Discrete space
    state_space_size = env.observation_space.n # total number of possible states

# Initialize agent
agent = A2CAgent(env.observation_space.shape[0], env.action_space.n, device=device)

# Initialize WandB
wandb.init(project="A2C", entity="naddeok")
wandb.config.episodes = episodes
wandb.config.max_time_steps = max_time_steps


# Initialize storage
frames = []
episode_rewards = []
visited_states = set() # keep track of visited states

# Train
for i_episode in tqdm(range(episodes),desc="Episodes"):
    state, info = env.reset()
    episode_reward = 0
    done = False
    frame_number = 0
    time_step = 0
    while not done and time_step < max_time_steps:
        # Episode step
        action, logits = agent.act(state, return_logits=True)
        next_state, reward, done, _, _ = env.step(action)

        # calculate exploration efficiency
        visited_states.add(tuple(state))
        novelty_based_exploration_efficiency = calculate_novelty_exploration_efficiency(next_state, visited_states)
        entropy_based_exploration_efficiency = calculate_entropy_exploration_efficiency(logits)
        

        # Backward pass
        agent.update(state, action, reward, next_state, done)

        # Store
        state = next_state
        episode_reward += reward
        time_step += 1
        
        # Log
        if done or time_step >= max_time_steps:
            # Calculate moving average
            episode_rewards.append(episode_reward)
            moving_avg = np.mean(episode_rewards[-int(episodes*0.1):])

            # Calculate coverage
            coverage_based_exploration_efficiency = calculate_coverage_exploration_efficiency(visited_states, state_space_size)
            
            # Log
            wandb.log({
                "i_episode": i_episode,
                "novelty_based_exploration_efficiency": novelty_based_exploration_efficiency,
                "entropy_based_exploration_efficiency": entropy_based_exploration_efficiency,
                "coverage_based_exploration_efficiency": coverage_based_exploration_efficiency,
                "episode_reward": episode_reward,
                "moving_average": moving_avg
            })

        # Display
        if i_episode % render_freq == 0:
            frame = env.render()  

            # Add text to the frame
            frame = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), f"Episode: {i_episode}, Frame: {frame_number}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_number += 1

            if done:
                print(f"Episode {i_episode} Reward: {episode_reward}")
                print(f"Moving average of last 10% of episodes: {moving_avg}")

        if i_episode % save_freq == 0:
            # Save the trained model
            torch.save(agent.model.state_dict(), model_path)
            print(f"Trained model saved at {model_path}")

# Close the wandb and env run when finished
wandb.finish()            
env.close()

# Save the trained model
torch.save(agent.model.state_dict(), model_path)
print(f"Trained model saved at {model_path}")

# Create video of episodes
if len(frames) > 0:
    fps = 5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Rendered {len(frames)} frames as {output_path}")
else:
    print("No frames rendered")

