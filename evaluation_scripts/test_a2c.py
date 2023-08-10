import gym
import torch
from torch.distributions import Categorical

import gym
import torch
import cv2
import os
import imageio

from a2c import A2CAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1', render_mode='rgb_array')

model_path = "saved_models/model.pt"
agent = A2CAgent(env.observation_space.shape[0], env.action_space.n, device=device, pretrained_weights=model_path)

gpu_number = "2"
if gpu_number:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

episodes = 5
render_freq = 1 # render every 10 episodes
frames = []

for i_episode in range(episodes):
    with torch.no_grad():
        state, info = env.reset()
        episode_reward = 0
        done = False
        frame_number = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if i_episode % render_freq == 0:
                frame = env.render()  

                # Add text to the frame
                frame = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), f"Episode: {i_episode}, Frame: {frame_number}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_number += 1
        print(f"Episode {i_episode}: {episode_reward}")
env.close()

if len(frames) > 0:
    output_path = 'results/test_rendered.mp4'
    fps = 5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Rendered {len(frames)} frames as {output_path}")
else:
    print("No frames rendered")
