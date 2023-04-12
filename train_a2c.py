import gym
import torch
import cv2
import os
import imageio
from tqdm import tqdm

from a2c import A2CAgent

gpu_number = "2"
if gpu_number:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1', render_mode='rgb_array')
agent = A2CAgent(env.observation_space.shape[0], env.action_space.n, device=device)

episodes = 500
max_time_steps = 500
render_freq = 50 # render every 10 episodes
frames = []

for i_episode in tqdm(range(episodes),desc="Episodes"):
    state, info = env.reset()
    episode_reward = 0
    done = False
    frame_number = 0
    time_step = 0
    while not done or time_step > max_time_steps:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        time_step += 1
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
env.close()

# Save the trained model
model_path = "saved_models/model.pt"
torch.save(agent.model.state_dict(), model_path)
print(f"Trained model saved at {model_path}")

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

