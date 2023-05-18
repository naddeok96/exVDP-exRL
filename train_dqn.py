
import sys
sys.path.append("..")
import kyus_gym.gym as gym

import torch
import os
import wandb

from dqn import DQNAgent

def train(env, agent, batch_size=32, episodes=500, max_steps=200):
    num_success = 0
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                reward = -100

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            loss, prev_epsilon = agent.replay(batch_size)

            if done or step == max_steps - 1:
                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            # Log
            wandb.log({
                "total_reward": total_reward,
                "i_episode": e,
                "step": step,
                "total_reward": total_reward,
                "loss": loss,
                "prev_epsilon": prev_epsilon,
            })

        if total_reward >= max_steps:
            num_success += 1
            if num_success == 100:
                print(f"CartPole solved in {e+1} episodes!")
                agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_dqn_at_100_successes.pt")

    agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_dqn.pt")
    
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
    wandb.init(project="DQN LunarLander", entity="naddeok") #, mode="disabled")
    wandb.config.fc1_size = fc1_size = 128
    wandb.config.fc2_size = fc2_size = 128  

    wandb.config.gamma = gamma                  = 0.99
    wandb.config.epsilon = epsilon              = 1.0
    wandb.config.epsilon_min = epsilon_min      = 0.01
    wandb.config.epsilon_decay = epsilon_decay  = 0.9995
    wandb.config.learning_rate = learning_rate  = 0.0001
    wandb.config.memory_size = memory_size      = 10000

    wandb.config.batch_size = batch_size    = 32
    wandb.config.episodes = episodes        = 10000
    wandb.config.max_steps = max_steps      = 300

    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)

    # Standard train
    train(env, agent, batch_size, episodes, max_steps)
    

