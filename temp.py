
import gym

import torch
import os
import wandb

from dqn import DQNAgent

def train(env, agent, batch_size=32, episodes=500, max_steps=200, target_reward=-100, target_successes=100):
    num_success = 0
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state) 
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                reward = -10

            # agent.remember(state, action, reward, next_state, done)
            state = next_state

            # loss, prev_epsilon = agent.replay(batch_size)

            # Log
            wandb.log({
                "step": step,
                "total_reward": total_reward,
                "action" : action
                # "loss": loss,
                # "prev_epsilon": prev_epsilon,
            })

            if done or step == max_steps - 1:
                wandb.log({
                    "total_reward": total_reward,
                    "i_episode": e
                })


                # agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if total_reward <= target_reward:
            num_success += 1
        else:
            num_success = 0

        if num_success == target_successes:
            print(f"Acrobot solved in {e+1} episodes!")
            agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_dqn_at_100_successes.pt")

    agent.save("saved_models/" + wandb.run.project + "_" + wandb.run.name + "_dqn.pt")

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
    wandb.init(project="DQN Acrobot", entity="naddeok") #, mode="disabled")
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
    wandb.config.target_reward = target_reward = -100
    wandb.config.target_successes = target_successes = 100

    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, fc1_size=fc1_size, fc2_size=fc2_size, device=device, gamma=gamma,epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_size=memory_size)
    agent.load_model("saved_models/DQN Acrobot_olive-pyramid-9_dqn_at_converge.pt")

    # Standard train
    train(env, agent, batch_size, episodes, max_steps, target_reward, target_successes)
