import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=128, fc2_size=128):
        super(DQN, self).__init__()
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, fc1_size=128, fc2_size=128, device='cpu', gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = device
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.model = DQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.target_model = DQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.type(torch.int64))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        prev_epsilon = self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay * self.epsilon

        return loss, prev_epsilon
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def save(self, path = None):
        if path is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = path = "saved_models/dqn_"  + current_time + ".pt"
            
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        self.update_target_model()
