import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

import torch.nn as nn
import torch

class ConvDQN(nn.Module):
    def __init__(self, input_shape, action_size,
                 conv1_out_channels=32, conv1_kernel_size=8, conv1_stride=4,
                 conv2_out_channels=64, conv2_kernel_size=4, conv2_stride=2,
                 conv3_out_channels=64, conv3_kernel_size=3, conv3_stride=1,
                 fc1_size=512):
        super(ConvDQN, self).__init__()

        # Assuming input shape is in the format: (height, width, channels)
        self.input_shape = input_shape
        self.action_size = action_size

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[-1], conv1_out_channels, kernel_size=conv1_kernel_size, stride=conv1_stride),  # input_shape[2] is the channel dimension
            nn.ReLU(),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=conv2_kernel_size, stride=conv2_stride),
            nn.ReLU(),
            nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=conv3_kernel_size, stride=conv3_stride),
            nn.ReLU()
        )
        
        self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, action_size)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change shape from [batch, H, W, C] to [batch, C, H, W]
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)



    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[::-1])).view(1, -1).size(1)  # Reverse the input shape to (channels, height, width) for PyTorch


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=128, fc2_size=128):
        super(DQN, self).__init__()
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, 
                 state_size,
                 action_size,
                 fc1_size=128,
                 fc2_size=128,
                 device='cpu',
                 gamma=0.99,
                 explore=True,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 memory_size=10000,
                 use_conv=True,
                 conv1_out_channels=32, conv1_kernel_size=8, conv1_stride=4,
                 conv2_out_channels=64, conv2_kernel_size=4, conv2_stride=2,
                 conv3_out_channels=64, conv3_kernel_size=3, conv3_stride=1):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.explore = explore
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = device
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.use_conv = use_conv


        if use_conv:
            Network = ConvDQN
            network_args = {
                'conv1_out_channels': conv1_out_channels, 
                'conv1_kernel_size': conv1_kernel_size,
                'conv1_stride': conv1_stride,
                'conv2_out_channels': conv2_out_channels,
                'conv2_kernel_size': conv2_kernel_size,
                'conv2_stride': conv2_stride,
                'conv3_out_channels': conv3_out_channels,
                'conv3_kernel_size': conv3_kernel_size,
                'conv3_stride': conv3_stride,
                'fc1_size': fc1_size
            }
        else:
            Network = DQN
            network_args = {'fc1_size': fc1_size,
                            'fc2_size': fc2_size}
        
        self.model = Network(state_size, action_size, **network_args).to(self.device)
        self.target_model = Network(state_size, action_size, **network_args).to(self.device)
        
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, return_q_values=False):
        if np.random.rand() <= self.epsilon and self.explore:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        
        if return_q_values:
            return action, q_values
        else:
            return action

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
        self.target_model.to(self.device)
        self.target_model.eval()

    def save(self, path = None):
        if path is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = path = "saved_models/dqn_"  + current_time + ".pt"
            
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)
        self.update_target_model()
