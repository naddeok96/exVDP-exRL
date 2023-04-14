
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # Embed
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor and Critic
        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

class A2CAgent:
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.001, device='cpu', pretrained_weights=None):
        self.gamma = gamma
        self.device = device
        self.model = ActorCritic(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if pretrained_weights is not None:
            self.model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu')))

    def act(self, state, return_logits=False):
        state = torch.from_numpy(state).float().to(self.device)
        logits, _ = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        if return_logits:
            return action.item(), logits
        else:
            return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor([actions]).to(self.device)
        rewards = torch.FloatTensor([rewards]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        _, values = self.model(states)
        _, next_values = self.model(next_states)

        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        critic_loss = deltas.pow(2).mean()
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        logits, _ = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * deltas.detach()).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

