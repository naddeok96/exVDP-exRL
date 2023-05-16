"""

Author: Kyle Naddeo
Date: 5/3/2023
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels):
    """
    Compute the negative log-likelihood of a Gaussian distribution.

    Args:
        y_test (torch.Tensor): A tensor of shape (batch_size, num_labels).
        y_pred_mean (torch.Tensor): A tensor of shape (batch_size, num_labels).
        y_pred_sd (torch.Tensor): A tensor of shape (batch_size, num_labels, num_labels).
        num_labels (int): The number of output labels.

    Returns:
        torch.Tensor: A scalar tensor representing the negative log-likelihood of the predicted distribution.
    """
    # Collect Stats
    batch_size = y_test.size(0)

    # Declare device
    device = y_pred_mean.device
     
    # Add small constant to diagonal for numerical stability
    NS = torch.diag_embed(torch.full((batch_size, num_labels), 1e-3)).to(device)
    y_pred_sd_ns = y_pred_sd + NS
    
    # Invert sigma
    y_pred_sd_inv = torch.linalg.inv(y_pred_sd_ns)

    # Calculate error
    mu_ = y_pred_mean - y_test

    # First term is error over sigma
    mu_sigma = torch.matmul(mu_.permute(0,2,1), y_pred_sd_inv)
    ms1 = torch.mean(torch.squeeze(torch.matmul(mu_sigma, mu_)))

    # Second term is log determinant
    ms2 = torch.mean(torch.linalg.slogdet(y_pred_sd_ns)[1])

    # Compute the mean
    ms = 0.5 * ms1 + 0.5 * ms2
    return ms

def compute_kl_loss(mu, sigma):
    """
    Computes the Kullback-Leibler (KL) divergence between a Gaussian distribution
    with mean `mu` and covariance `sigma` and a standard normal distribution.

    Parameters:
        mu (torch.Tensor): the mean of the Gaussian distribution (batch_size, num_features)
        sigma (torch.Tensor): the covariance matrix of the Gaussian distribution (batch_size, num_features, num_features)

    Returns:
        torch.Tensor: the KL divergence between the Gaussian distribution and the standard normal distribution (batch_size,)

    References:
        - Derivation from https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        - Assumes the prior is a standard normal distribution

    Formula:
        The KL divergence between a Gaussian distribution q(z|x) with mean `mu` and covariance `sigma` and a
        standard normal distribution p(z) is given by:

        KL(q(z|x) || p(z)) = 0.5 * (mu^T mu + tr(sigma) - k - log(det(sigma)))

        where `tr(sigma)` is the trace of the covariance matrix, `mu^T mu` is the dot product of the mean vector with
        itself, `k` is the dimension of the Gaussian distribution (i.e., the number of features), and `det(sigma)`
        is the determinant of the covariance matrix.
    """
    device = mu.device

    # calculate the KL divergence
    k = torch.tensor(mu.size(0)).view(-1, 1).to(device)
    trace_sigma = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1).view(-1, 1)
    mu_sq = torch.bmm(mu.t().unsqueeze(1), mu.t().unsqueeze(2)).view(-1, 1)
    logdet_sigma = torch.slogdet(sigma)[1].view(-1, 1)
    kl_loss = 0.5 * (trace_sigma + mu_sq - k - logdet_sigma).sum()
    
    return kl_loss

class RVLinearlayer(nn.Module):
    """
    Custom Bayesian Linear Input Layer that takes random variable input.

    Attributes:
        size_in (int): The input size of the layer.
        size_out (int): The output size of the layer.
        w_mu (nn.Parameter): The weight mean parameter.
        w_sigma (nn.Parameter): The weight sigma parameter.
        b_mu (nn.Parameter): The bias mean parameter.
        b_sigma (nn.Parameter): The bias sigma parameter.
    """
    def __init__(self, size_in, size_out):
        super(RVLinearlayer, self).__init__()
        # collect stats
        self.size_in, self.size_out = size_in, size_out

        # initialize weight and bias mean and sigma parameters
        self.w_mu       = nn.Parameter(torch.Tensor(size_in,  size_out))
        self.w_sigma    = nn.Parameter(torch.Tensor(size_out, size_in))
        self.b_mu       = nn.Parameter(torch.Tensor(size_out, 1))
        self.b_sigma    = nn.Parameter(torch.Tensor(size_out,))

        # initialize weights and biases using normal and uniform distributions
        nn.init.uniform_(self.w_mu, a=-(1/size_in)**(1/2), b=(1/size_in)**(1/2))
        nn.init.uniform_(self.b_mu, a=-(1/size_in)**(1/2), b=(1/size_in)**(1/2))
        nn.init.uniform_(self.w_sigma, a=-12, b=-2.0)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-2.0)
        # nn.init.normal_(self.w_mu, mean=0.0, std=0.00005)
        # nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)

    def forward(self, mu_in, sigma_in):
        
        # Extract stats
        device = self.w_mu.device
        batch_size = mu_in.size(0)

        mu_out = torch.matmul(self.w_mu.transpose(1, 0), mu_in.view(batch_size, self.size_in, 1)) + self.b_mu

        # Perform a reparameterization trick
        W_Sigma = torch.log(1. + torch.exp(self.w_sigma))
        B_Sigma = torch.log(1. + torch.exp(self.b_sigma))
        
        # Creat diagonal matrices
        W_Sigma = torch.diag_embed(W_Sigma)
        B_Sigma = torch.diag_embed(B_Sigma)

        # Calculate Sigma_out
        mu_in_t_W_Sigma_mu_in = torch.bmm(torch.matmul(W_Sigma, mu_in.view(batch_size, 1, self.size_in, 1)).view(batch_size, self.size_out, self.size_in), mu_in.view(batch_size, self.size_in, 1)).squeeze()

        if sigma_in is not None:
            tr_W_Sigma_and_sigma_in = torch.matmul(W_Sigma.view(self.size_out, -1), sigma_in.view(-1, batch_size)).view(batch_size, self.size_out)
            mu_w_t_sigma_in_mu_w = torch.matmul(torch.matmul(self.w_mu.t(), sigma_in), self.w_mu)
            Sigma_out = (torch.diag_embed(tr_W_Sigma_and_sigma_in) + mu_w_t_sigma_in_mu_w + torch.diag_embed(mu_in_t_W_Sigma_mu_in)) + B_Sigma
            
        else:
            Sigma_out = torch.diag_embed(mu_in_t_W_Sigma_mu_in) + B_Sigma
      
        # KL loss
        kl_loss = compute_kl_loss(self.w_mu, W_Sigma)
        
        return mu_out, Sigma_out , kl_loss

class RVNonLinearFunc(nn.Module):
    """
    Custom Bayesian ReLU activation function for random variables.

    Attributes:
        None
    """
    def __init__(self, func):
        super(RVNonLinearFunc, self).__init__()
        self.func = func

    def forward(self, mu_in, Sigma_in):
        """
        Forward pass of the Bayesian ReLU activation function.

        Args:
            mu_in (torch.Tensor): A tensor of shape (batch_size, input_size),
                representing the mean input to the ReLU activation function.
            Sigma_in (torch.Tensor): A tensor of shape (batch_size, input_size, input_size),
                representing the covariance input to the ReLU activation function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors,
                including the mean of the output and the covariance of the output.
        """
        # Collect stats
        batch_size = mu_in.size(0)
        
        # Mean
        mu_out = self.func(mu_in)

        # Compute the derivative of the ReLU activation function with respect to the input mean
        gradi = torch.autograd.grad(mu_out, mu_in, grad_outputs=torch.ones_like(mu_out), create_graph=True)[0].view(batch_size,-1)

        # add an extra dimension to gradi at position 2 and 1
        grad1 = gradi.unsqueeze(dim=2)
        grad2 = gradi.unsqueeze(dim=1)
        
        # compute the outer product of grad1 and grad2
        outer_product = torch.bmm(grad1, grad2)
        
        # element-wise multiply Sigma_in with the outer product
        # and return the result
        Sigma_out = torch.mul(Sigma_in, outer_product)

        return mu_out, Sigma_out

class VDPDQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=128, fc2_size=128):
        super(VDPDQN, self).__init__()
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = RVLinearlayer(input_dim, fc1_size)
        self.fc2 = RVLinearlayer(fc1_size, fc2_size)
        self.fc3 = RVLinearlayer(fc2_size, output_dim)

        self.relu   = RVNonLinearFunc(func = torch.nn.functional.relu)

    def forward(self, x):
        m, s, kl_1 = self.fc1(x, None)
        m, s = self.relu(m, s)
        m, s, kl_2  = self.fc2(m, s)
        m, s = self.relu(m, s)
        m, s, kl_3 = self.fc3(m, s)

        return m, s, [kl_1, kl_2, kl_3]

class VDPDQNAgent:
    def __init__(self, state_size, action_size, fc1_size=128, fc2_size=128, device='cpu', gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, kl_factor=0.0001, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.kl_factor = kl_factor
        self.device = device
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.model = VDPDQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.target_model = VDPDQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, rewards, next_state, done):
        self.memory.append((state, action, rewards, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values, _, _ = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None, 3*[None], None
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(torch.stack(rewards,dim=0)).to(self.device)
        next_states = torch.FloatTensor(torch.stack(next_states,dim=0)).to(self.device)
        dones = torch.FloatTensor(torch.stack(dones,dim=0)).to(self.device)

        current_q_values, current_q_sigmas, current_kl_losses = self.model(states)

        target_q_values = torch.zeros_like(current_q_values)
        for i in range(self.action_size):
            next_q_values_i, _, _ = self.target_model(next_states[:,i,:])
            next_action_q_value_i = next_q_values_i.squeeze().max(1)[0].detach().squeeze()

            target_q_values[:,i,0] = rewards[:,i] + (1 - dones[:,i]) * self.gamma * next_action_q_value_i

        loss = nll_gaussian(target_q_values, current_q_values, current_q_sigmas, self.action_size)
        total_loss = loss + self.kl_factor*sum(current_kl_losses)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        prev_epsilon = self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay * self.epsilon

        return total_loss, loss, current_kl_losses, prev_epsilon
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path = "saved_models/vdp_dqn.pt"):
        torch.save(self.model.state_dict(), path)
