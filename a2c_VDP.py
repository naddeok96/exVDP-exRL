"""
Extended Variational Density Propagation for Advantage Actor Critic (A2C) in PyTorch.

This script implements a Bayesian multi-layer perceptron with KL Loss
using extended variational density propagation (VDP) for A2C in PyTorch. The code is modified
from exVDP_MNIST.py in https://github.com/dimahdera/Robust-Anomaly-Detection,
which is an implementation of the paper "PremiUm-CNN: Propagating Uncertainty Towards 
Robust Convolutional Neural Networks" by Dera et al.
The original code was authored by Dimah Dera.

The script defines several custom PyTorch modules, including `Constant2RVLinearlayer`,
`RV2RVLinearlayer`, `RVRelu`, and `RVSoftmax`, which are used to build the Bayesian MLP.
It also defines a `nll_gaussian` function for computing the negative log-likelihood of
a Gaussian distribution, as well as an `exVDPMLP` class that encapsulates the entire
Bayesian MLP.

Author: Kyle Naddeo
Date: 4/12/2023
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    mu_sigma = torch.matmul(mu_.unsqueeze(dim=1), y_pred_sd_inv)
    ms1 = torch.mean(torch.squeeze(torch.matmul(mu_sigma, mu_.unsqueeze(dim=2))))

    # Second term is log determinant
    ms2 = torch.mean(torch.squeeze(torch.linalg.slogdet(y_pred_sd_ns)[1]))

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
        nn.init.normal_(self.w_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.0)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-2.0)

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
        mu_in_t_W_Sigma_mu_in = torch.bmm(torch.matmul(W_Sigma, mu_in.transpose(2, 1).unsqueeze(-1)).squeeze(-1), mu_in).squeeze()

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

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()

        self.relu   = RVNonLinearFunc(func = torch.nn.functional.relu)
        
        self.fc1 = RVLinearlayer(input_shape, 256) # self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = RVLinearlayer(256, 128) # nn.Linear(256, 128)
        self.actor = RVLinearlayer(128, num_actions) # nn.Linear(128, num_actions)
        self.critic = RVLinearlayer(128, 1) # nn.Linear(128, 1)

    def forward(self, x):
        # Embed
        m, s, kl_1 = self.fc1(x, None) # x = self.relu(self.fc1(x))
        m, s = self.relu_1(m, s)

        m, s, kl_2 = self.fc2(m, s) # x = self.relu(self.fc2(x))
        m, s = self.relu_1(m, s)

        # Actor and Critic
        logits, actor_sigma, actor_kl = self.actor(m, s)
        value, critic_sigma, critic_kl = self.critic(m, s)

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

