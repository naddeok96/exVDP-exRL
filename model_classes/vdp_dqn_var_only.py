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
from datetime import datetime
import math
import time

import torch

import torch

def nll_gaussian(y_test, y_pred_mean, y_pred_sigma_sq, return_components=False):
    """
    Compute the negative log-likelihood of a Gaussian distribution.

    Args:
        y_test (torch.Tensor): A tensor of shape (batch_size, 1).
        y_pred_mean (torch.Tensor): A tensor of shape (batch_size, 1).
        y_pred_sigma_sq (torch.Tensor): A tensor of shape (batch_size,).
        return_components (bool): Whether to return individual loss components.

    Returns:
        torch.Tensor: A scalar tensor representing the negative log-likelihood of the predicted distribution.
    """
    # Calculate error
    error = y_pred_mean - y_test
    mse = (error**2).mean().item()

    # Add small constant to variance for numerical stability
    epsilon = 1e-8
    variance = y_pred_sigma_sq + epsilon

    # First term is error over sigma
    error_over_sigma = torch.mean((error / variance).pow(2))

    # Second term is log determinant
    log_det = torch.mean(torch.log(variance))

    # Compute the mean
    nll_loss = (error_over_sigma + log_det) / 2

    if return_components:
        return nll_loss, mse, error_over_sigma, log_det
    else:
        return nll_loss

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
    k = torch.tensor(mu.size(0)).to(device)
    norm_squared = torch.norm(mu, p=2, dim=1).pow(2)
    log_sigma_squared = torch.log(sigma) 
    kl_loss = 0.5 * torch.sum(norm_squared - k * (1 - sigma + log_sigma_squared))
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
        
        # Extract input info
        batch_size = mu_in.size(0)
        device = mu_in.device

        mu_out = (torch.matmul(self.w_mu.T.unsqueeze(0), mu_in.view(batch_size, self.size_in, 1)).squeeze(2)) + self.b_mu.view(-1)
        
    
        # Perform a reparameterization trick
        W_sigma = torch.log(1. + torch.exp(self.w_sigma))
        B_sigma = torch.log(1. + torch.exp(self.b_sigma))

        # Numerical stability
        NS = torch.full((W_sigma.size()), 1e-6).to(device)
        W_sigma = W_sigma + NS

        NS = torch.full((B_sigma.size()), 1e-6).to(device)
        B_sigma = B_sigma + NS

        # # Calculate sigma_out
        if sigma_in is not None:
            tr_W_sigma_and_sigma_in = torch.einsum('ij,bj->bi', W_sigma, sigma_in)
            mu_w_t_sigma_in_mu_w = torch.einsum('ij,bj->bi', (self.w_mu.t()**2), sigma_in)

        mu_in_t_W_sigma_mu_in = torch.einsum('bi,oi->bo', (mu_in**2), W_sigma)

        sigma_out = tr_W_sigma_and_sigma_in + mu_w_t_sigma_in_mu_w + mu_in_t_W_sigma_mu_in + B_sigma if sigma_in is not None else mu_in_t_W_sigma_mu_in + B_sigma

        # KL loss
        w_kl_loss = compute_kl_loss(self.w_mu, W_sigma)
        b_kl_loss = compute_kl_loss(self.b_mu, B_sigma)
        
        return mu_out, sigma_out, w_kl_loss, b_kl_loss

class RVNonLinearFunc(nn.Module):
    """
    Custom Bayesian ReLU activation function for random variables.

    Attributes:
        None
    """
    def __init__(self, func):
        super(RVNonLinearFunc, self).__init__()
        self.func = func

    def forward(self, mu_in, sigma_in):
        """
        Forward pass of the Bayesian ReLU activation function.

        Args:
            mu_in (torch.Tensor): A tensor of shape (batch_size, input_size),
                representing the mean input to the ReLU activation function.
            sigma_in (torch.Tensor): A tensor of shape (batch_size, input_size, input_size),
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
        gradi = torch.autograd.grad(mu_out, mu_in, grad_outputs=torch.ones_like(mu_out), create_graph=True, retain_graph=True)[0].view(batch_size,-1)
        
        # Calculate the diagonal elements of the outer product
        diag_outer_product = torch.mul(gradi, gradi).view(batch_size, -1)
        
        # element-wise multiply sigma_in with the outer product
        # and return the result
        sigma_out = torch.mul(sigma_in, diag_outer_product)

        return mu_out, sigma_out

class VarOnlyVDPDQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=128, fc2_size=128):
        super(VarOnlyVDPDQN, self).__init__()
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = RVLinearlayer(input_dim, fc1_size)
        self.fc2 = RVLinearlayer(fc1_size, fc2_size)
        self.fc3 = RVLinearlayer(fc2_size, output_dim)

        self.relu   = RVNonLinearFunc(func = torch.nn.functional.relu)

    def forward(self, x, return_sigmas = False):
        m, s1, w_kl_1, b_kl_1 = self.fc1(x, None)
        m, s2 = self.relu(m, s1)
        m, s3, w_kl_2, b_kl_2  = self.fc2(m, s2)
        m, s4 = self.relu(m, s3)
        m, s5, w_kl_3, b_kl_3 = self.fc3(m, s4)

        kl_losses = {   "w" : { "fc1": w_kl_1,
                                "fc2": w_kl_2,
                                "fc3": w_kl_3},
                        "b" : { "fc1": b_kl_1,
                                "fc2": b_kl_2,
                                "fc3": b_kl_3},   
                         }

        if not return_sigmas:
            return m, s5, kl_losses
        else:   
            sigmas = {"fc1" : torch.norm(s1).item(),
                      "relu1": torch.norm(s2).item(),
                      "fc2" : torch.norm(s3).item(),
                      "relu2" : torch.norm(s4).item(),
                      "fc3" : torch.norm(s5).item()}
            
            return m, s5, kl_losses, sigmas
            
class VarOnlyVDPDQNAgent:
    def __init__(self, state_size, action_size, fc1_size=128, fc2_size=128, device='cpu', gamma=0.99, k=0.1, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, explore=True, learning_rate=0.001, kl_w_factor=0.0001, kl1_w_factor = 1, kl2_w_factor = 1, kl3_w_factor = 1, kl_b_factor=0.0001, kl1_b_factor = 1, kl2_b_factor = 1, kl3_b_factor = 1, memory_size=10000, teacher_model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon    
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.k = k
        self.explore = explore
        self.learning_rate = learning_rate

        self.kl_w_factor = kl_w_factor
        self.kl1_w_factor = kl1_w_factor
        self.kl2_w_factor = kl2_w_factor
        self.kl3_w_factor = kl3_w_factor

        self.kl_b_factor = kl_b_factor
        self.kl1_b_factor = kl1_b_factor
        self.kl2_b_factor = kl2_b_factor
        self.kl3_b_factor = kl3_b_factor

        self.device = device
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.model = VarOnlyVDPDQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.teacher_model = teacher_model
        self.target_model = VarOnlyVDPDQN(state_size, action_size, fc1_size, fc2_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, rewards, next_state, done):
        self.memory.append((state, action, rewards, next_state, done))

    def distill_remember(self, state, q_values):
        self.memory.append((state, q_values))

    def act(self, state, return_sigma = False, use_uncert_epsilon = False):
    
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values, q_sigmas, _ = self.model(state)

        if use_uncert_epsilon:
            self.epsilon = self.get_epsilon(q_sigmas)
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon_decay * self.epsilon

        if self.explore and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = torch.argmax(q_values).item()

        if return_sigma:
            return action, q_values, q_sigmas
        else:
            return action

    def replay(self, batch_size, return_uncertainty_values = False):
        if len(self.memory) < batch_size:
            if return_uncertainty_values:
                return 12*[None]
            else:
                return None, None, None
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        if return_uncertainty_values:
            current_q_values, current_q_sigmas, current_kl_losses, predictive_sigmas = self.model(states, return_sigmas = return_uncertainty_values)
            current_q_sigmas = current_q_sigmas.gather(1, actions.type(torch.int64)).squeeze()
        else:
            current_q_values, current_q_sigmas, current_kl_losses = self.model(states)
            
        current_q_values = current_q_values.gather(1, actions.type(torch.int64)).squeeze()

        next_q_values, _, _ = self.target_model(next_states)
        next_q_values = next_q_values.squeeze().max(1)[0].detach().squeeze()

        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        nll_loss, mse, error_over_sigma, log_determinant = nll_gaussian( y_test=target_q_values, 
                                                                    y_pred_mean=current_q_values, 
                                                                    y_pred_sigma_sq=current_q_sigmas, 
                                                                    return_components=return_uncertainty_values)
        nll_loss = nll_loss + 10 # Offset to keep positive
        
        weighted_w_kl_loss = self.kl_w_factor * (self.kl1_w_factor*current_kl_losses["w"]["fc1"] + self.kl2_w_factor*current_kl_losses["w"]["fc2"] + self.kl3_w_factor*current_kl_losses["w"]["fc3"])
        weighted_b_kl_loss = self.kl_b_factor * (self.kl1_b_factor*current_kl_losses["b"]["fc1"] + self.kl2_b_factor*current_kl_losses["b"]["fc2"] + self.kl3_b_factor*current_kl_losses["b"]["fc3"])
        
        # weighted_predictive_sigmas = self.pred_factor * (self.pred_fc1_factor*predictive_sigmas["fc1"] + self.pred_relu1_factor*predictive_sigmas["relu1"] + self.pred_fc2_factor*predictive_sigmas["fc2"] + self.pred_relu2_factor*predictive_sigmas["relu2"] + self.pred_fc3_factor*predictive_sigmas["fc3"])
        weighted_predictive_sigmas = 0.1 * (predictive_sigmas["fc1"] + predictive_sigmas["relu1"] + (1/1600)*predictive_sigmas["fc2"] + (1/1600)*predictive_sigmas["relu2"] + (1/5e6)*predictive_sigmas["fc3"])
        total_loss = nll_loss + weighted_w_kl_loss + weighted_b_kl_loss # + weighted_predictive_sigmas
        
        if torch.isnan(total_loss):
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.zero_grad()
        total_loss.backward()

        if [torch.isnan(param.grad).any().item() for name, param in self.model.named_parameters() if "fc3.w_sigma" in name ][0]:
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.step()

        if return_uncertainty_values:
            model_sigmas = self.get_covariance_matrices()
            return total_loss, nll_loss, weighted_w_kl_loss, weighted_b_kl_loss, weighted_predictive_sigmas, current_kl_losses, model_sigmas, predictive_sigmas, mse, error_over_sigma, log_determinant, len(self.memory)
        else:
            return total_loss, nll_loss, current_kl_losses
        
    def distill_replay(self, batch_size, return_uncertainty_values = False):
        if len(self.memory) < batch_size:
            if return_uncertainty_values:
                return None, None, None, None, None, None, None, None, None, None, None, None
            else:
                return None, None, None
        
        minibatch = random.sample(self.memory, batch_size)
        states, target_q_values = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        target_q_values = torch.stack(target_q_values,dim=0).permute(0,2,1)

        if return_uncertainty_values:
            current_q_values, current_q_sigmas, current_kl_losses, predictive_sigmas = self.model(states, return_sigmas=return_uncertainty_values)
        else:
            current_q_values, current_q_sigmas, current_kl_losses = self.model(states)

        nll_loss, mse, error_over_sigma, log_determinant = nll_gaussian(  y_test=target_q_values, 
                                                                            y_pred_mean=current_q_values, 
                                                                            y_pred_sigma_sq=current_q_sigmas, 
                                                                            num_labels=self.action_size, 
                                                                            return_components=return_uncertainty_values)
        nll_loss = nll_loss + 10 # Offset to keep positive
        
        weighted_w_kl_loss = self.kl_w_factor * (self.kl1_w_factor*current_kl_losses["w"]["fc1"] + self.kl2_w_factor*current_kl_losses["w"]["fc2"] + self.kl3_w_factor*current_kl_losses["w"]["fc3"])
        weighted_b_kl_loss = self.kl_b_factor * (self.kl1_b_factor*current_kl_losses["b"]["fc1"] + self.kl2_b_factor*current_kl_losses["b"]["fc2"] + self.kl3_b_factor*current_kl_losses["b"]["fc3"])
        
        # weighted_predictive_sigmas = self.pred_factor * (self.pred_fc1_factor*predictive_sigmas["fc1"] + self.pred_relu1_factor*predictive_sigmas["relu1"] + self.pred_fc2_factor*predictive_sigmas["fc2"] + self.pred_relu2_factor*predictive_sigmas["relu2"] + self.pred_fc3_factor*predictive_sigmas["fc3"])
        weighted_predictive_sigmas = 0.1 * (predictive_sigmas["fc1"] + predictive_sigmas["relu1"] + (1/1600)*predictive_sigmas["fc2"] + (1/1600)*predictive_sigmas["relu2"] + (1/5e6)*predictive_sigmas["fc3"])
        total_loss = nll_loss + weighted_w_kl_loss + weighted_b_kl_loss #+ 0.01*mse# + weighted_predictive_sigmas
        
        if torch.isnan(total_loss):
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        if [torch.isnan(param.grad).any().item() for name, param in self.model.named_parameters() if "fc3.w_sigma" in name ][0]:
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.step()

        if return_uncertainty_values:
            model_sigmas = self.get_covariance_matrices()
            return total_loss, nll_loss, weighted_w_kl_loss, weighted_b_kl_loss, weighted_predictive_sigmas, current_kl_losses, model_sigmas, predictive_sigmas, mse, error_over_sigma, log_determinant, len(self.memory)
        else:
            return total_loss, nll_loss, current_kl_losses
        
    def replay_distill_from_target(self, batch_size):
        if len(self.memory) < batch_size:
            return 12*[None]
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values, current_q_sigmas, current_kl_losses, predictive_sigmas = self.model(states, return_sigmas=True)
        next_q_values, _, _, _ = self.target_model(next_states, return_sigmas=True)
        one_step_target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values.max(1)[0].detach().squeeze()

        target_q_values, _, _, _ = self.target_model(next_states, return_sigmas=True)
        mask = torch.zeros_like(target_q_values, dtype=torch.bool)
        mask[torch.arange(target_q_values.size(0)), actions.squeeze(), :] = True
        target_q_values[mask] = one_step_target_q_values

        nll_loss, mse, error_over_sigma, log_determinant = nll_gaussian(y_test              = target_q_values,
                                                                        y_pred_mean         = current_q_values,
                                                                        y_pred_sigma_sq     = current_q_sigmas,
                                                                        num_labels          = self.action_size,
                                                                        return_components   = True)
        nll_loss = nll_loss + 10 # Offset to keep positive
        
        weighted_w_kl_loss = self.kl_w_factor * (self.kl1_w_factor*current_kl_losses["w"]["fc1"] + self.kl2_w_factor*current_kl_losses["w"]["fc2"] + self.kl3_w_factor*current_kl_losses["w"]["fc3"])
        weighted_b_kl_loss = self.kl_b_factor * (self.kl1_b_factor*current_kl_losses["b"]["fc1"] + self.kl2_b_factor*current_kl_losses["b"]["fc2"] + self.kl3_b_factor*current_kl_losses["b"]["fc3"])
        
        # weighted_predictive_sigmas = self.pred_factor * (self.pred_fc1_factor*predictive_sigmas["fc1"] + self.pred_relu1_factor*predictive_sigmas["relu1"] + self.pred_fc2_factor*predictive_sigmas["fc2"] + self.pred_relu2_factor*predictive_sigmas["relu2"] + self.pred_fc3_factor*predictive_sigmas["fc3"])
        weighted_predictive_sigmas = 0.1 * (predictive_sigmas["fc1"] + predictive_sigmas["relu1"] + (1/1600)*predictive_sigmas["fc2"] + (1/1600)*predictive_sigmas["relu2"] + (1/5e6)*predictive_sigmas["fc3"])
        total_loss = nll_loss + weighted_w_kl_loss + weighted_b_kl_loss
        
        if torch.isnan(total_loss):
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        if [torch.isnan(param.grad).any().item() for name, param in self.model.named_parameters() if "fc3.w_sigma" in name ][0]:
            print("The loss is NaN") 
            for i in range(24 * 60 * 60):
                time.sleep(1)

        self.optimizer.step()

        model_sigmas = self.get_covariance_matrices()
        return total_loss, nll_loss, weighted_w_kl_loss, weighted_b_kl_loss, weighted_predictive_sigmas, current_kl_losses, model_sigmas, predictive_sigmas, mse, error_over_sigma, log_determinant, len(self.memory)
        
    def get_epsilon(self, q_sigmas):
        # Calculate the determinants of sigma matrices
        det_sigmas = torch.linalg.det(q_sigmas)
        
        # Calculate ε using the equation: ε = (1 - exp(-k * |det(sigma)|))
        epsilon = 1 - torch.exp(-self.k * det_sigmas)
        
        return epsilon.item()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def get_covariance_matrices(self):
        layer_magnitudes = {}
        for name, param in self.model.named_parameters():
            if 'sigma' in name.lower():
                layer_magnitude = torch.norm(param).item()
                layer_magnitudes[name] = layer_magnitude
        return layer_magnitudes
                
    def save(self, path = None):
        if path is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = "saved_models/vdp_dqn_"  + current_time + ".pt"
            
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)
        self.update_target_model()


if __name__ == "__main__":
    agent = VDPDQNAgent(6,3)
    agent.model(torch.randn(2,6))
    