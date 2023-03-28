"""
Extended Variational Density Propagation in PyTorch.

This script implements a Bayesian multi-layer perceptron with KL Loss
using extended variational density propagation (VDP) in PyTorch. The code is modified
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
Date: 3/6/2023
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import timeit
import pickle
import numpy as np
import matplotlib.pyplot as plt


def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):
    """
    Compute the negative log-likelihood of a Gaussian distribution.

    Args:
        y_test (torch.Tensor): A tensor of shape (batch_size, num_labels).
        y_pred_mean (torch.Tensor): A tensor of shape (batch_size, num_labels).
        y_pred_sd (torch.Tensor): A tensor of shape (batch_size, num_labels, num_labels).
        num_labels (int): The number of output labels.
        batch_size (int): The batch size.

    Returns:
        torch.Tensor: A scalar tensor representing the negative log-likelihood of the predicted distribution.
    """
    # Add small constant to diagonal for numerical stability
    NS = torch.diag_embed(torch.full((batch_size, num_labels), 1e-3))
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

import torch

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
    
    kl_losses = []
    for i in range(mu.size(1)):
        mu_i = mu[:, i].unsqueeze(1)
        sigma_i = sigma[i, :, :]
        sigma_i_det = torch.det(sigma_i)
        
        term1 = torch.matmul(mu_i.t(), mu_i)
        term2 = torch.trace(sigma_i)
        term3 = mu_i.size(0)
        term4 = torch.log(sigma_i_det) if sigma_i_det.item() > 0 else -1e3
        
        # compute the KL divergence
        kl_losses.append(0.5 * (term1 + term2 - term3 - term4))
        
    return sum(kl_losses)

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
        batch_size = mu_in.size(0)
        
        if sigma_in is None:
            sigma_in = torch.zeros((batch_size, self.size_in, self.size_in))

        # Broadcast to batch size
        # [batch_size size_in size_out] <- [size_in size_out]]
        w_t_expanded = self.w_mu.transpose(1, 0).unsqueeze(0).expand(batch_size, -1, -1) 

        # Linear Layer
        # [batch size_out 1] <- [batch size_out size_in] X [batch size_in 1] + [batch size_out 1]
        mu_out = torch.bmm(w_t_expanded, mu_in) + self.b_mu
        
        # Perform a reparameterization trick
        W_Sigma = torch.log(1. + torch.exp(self.w_sigma))
        B_Sigma = torch.log(1. + torch.exp(self.b_sigma))
        
        # Creat diagonal matrices
        W_Sigma = torch.diag_embed(W_Sigma)
        B_Sigma = torch.diag_embed(B_Sigma)

        # Calculate Sigma_out
        Sigma_out = torch.empty((batch_size, self.size_out, self.size_out))
        for i in range(self.size_out):
            mu_i = self.w_mu[:, i].expand(batch_size, -1).unsqueeze(1)
            sigma_i = W_Sigma[i, :, :].expand(batch_size, -1, -1)
            
            for j in range(self.size_out):
                mu_j = self.w_mu[:, j].expand(batch_size, -1).unsqueeze(2)
                
                tr_sigma_i_and_sigma_in = torch.bmm(sigma_i, sigma_in).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) if i == j else torch.zeros(batch_size)
                mu_w_i_t_sigma_in_mu_w_j = torch.bmm(mu_i, torch.bmm(sigma_in, mu_j)).view(-1)
                mu_in_t_sigma_i_mu_in = torch.bmm(mu_in.transpose(2, 1), torch.bmm(sigma_i, mu_in)).view(-1) if i == j else torch.zeros(batch_size)
            
                Sigma_out[:, i, j] = tr_sigma_i_and_sigma_in + mu_w_i_t_sigma_in_mu_w_j + mu_in_t_sigma_i_mu_in
                
        # Add Reparameterized b_sigma
        for i in range(Sigma_out.size(0)):
            Sigma_out[i] = Sigma_out[i] + B_Sigma
        
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

class RVSoftmax(nn.Module):
    """
    Custom Bayesian Softmax activation function for random variables.

    Attributes:
        None
    """
    def __init__(self):
        super(RVSoftmax, self).__init__()
        
    def softmax(self, x):
        # Apply softmax function along feature dimension
        return torch.softmax(x, dim=0)      
        
    def forward(self, mu_in, Sigma_in):
        """
        Forward pass of the Bayesian Softmax activation function.

        Args:
            mu_in (torch.Tensor): A tensor of shape (batch_size, input_size),
                representing the mean input to the Softmax activation function.
            Sigma_in (torch.Tensor): A tensor of shape (batch_size, input_size, input_size),
                representing the covariance input to the Softmax activation function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors,
                including the mean of the output and the covariance of the output.
        """
        
        # Collect stats
        batch_size, feature_size = mu_in.size()[:2]
        
        # Mean
        mu_out = self.softmax(mu_in)  # shape: [batch_size, output_size]

        # Compute jacobian
        jac = torch.empty((batch_size, feature_size, feature_size))
        for i in range(batch_size):
            jac[i] = torch.autograd.functional.jacobian(self.softmax, mu_in[i].view(-1))

        # Compute covariance
        Sigma_out = torch.bmm(jac, torch.bmm(Sigma_in, jac.permute(0, 2, 1)))
        
        return mu_out, Sigma_out

class exVDPMLP(nn.Module):
    """
    A Bayesian Multi-Layer Perceptron with KL Loss.

    Attributes:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        output_dim (int): The number of output classes.
    """
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super(exVDPMLP, self).__init__()
        self.linear_1 = RVLinearlayer(input_dim, hidden_dim)
        self.relu_1   = RVNonLinearFunc(func = torch.nn.functional.relu)
        self.linear_2 = RVLinearlayer(hidden_dim, output_dim)
        self.softmax  = RVSoftmax()

    def forward(self, x):
        """
        Forward pass of the Bayesian Multi-Layer Perceptron with KL Loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, input_dim),
                representing the input to the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors,
                including the mean of the output, the covariance of the output,
                and the sum of the KL regularization loss terms.
        """
        m, s, kl_1 = self.linear_1(x, None)
        m, s = self.relu_1(m, s)
        m, s, kl_2  = self.linear_2(m, s)
        outputs, Sigma = self.softmax(m, s)
        
        return outputs, Sigma, sum(kl_1, kl_2)
