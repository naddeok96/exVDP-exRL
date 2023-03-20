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

def x_Sigma_w_x_T(x, W_Sigma):
    """
    Compute the outer product of x with itself, multiplied by W_Sigma.

    Args:
        x (torch.Tensor): A tensor of shape (batch_size, feature_dim).
        W_Sigma (torch.Tensor): A tensor of shape (feature_dim, feature_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, feature_dim, feature_dim),
            where the (i,j) entry corresponds to the product of the i-th and j-th
            features of example k in the batch.
    """
    # Collect batch size
    batch_size = x.size(0)

    # Repeat W_sigma
    W_Sigma = W_Sigma.unsqueeze(0).repeat(batch_size, 1, 1)
    print(W_Sigma.size())

    # Get x transpose
    x_t = x.unsqueeze(1).permute(0, 2 ,1)
    print(x_t.size())

    # print(torch.bmm(x.unsqueeze(1), torch.bmm(W_Sigma, x_t)).size())
    exit()
    return torch.bmm(x.unsqueeze(1), torch.bmm(W_Sigma, x_t))

    # # Compute the element-wise multiplication of x with itself
    # # and then sum across the rows (i.e., dim=1) to get xx_t
    # xx_t = torch.sum(torch.mul(x, x), dim=1, keepdim=True)

    # # add an extra dimension to xx_t at position 2
    # xx_t_e = xx_t.unsqueeze(dim=2)
    # # multiply xx_t_e with W_Sigma element-wise
    # # and return the result
    # return torch.mul(xx_t_e, W_Sigma)

def w_t_Sigma_i_w(w_mu, in_Sigma):
    """
    Compute the quadratic form w^T Sigma_i w.

    Args:
        w_mu (torch.Tensor): A tensor of shape (batch_size, feature_dim, 1).
        in_Sigma (torch.Tensor): A tensor of shape (batch_size, feature_dim, feature_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,), representing the quadratic form w^T Sigma_i w for each example in the batch.
    """
    # compute the matrix product of w_mu and in_Sigma
    Sigma_1_1 = torch.matmul(w_mu.transpose(1, 2), in_Sigma)
    Sigma_1_1_w = torch.matmul(Sigma_1_1, w_mu)
    # squeeze out the redundant dimension and return the result as a tensor of shape (batch_size,)
    return Sigma_1_1_w.squeeze()

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    """
    Compute the trace of Sigma_w Sigma_in.

    Args:
        in_Sigma (torch.Tensor): A tensor of shape (batch_size, feature_dim, feature_dim).
        W_Sigma (torch.Tensor): A tensor of shape (feature_dim, feature_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,), representing the trace of Sigma_w Sigma_in for each example in the batch.
    """
    # compute the trace of in_Sigma for each example in the batch
    Sigma_3_1 = torch.einsum('...ii->...', in_Sigma)
    # add two extra dimensions to Sigma_3_1
    Sigma_3_2 = Sigma_3_1.unsqueeze(dim=1)
    Sigma_3_3 = Sigma_3_2.unsqueeze(dim=1)
    # multiply Sigma_3_3 with W_Sigma element-wise
    # and return the result as a tensor of shape (batch_size,)
    return torch.mul(Sigma_3_3, W_Sigma).squeeze()

def activation_Sigma(gradi, Sigma_in):
    """
    Compute the activation covariance matrix.

    Args:
        gradi (torch.Tensor): A tensor of shape (batch_size, feature_dim).
        Sigma_in (torch.Tensor): A tensor of shape (batch_size, feature_dim, feature_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, feature_dim, feature_dim),
            representing the activation covariance matrix for each example in the batch.
    """
    # add an extra dimension to gradi at position 2 and 1
    grad1 = gradi.unsqueeze(dim=2)
    grad2 = gradi.unsqueeze(dim=1)
    # compute the outer product of grad1 and grad2
    outer_product = torch.matmul(grad1, grad2)
    # element-wise multiply Sigma_in with the outer product
    # and return the result
    return torch.mul(Sigma_in, outer_product)

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
        self.w_sigma    = nn.Parameter(torch.Tensor(size_in * size_out))
        self.b_mu       = nn.Parameter(torch.Tensor(size_out, ))
        self.b_sigma    = nn.Parameter(torch.Tensor(size_out))

        # initialize weights and biases using normal and uniform distributions
        nn.init.normal_(self.w_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.2)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-10.0)

    def forward(self, mu_in, sigma_in):
       
        # Extract stats
        batch_size = mu_in.size(0)

        # Broadcast to batch size
        # [batch_size size_in size_out] <- [size_in size_out]]
        w_t_expanded = self.w_mu.transpose(1, 0).unsqueeze(0).expand(batch_size, -1, -1) 

        # Linear Layer
        # [batch size_out 1] <- [batch size_out size_in] X [batch size_in 1] + [batch size_out 1]
        z = torch.bmm(w_t_expanded, mu_in) + self.b_mu

        # Expand variance to diag of cov and perform a reparameterization trick
        # [batch_size, size_in*size_out, size_in*size_out] <- diag(size_in*size_out)
        W_Sigma = torch.diag_embed(torch.log(1. + torch.exp(self.w_sigma))).unsqueeze(0).expand(batch_size, -1, -1) 

        # x Sigma x^T
        # [batch_size size_in 1] x [batch_size, size_in*size_out, size_in*size_out ] x [batch_size 1 size_in]
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma) + torch.log(1. + torch.exp(self.b_sigma))
        exit()

        
        # KL loss
        Term1 = self.w_mu.size(0) * torch.log(torch.log(1. + torch.exp(self.w_sigma)))
        Term2 = torch.sum(torch.sum(torch.abs(self.w_mu)))
        Term3 = self.w_mu.size(0) * torch.log(1. + torch.exp(self.w_sigma))
        
        kl_loss = -0.5 * torch.mean(Term1 - Term2 - Term3)
        
        return mu_out, Sigma_out, kl_loss


class Constant2RVLinearlayer(nn.Module):
    """
    Custom Bayesian Linear Input Layer that takes a constant input and a random variable input.

    Attributes:
        size_in (int): The input size of the layer.
        size_out (int): The output size of the layer.
        w_mu (nn.Parameter): The weight mean parameter.
        w_sigma (nn.Parameter): The weight sigma parameter.
        b_mu (nn.Parameter): The bias mean parameter.
        b_sigma (nn.Parameter): The bias sigma parameter.
    """
    def __init__(self, size_in, size_out):
        super(Constant2RVLinearlayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out

        # initialize weight and bias mean and sigma parameters
        self.w_mu       = nn.Parameter(torch.Tensor(size_in, size_out))
        self.w_sigma    = nn.Parameter(torch.Tensor(size_out,))
        self.b_mu       = nn.Parameter(torch.Tensor(size_out,))
        self.b_sigma    = nn.Parameter(torch.Tensor(size_out,))

        # initialize weights and biases using normal and uniform distributions
        nn.init.normal_(self.w_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.2)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-10.0)

    def forward(self, inputs):
        """
        Forward pass of the Bayesian Linear Input Layer.

        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, num_inputs),
                representing the input to the layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors,
                including the mean of the output, the covariance of the output,
                and the KL divergence loss.
        """
        # Extract stats
        batch_size = inputs.size(0)

        # Mean
        # [batch 1 size_out] <- [batch 1 size_in] X [batch size_in size_out] + [batch 1 size_out]
        mu_out = torch.bmm(inputs.unsqueeze(1), self.w_mu.unsqueeze(0).repeat(batch_size, 1, 1)) + self.b_mu.unsqueeze(0).repeat(batch_size, 1, 1)

        # Variance
        # [size_out size_out] <- diag([size_out])
        W_Sigma = torch.diag(torch.log(1. + torch.exp(self.w_sigma)))\

        print(inputs.size(), W_Sigma.size(), self.b_sigma.size(), torch.log(1. + torch.exp(self.b_sigma)).size())
        
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma) + torch.log(1. + torch.exp(self.b_sigma))
        exit()
        # KL loss
        Term1 = self.w_mu.size(0) * torch.log(torch.log(1. + torch.exp(self.w_sigma)))
        Term2 = torch.sum(torch.sum(torch.abs(self.w_mu)))
        Term3 = self.w_mu.size(0) * torch.log(1. + torch.exp(self.w_sigma))
        
        kl_loss = -0.5 * torch.mean(Term1 - Term2 - Term3)
        
        return mu_out, Sigma_out, kl_loss

class RV2RVLinearlayer(nn.Module):
    """
    Custom Bayesian Linear Input Layer that takes a random variable input and outputs a random variable.

    Attributes:
        size_in (int): The input size of the layer.
        size_out (int): The output size of the layer.
        w_mu (nn.Parameter): The weight mean parameter.
        w_sigma (nn.Parameter): The weight sigma parameter.
        b_mu (nn.Parameter): The bias mean parameter.
        b_sigma (nn.Parameter): The bias sigma parameter.
    """
    def __init__(self, size_in, size_out):
        super(RV2RVLinearlayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out

        # initialize weight and bias mean and sigma parameters
        self.w_mu       = nn.Parameter(torch.Tensor(size_in, size_out))
        self.w_sigma    = nn.Parameter(torch.Tensor(size_out,))
        self.b_mu       = nn.Parameter(torch.Tensor(size_out,))
        self.b_sigma    = nn.Parameter(torch.Tensor(size_out,))

        # initialize weights and biases using normal and uniform distributions
        nn.init.normal_(self.w_mu, mean=0.0, std=0.05)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.2)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-10.0)

    def forward(self, mu_in, Sigma_in):
        """
        Forward pass of the Bayesian Linear Input Layer.

        Args:
            mu_in (torch.Tensor): A tensor of shape (batch_size, size_in),
                representing the input mean to the layer.
            Sigma_in (torch.Tensor): A tensor of shape (batch_size, size_in, size_in),
                representing the input covariance to the layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors,
                including the mean of the output, the covariance of the output,
                and the KL divergence loss.
        """
        # Mean
        mu_out = torch.matmul(mu_in, self.w_mu) + self.b_mu
        print(mu_in.size(), self.w_mu.size(), self.b_mu.size(), mu_out.size())
        exit()

        # Variance
        W_Sigma = torch.diag(torch.log(1. + torch.exp(self.w_sigma)))
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)
        Sigma_3 = tr_Sigma_w_Sigma_in(Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3 + torch.diag(torch.log(1. + torch.exp(self.b_sigma)))
        
        # KL loss
        Term1 = self.w_mu.size(0) * torch.log(torch.log(1. + torch.exp(self.w_sigma)))
        Term2 = torch.sum(torch.sum(torch.abs(self.w_mu)))
        Term3 = self.w_mu.size(0) * torch.log(1. + torch.exp(self.w_sigma))
        kl_loss = -0.5 * torch.mean(Term1 - Term2 - Term3)
        
        return mu_out, Sigma_out, kl_loss

class RVRelu(nn.Module):
    """
    Custom Bayesian ReLU activation function for random variables.

    Attributes:
        None
    """
    def __init__(self):
        super(RVRelu, self).__init__()

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
        # Mean
        mu_out = torch.relu(mu_in)

        # Compute the derivative of the ReLU activation function with respect to the input mean
        gradi = torch.autograd.grad(mu_out, mu_in, grad_outputs=torch.ones_like(mu_out), create_graph=True)[0]

        # Compute the covariance of the output
        Sigma_out = activation_Sigma(gradi, Sigma_in)

        return mu_out, Sigma_out

class RVSoftmax(nn.Module):
    """
    Custom Bayesian Softmax activation function for random variables.

    Attributes:
        None
    """
    def __init__(self):
        super(RVSoftmax, self).__init__()

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
        # Mean
        mu_out = torch.softmax(mu_in, dim=1)  # shape: [batch_size, output_size]

        # Compute the covariance of the output
        pp1 = mu_out.unsqueeze(dim=2)  # shape: [batch_size, output_size, 1]
        pp2 = mu_out.unsqueeze(dim=1)  # shape: [batch_size, 1, output_size]
        ppT = torch.matmul(pp1, pp2)   # shape: [batch_size, output_size, output_size]
        p_diag = torch.diag_embed(mu_out)  # shape: [batch_size, output_size, output_size]

        grad = p_diag - ppT
        Sigma_out = torch.matmul(grad, torch.matmul(Sigma_in, grad.permute(0, 2, 1)))

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
        self.myrelu_1 = RVRelu()
        self.linear_2 = RV2RVLinearlayer(hidden_dim, output_dim)
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
        m, s, kl_loss_1 = self.linear_1(x, None)
        m, s = self.myrelu_1(m, s)
        m, s, kl_loss_2  = self.linear_2(m, s)
        outputs, Sigma = self.softmax(m, s)
        return outputs, Sigma, kl_loss_1 + kl_loss_2
