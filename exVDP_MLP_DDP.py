# Extended Variational Density Propagation in Pytorch
# Modified from exVDP_MNIST.py in https://github.com/dimahdera/Robust-Anomaly-Detection
# Author : Kyle Naddeo
# Date : 3/6/2023

# Imports
import os
import argparse
import torch
import torch.nn as nn
import timeit
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.size(0)
    xx_t = torch.sum(torch.mul(x, x), axis=1, keepdim=True)
    xx_t_e = xx_t.unsqueeze(dim=2)
    return torch.mul(xx_t_e, W_Sigma)

def w_t_Sigma_i_w(w_mu, in_Sigma):
    Sigma_1_1 = torch.matmul(w_mu.t(), in_Sigma)
    Sigma_1_2 = torch.matmul(Sigma_1_1, w_mu)
    return Sigma_1_2

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    Sigma_3_1 = torch.einsum('...ii->...', in_Sigma)
    Sigma_3_2 = Sigma_3_1.unsqueeze(dim=1)
    Sigma_3_3 = Sigma_3_2.unsqueeze(dim=1)
    return torch.mul(Sigma_3_3, W_Sigma)

def activation_Sigma(gradi, Sigma_in):
    grad1 = gradi.unsqueeze(dim=2)
    grad2 = gradi.unsqueeze(dim=1)
    return torch.mul(Sigma_in, torch.matmul(grad1, grad2))


def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):
    # Numerical stability
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
    ms = 0.5 * ms1 + 0.5 * ms2

    return ms

class Constant2RVLinearlayer(nn.Module): # Constant Input X Random Variable Layer
    """ Custom Bayesian Linear Input Layer """
    def __init__(self, size_in, size_out):
        super(Constant2RVLinearlayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out

        self.w_mu       = nn.Parameter(torch.Tensor(size_in, size_out))
        self.w_sigma    = nn.Parameter(torch.Tensor(size_out,)) 
        
        self.b_mu = nn.Parameter(torch.Tensor(size_out,))
        self.b_sigma = nn.Parameter(torch.Tensor(size_out,))

        # initialize weights and biases
        nn.init.normal_(self.w_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.2)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-10.0)

    def forward(self, inputs):
        # Mean
        mu_out = torch.matmul(inputs, self.w_mu) + self.b_mu                         # Mean of the output
        
        # Variance
        W_Sigma = torch.diag(torch.log(1. + torch.exp(self.w_sigma)))
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma) + torch.log(1. + torch.exp(self.b_sigma))
        
        # KL loss
        Term1 = self.w_mu.size(0) * torch.log(torch.log(1. + torch.exp(self.w_sigma)))
        Term2 = torch.sum(torch.sum(torch.abs(self.w_mu)))
        Term3 = self.w_mu.size(0) * torch.log(1. + torch.exp(self.w_sigma))
        
        kl_loss = -0.5 * torch.mean(Term1 - Term2 - Term3)
        
        return mu_out, Sigma_out, kl_loss

class RV2RVLinearlayer(nn.Module): # Random Variable Input X Random Variable Layer
    """ Custom Bayesian Linear Input Layer """
    def __init__(self, size_in, size_out):
        super(RV2RVLinearlayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out

        self.w_mu       = nn.Parameter(torch.Tensor(size_in, size_out))
        self.w_sigma    = nn.Parameter(torch.Tensor(size_out,)) 
        
        self.b_mu = nn.Parameter(torch.Tensor(size_out,))
        self.b_sigma = nn.Parameter(torch.Tensor(size_out,))

        # initialize weights and biases
        nn.init.normal_(self.w_mu, mean=0.0, std=0.05)
        nn.init.uniform_(self.w_sigma, a=-12.0, b=-2.2)
        nn.init.normal_(self.b_mu, mean=0.0, std=0.00005)
        nn.init.uniform_(self.b_sigma, a=-12.0, b=-10.0)

    def forward(self, mu_in, Sigma_in):
        # Mean
        mu_out = torch.matmul(mu_in, self.w_mu) + self.b_mu
        
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
    """ReLU"""

    def __init__(self):
        super(RVRelu, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_out = torch.relu(mu_in)

        gradi = torch.autograd.grad(mu_out, mu_in, grad_outputs=torch.ones_like(mu_out), create_graph=True)[0]

        Sigma_out = activation_Sigma(gradi, Sigma_in)

        return mu_out, Sigma_out

class RVSoftmax(nn.Module):
    """Random Variable Softmax"""

    def __init__(self):
        super(RVSoftmax, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_out = torch.softmax(mu_in, dim=1) # shape: [batch_size, output_size]

        pp1 = mu_out.unsqueeze(dim=2)  # shape: [batch_size, output_size, 1]
        pp2 = mu_out.unsqueeze(dim=1)  # shape: [batch_size, 1, output_size]
        ppT = torch.matmul(pp1, pp2)   # shape: [batch_size, output_size, output_size]
        p_diag = torch.diag_embed(mu_out)  # shape: [batch_size, output_size]

        grad = p_diag - ppT
        Sigma_out = torch.matmul(grad, torch.matmul(Sigma_in, grad.permute(0, 2, 1)))

        return mu_out, Sigma_out

class exVDPMLP(nn.Module):
    """Stack of Linear layers with a KL regularization loss."""

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(exVDPMLP, self).__init__()
        self.linear_1 = Constant2RVLinearlayer(input_dim, hidden_dim)
        self.myrelu_1 = RVRelu()
        self.linear_2 = RV2RVLinearlayer(hidden_dim, output_dim)
        self.softmax  = RVSoftmax()

    def forward(self, inputs):
        m, s, kl_loss_1 = self.linear_1(inputs)
        m, s = self.myrelu_1(m, s)
        m, s, kl_loss_2  = self.linear_2(m, s)
        outputs, Sigma = self.softmax(m, s)
        return outputs, Sigma, kl_loss_1 + kl_loss_2

def train_model(rank, mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH):

    def update_progress(progress):
        """Helper function to display progress bar during training."""
        barLength = 10 # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rTraining Progress: [{0}] {1}% {2}".format( "#" * block + "-" * (barLength-block), round(progress*100, 2), status)
        print(text, end='', flush=True)

    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)

    train_acc = np.zeros(epochs) 
    valid_acc = np.zeros(epochs)
    train_err = np.zeros(epochs)
    valid_error = np.zeros(epochs)
    start = timeit.default_timer()

    for epoch in range(epochs):
        print('Epoch: ', epoch+1, '/' , epochs)

        acc1 = 0 
        acc_valid1 = 0 
        err1 = 0
        err_valid1 = 0
        tr_no_steps = 0          
        #Training
        for step, (x, y) in enumerate(train_loader):
            update_progress(step / int(len(train_loader)) ) 
            x = x.reshape(-1, 28*28).to(rank)
            y = y.to(rank)

             # Convert labels to one hot encoding
            labels_one_hot = F.one_hot(y, num_classes=10)

            optimizer.zero_grad()

            logits, sigma, kl_loss = mlp_model(x)

            log_loss = nll_gaussian(labels_one_hot, logits, sigma.clamp(min=-1e+10, max=1e+10), len(train_loader.dataset.classes), batch_size)
            total_loss = log_loss + kl_factor * kl_loss

            total_loss.backward()
            optimizer.step()

            err1 += total_loss.item()
            corr = torch.eq(torch.argmax(logits, axis=1), torch.argmax(labels_one_hot,axis=1))
            accuracy = torch.mean(corr.float())
            acc1 += accuracy.item()
            tr_no_steps += 1
        
        train_acc[epoch] = acc1/tr_no_steps
        train_err[epoch] = err1/tr_no_steps

        print('Training Acc  ', train_acc[epoch])
        print('Training error  ', train_err[epoch])

    stop = timeit.default_timer()
    print('Total Training Time: ', stop - start)
    print('Training Acc  ', np.mean(train_acc))          
    print('Training error  ', np.mean(train_err))       

    torch.save(mlp_model.state_dict(), PATH + 'DP_MLP_model.pth')

    if (epochs > 1):
        fig = plt.figure(figsize=(15,7))
        plt.plot(train_acc, 'b', label='Training acc')
        plt.ylim(0, 1.1)
        plt.title("Density Propagation MLP on MNIST Data")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')
        plt.savefig(PATH + 'DP_MLP_on_MNIST_Data_acc.png')
        plt.close(fig)

        fig = plt.figure(figsize=(15,7))
        plt.plot(train_err, 'b', label='Training error')          
        plt.title("Density Propagation MLP on MNIST Data")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.legend(loc='lower right')
        plt.savefig(PATH)

def test_mlp_model(mlp_model, x_test, y_test, val_dataset, batch_size, input_dim, output_size, PATH, epochs, lr, gaussain_noise_std=0.0, Random_noise=False):
    device = torch.device("cpu")
    test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
    mlp_model.load_state_dict(torch.load(PATH + 'DP_MLP_model'))
    test_no_steps = 0
    err_test = 0
    acc_test = 0
    true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim])
    true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
    logits_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
    sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size, output_size])
    
    for step, (x, y) in enumerate(val_dataset):
        update_progress(step / int(x_test.shape[0] / (batch_size)))
        true_x[test_no_steps, :, :] = x.numpy()
        true_y[test_no_steps, :, :] = y.numpy()
        if Random_noise:
            noise = torch.normal(mean=0.0, std=gaussain_noise_std, size=(batch_size, input_dim))
            x = x + noise
        logits, sigma = mlp_model(x)
        logits_[test_no_steps,:,:] = logits.detach().numpy()
        sigma_[test_no_steps, :, :, :] = sigma.detach().numpy()
        tloss = nll_gaussian(y.numpy(), logits.detach().numpy(), 
                             np.clip(sigma.detach().numpy(), -1e+10, 1e+10), output_size, batch_size)
        err_test += tloss
        
        corr = (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).float()
        accuracy = torch.mean(corr)
        acc_test += accuracy
        
        if step % 500 == 0:
            print("Step:", step, "Loss:", float(tloss))
            print("Total running accuracy so far: %.3f" % accuracy)
        
        test_no_steps += 1
    
    test_acc = acc_test / test_no_steps
    test_error = err_test / test_no_steps
    print('Test accuracy : ', test_acc.numpy())
    print('Test error : ', test_error.numpy())
    
    with open(PATH + test_path + 'uncertainty_info.pkl', 'wb') as pf:
        pickle.dump([logits_, sigma_, true_x, true_y, test_acc.numpy(), test_error.numpy()], pf)
    
    with open(PATH + test_path + 'Related_hyperparameters.txt', 'w') as textfile:
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr))          
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc.numpy()))
        textfile.write("\n Averaged Test error : "+ str(test_error.numpy()))            
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))              
        textfile.write("\n---------------------------------")

def main(rank, world_size, input_dim=784, hidden_dim=500, output_dim=10, batch_size=256, epochs=1, lr=0.001, kl_factor = 0.01,
                  random_noise=True, gaussian_noise_std=10000, training=False, PATH="dump/"):
    # Set up distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='/data/', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    test_dataset = torchvision.datasets.MNIST(root='/data/', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            sampler=test_sampler)
    
    # Cutom Training Loop with Graph
    mlp_model = DDP(exVDPMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim))

    train_model(rank, mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH)

    # Clean up
    dist.destroy_process_group()

def parse_arguments():
    parser = argparse.ArgumentParser(description='MNIST Density Propagation')
    parser.add_argument('--input_dim', type=int, default=784,
                        help='Dimension of input data')
    parser.add_argument('--hidden_dim', type=int, default=500,
                        help='Dimension of hidden layer')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Dimension of output layer')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--kl_factor', type=float, default=0.01,
                        help='Factor to weigh the KL divergence loss term')
    parser.add_argument('--random_noise', type=bool, default=True,
                        help='Whether to add random noise to input data')
    parser.add_argument('--gaussian_noise_std', type=float, default=10000,
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--training', type=bool, default=False,
                        help='Whether to train the model or not')
    parser.add_argument('--PATH', type=str, default='dump/',
                        help='Path to save the model and related hyperparameters')
    parser.add_argument('--gpus', type=str, default='0 1',
                        help='GPUs to Use')
    return parser.parse_args()

# Run the function
if __name__ == '__main__':
    # Unload argparser
    args = parse_arguments()
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    kl_factor = args.kl_factor
    random_noise = args.random_noise
    gaussian_noise_std = args.gaussian_noise_std
    training = args.training
    PATH = args.PATH
    gpus = args.gpus

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    main(rank, world_size, input_dim, hidden_dim, output_dim, batch_size, epochs, lr, kl_factor,
                  random_noise, gaussian_noise_std, training, PATH)