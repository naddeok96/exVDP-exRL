# Imports
import argparse
import os
import timeit
import numpy as np

import torch 
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from torch_exVDP_MLP import exVDPMLP, nll_gaussian

def train_model(mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH, device):

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

    train_acc, valid_acc, train_err, valid_error = np.zeros((4, epochs))
    start = timeit.default_timer()
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1, '/' , epochs)
             
        # Training
        acc1 = acc_valid1 = err1 = err_valid1 = tr_no_steps = 0  
        for step, (x, y) in enumerate(train_loader):
            # Update progress bar
            update_progress(step / int(len(train_loader)))

            # Flatten input and move to device
            x = x.reshape(-1, 28*28, 1).to(device)

            # Move labels to device and convert to one hot encoding
            labels_one_hot = F.one_hot(y, num_classes=10).to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass through the model
            outputs, sigma, kl_loss = mlp_model(x)

            # Compute negative log-likelihood of Gaussian
            log_loss = nll_gaussian(labels_one_hot, outputs, sigma.clamp(min=-1e+6, max=1e+6), len(train_loader.dataset.classes))

            # Compute total loss, including KL divergence
            total_loss = log_loss + kl_factor * kl_loss

            # Compute gradients and perform optimizer step
            total_loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            err1 += total_loss.item()
            corr = torch.eq(torch.argmax(outputs, axis=1), torch.argmax(labels_one_hot,axis=1))
            accuracy = torch.mean(corr.float())
            acc1 += accuracy.item()

            # Increment training step counter
            tr_no_steps += 1
        
        train_acc[epoch] = acc1/tr_no_steps
        train_err[epoch] = err1/tr_no_steps

        print('Training Acc  ', train_acc[epoch])
        print('Training error  ', train_err[epoch])

    stop = timeit.default_timer()
    print('Total Training Time: ', stop - start)
    print('Training Acc  ', np.mean(train_acc))          
    print('Training error  ', np.mean(train_err))       

    torch.save(mlp_model.state_dict(), PATH + 'VDP_MLP_model.pth')

def test_mlp_model(mlp_model, x_test, y_test, val_dataset, batch_size, input_dim, output_size, PATH, epochs, lr, gaussain_noise_std=0.0, Random_noise=False):
    device = torch.device("cpu")
    test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
    mlp_model.load_state_dict(torch.load(PATH + 'DP_MLP_model'))
    test_no_steps = 0
    err_test = 0
    acc_test = 0
    true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim])
    true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
    outputs_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
    sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size, output_size])
    
    for step, (x, y) in enumerate(val_dataset):
        update_progress(step / int(x_test.shape[0] / (batch_size)))
        true_x[test_no_steps, :, :] = x.numpy()
        true_y[test_no_steps, :, :] = y.numpy()
        if Random_noise:
            noise = torch.normal(mean=0.0, std=gaussain_noise_std, size=(batch_size, input_dim))
            x = x + noise
        outputs, sigma = mlp_model(x)
        outputs_[test_no_steps,:,:] = outputs.detach().numpy()
        sigma_[test_no_steps, :, :, :] = sigma.detach().numpy()
        tloss = nll_gaussian(y.numpy(), outputs.detach().numpy(), 
                             np.clip(sigma.detach().numpy(), -1e+10, 1e+10), output_size)
        err_test += tloss
        
        corr = (torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).float()
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
        pickle.dump([outputs_, sigma_, true_x, true_y, test_acc.numpy(), test_error.numpy()], pf)
    
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

def main_function(input_dim=784, hidden_dim=124, output_dim=10, batch_size=124, epochs=5, lr=0.001, kl_factor = 0.001,
                  random_noise=True, gaussian_noise_std=10000, training=False, PATH="dump/",  gpu="4"):
    
    # Set GPU device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='/data/', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='/data/', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    # Load model
    mlp_model = exVDPMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Move model to device
    mlp_model.to(device)

    # Fit model
    train_model(mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH, device)
    
if __name__ == '__main__':

    main_function(gpu="4") 
