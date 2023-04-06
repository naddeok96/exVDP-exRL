'''
This script will train a model and save it
'''
# Imports
import torch
import time
from data_setup import Data
import torch.nn.functional as F
from tqdm import tqdm

from torch_MLP import MLP
from torch_exVDP_MLP import exVDPMLP, nll_gaussian

# Training procedure
def train(net, data, gpu, n_epochs, learning_rate, batch_size, attack_type, epsilon, kl_factor, test_freq):
    #Get training data
    train_loader = data.get_train_loader(batch_size)

    #Create optimizer and loss functions
    optimizer = torch.optim.Adam(net.parameters(),  lr = learning_rate)


    #Loop for n_epochs
    for epoch in tqdm(range(n_epochs), desc='Epochs'):   
        for inputs, labels in tqdm(train_loader, desc='Batches'):
            # Flatten input
            inputs = inputs.view(-1, 28*28, 1)

            # Convert labels to one hot encoding
            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            # Push to gpu
            if gpu:
                inputs, labels, labels_one_hot = inputs.cuda(), labels.cuda(), labels_one_hot.cuda()

            #Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            if kl_factor:
                outputs, sigma, kl_loss = net(inputs)
            else:
                outputs = net(inputs.squeeze())

            # Get loss
            if kl_factor:
                log_loss = nll_gaussian(y_test=labels_one_hot, y_pred_mean=outputs, y_pred_sd=sigma.clamp(min=-1e+10, max=1e+10), num_labels=len(train_loader.dataset.classes))
                total_loss = log_loss + kl_factor * kl_loss
            else:
                total_loss = F.mse_loss(labels_one_hot, outputs)

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
        
        # if epoch % test_freq == 0:
        #     accuracy = test(net, data, gpu, kl_factor)
        #     print("Epoch: ", epoch + 1 , "\tLoss: ", total_loss.item(), "\tAcc:", accuracy)    

    return net 

# Evaluation procedure   
def test(net, data, gpu, kl_factor):
    # Initialize
    total_tested = 0
    correct = 0

    # Test images in test loader
    for inputs, labels in tqdm(data.test_loader, desc='Testing'):
        # Flatten input
        inputs = inputs.reshape(-1, 28*28, 1)

        # Convert labels to one hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=10).float()

        # Push to gpu
        if gpu:
            inputs, labels, labels_one_hot = inputs.cuda(), labels.cuda(), labels_one_hot.cuda()

        #Forward pass
        if kl_factor:
            outputs, _, _ = net(inputs)
        else:
            outputs = net(inputs.squeeze())

        # Update running sum
        total_tested += labels.size(0)
        correct += torch.sum(torch.eq(torch.argmax(outputs, axis=1), torch.argmax(labels_one_hot,axis=1))).item()
    
    # Calculate accuracy
    accuracy = (correct/total_tested)
    return accuracy

def main():
    # Hyperparameters
    #------------------------------#
    # Machine parameters
    seed       = 3
    gpu        = True
    gpu_number = "2"

    # Training parameters
    n_epochs        = 5
    batch_size      = 124
    learning_rate   = 0.001
    kl_factor       = 0.001
    test_freq       = 10

    # Model parameters
    hidden_dim = 124
    pretrained_weights_filename = None # "model_weights/VDP_MLP_model.pth"
    save_model                  = True

    # Data parameters
    set_name  = "MNIST"
    data_root = '../data/' + set_name

    # Adversarial Training parameters
    attack_type = None
    epsilon     = 0.15
    #------------------------------#

    # Display
    print("Epochs: ", n_epochs)
    print("Adversarial Training Type: ", attack_type)
    if attack_type is not None:
        print("Epsilon: ", epsilon)

    # Push to GPU if necessary
    if gpu:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

    # Declare seed and initalize network
    torch.manual_seed(seed)

    # Load data
    data = Data(gpu = gpu, 
                set_name = set_name,
                root = data_root,
                test_batch_size = batch_size)
    print(set_name, "Data Loaded")

    # Initalize Network
    if kl_factor:
        net = exVDPMLP(input_dim=784, hidden_dim=hidden_dim, output_dim=10)
    else:
        net = MLP(input_dim=784, hidden_dim=hidden_dim, output_dim=10)
    
    net.eval()

    # Load weights
    if pretrained_weights_filename:
        state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)

    # Push to GPU
    if gpu:
        net = net.cuda() 

    print("Network Loaded")

    # Fit Model
    if n_epochs > 0:
        start_time = time.time()
        net = train(net, data, gpu, n_epochs, learning_rate, batch_size, attack_type, epsilon, kl_factor, test_freq)
        print("Training took ", time.time() - start_time, "s")

    # Calculate accuracy on test set
    accuracy = test(net, data, gpu, kl_factor)
    print("Accuarcy: ", accuracy)

    # Save Model
    if save_model:
        # Define File Names
        if kl_factor:
            filename  = "model_weights/exVDP" + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
        else:
            filename  = "model_weights/MLP" + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
        
        # Save Models
        torch.save(net.state_dict(), filename)

if __name__ == "__main__":
    main()