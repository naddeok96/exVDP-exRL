'''
This script will train a model and save it
'''
# Imports
import torch
import pickle
from data_setup import Data
import torch.nn.functional as F
from torchsummary import summary
from torch_exVDP_MLP import exVDPMLP, nll_gaussian
from torch_MLP import MLP


# Hyperparameters
#------------------------------#
# Machine parameters
seed       = 3
gpu        = True
gpu_number = "2"

# Training parameters
n_epochs          = 5
batch_size        = 124
learning_rate     = 0.25

# Model parameters
pretrained_weights_filename = None
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
            root = data_root)
print(set_name, "Data Loaded")

# Load Network
# net = MLP(input_dim=784, hidden_dim=256, output_dim=10)
net = exVDPMLP(input_dim=784, hidden_dim=16, output_dim=10)

if gpu:
    net = net.cuda() 

if pretrained_weights_filename:
    state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)

net.eval()
print("Network Loaded")

# Training procedure
def train(net, data, gpu, n_epochs, batch_size, attack_type, epsilon):
    #Get training data
    train_loader = data.get_train_loader(batch_size)

    #Create optimizer and loss functions
    optimizer = torch.optim.Adam(net.parameters(),  lr = learning_rate)

    #Loop for n_epochs
    for epoch in range(n_epochs):      
        for inputs, labels in train_loader:
            # Flatten input
            inputs = inputs.reshape(-1, 28*28, 1)

            # Convert labels to one hot encoding
            labels_one_hot = F.one_hot(labels, num_classes=10)

            # Push to gpu
            if gpu:
                inputs, labels, labels_one_hot = inputs.cuda(), labels.cuda(), labels_one_hot.cuda()

            # Adversarial Training
            # if attack_type:
            #     # Initalize Attacker with current model parameters
            #     attacker = Attacker(net = net, data = data, gpu = gpu)

            #     # Generate attacks and replace them with orginal images
            #     inputs = attacker.get_attack_accuracy(attack = attack_type,
            #                                             attack_images = inputs,
            #                                             attack_labels = labels,
            #                                             epsilons = [epsilon],
            #                                             return_attacks_only = True,
            #                                             prog_bar = False)

            #Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            outputs, sigma, kl_loss = net(inputs)
            # outputs = net(inputs)
            exit()

            # Get loss
            log_loss = nll_gaussian(labels_one_hot, outputs, sigma.clamp(min=-1e+10, max=1e+10), len(train_loader.dataset.classes), batch_size)
            total_loss = log_loss + kl_factor * kl_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
        
        if epoch % 2 == 0:
            accuracy = test(net, data, gpu)
            print("Epoch: ", epoch + 1 , "\tLoss: ", loss.item(), "\tAcc:", accuracy)    

    return net 

# Evaluation procedure   
def test(net, data, gpu):
    # Initialize
    total_tested = 0
    correct = 0

    # Test images in test loader
    for inputs, labels in data.test_loader:
        # Flatten input
        inputs = inputs.reshape(-1, 28*28)

        # Convert labels to one hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=10)

        # Push to gpu
        if gpu:
            inputs, labels, labels_one_hot = inputs.cuda(), labels.cuda(), labels_one_hot.cuda()

        #Forward pass
        outputs, sigma, kl_loss = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update running sum
        total_tested += labels.size(0)
        correct += torch.eq(torch.argmax(logits, axis=1), torch.argmax(labels_one_hot,axis=1))
    
    # Calculate accuracy
    accuracy = (correct/total_tested)
    return accuracy

# Fit Model
print("Training")
net = train(net, data, gpu, n_epochs, batch_size, attack_type, epsilon)

# Calculate accuracy on test set
print("Testing")
accuracy = test(net, data, gpu)
print("Accuarcy: ", accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = "exVDP" + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(net.state_dict(), "models/pretrained/" + set_name + "/" + filename)