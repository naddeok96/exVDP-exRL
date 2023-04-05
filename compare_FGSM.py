# Imports
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchattacks
from tqdm import tqdm

from data_setup import Data
from torch_MLP import MLP
from torch_exVDP_MLP import exVDPMLP, nll_gaussian

def add_gaussian_noise(image):
    # Collect device
    device = image.device

    # Compute the variance of the image
    signal_var = torch.mean(torch.square(image))

    # Compute the variance of the noise
    snr = 0.2
    noise_var = signal_var / snr

    # Generate a noise image
    noise_image = torch.normal(mean=0, std=torch.sqrt(noise_var), size=image.shape)

    # Add the noise image to the original image
    noisy_image = image + noise_image.to(device)

    return noisy_image

# Push to GPU if necessary
gpu_number = "4"
if gpu_number:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Define the device to run the attack on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architectures
net_VDP = exVDPMLP(input_dim=784, hidden_dim=124, output_dim=10)
net_MLP = MLP(input_dim=784, hidden_dim=124, output_dim=10)

# Load the model weights
net_VDP.load_state_dict(torch.load('model_weights/VDP_MLP_model.pth', map_location=torch.device('cpu')))
net_MLP.load_state_dict(torch.load('model_weights/MLP_w_acc_97.pt', map_location=torch.device('cpu')))

# Set the models to evaluation mode
net_VDP.eval().to(device)
net_MLP.eval().to(device)

# Define the transform to apply to the input image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST test dataset
data = Data(set_name = "MNIST",
                        gpu = True if gpu_number else False,
                        test_batch_size = 1)

found = False
for i, (image, label) in enumerate(tqdm(data.test_loader, desc="Tried so far")):
    # Prepare the input image for the models
    image = image.to(device)

    # Pass the input image through both models
    output_VDP, sigma, kl_loss = net_VDP(image.view(-1, 784, 1))
    output_MLP = net_MLP(image.view(-1, 784))

    # Get the predicted labels and probabilities for each model
    pred_VDP = output_VDP.argmax(dim=1).item()
    prob_VDP = output_VDP[0, pred_VDP].item()

    pred_MLP = output_MLP.argmax(dim=1).item()
    prob_MLP = output_MLP[0, pred_MLP].item()

    # Check if the criteria are met
    if pred_VDP != label or pred_MLP != label:
        continue

    # Generate noisy image
    adv_image = add_gaussian_noise(image)

    # Pass the attacked image through both models
    adv_output_VDP, adv_sigma, adv_kl_loss = net_VDP(adv_image.view(-1, 784, 1))
    adv_output_MLP = net_MLP(adv_image.view(-1, 784))

    # Get the predicted labels and probabilities for each model
    adv_pred_VDP = adv_output_VDP.argmax(dim=1).item()
    adv_prob_VDP = adv_output_VDP[0, adv_pred_VDP].item()

    adv_pred_MLP = adv_output_MLP.argmax(dim=1).item()
    adv_prob_MLP = adv_output_MLP[0, adv_pred_MLP].item()

    if adv_pred_VDP == label and adv_pred_MLP != label:
        print("Found")
        break

# Display the original image and its label
plt.subplot(2, 2, 1)
plt.title(f"Original image (label: {label.item()})\nVDP prediction: {pred_VDP} (prob: {prob_VDP:.2f})\nMLP prediction: {pred_MLP} (prob: {prob_MLP:.2f})")
plt.imshow(image.squeeze().cpu().detach().numpy(), cmap='gray')
plt.axis('off')


# Display the attacked image and its label
plt.subplot(2, 2, 2)
plt.title(f"Attacked image (label: {label.item()})\nVDP prediction: {adv_pred_VDP} (prob: {adv_prob_VDP:.2f})\nMLP prediction: {adv_pred_MLP} (prob: {adv_prob_MLP:.2f})")
plt.imshow(adv_image.squeeze().cpu().detach().numpy(), cmap='gray')
plt.axis('off')

# Plot the heatmap with row and column labels
ax1 = plt.subplot(2, 2, 3)
plt.imshow(sigma.view(10,10).cpu().detach().numpy(), cmap='plasma')
plt.colorbar()
plt.xticks(np.arange(0.5, 10.5), range(10))
plt.yticks(np.arange(0.5, 10.5), range(10))
ax1.tick_params(which='both', bottom=False, left=False, labelbottom=True, labelleft=True)
for label in ax1.get_xticklabels():
    label.set_horizontalalignment('left')
for label in ax1.get_yticklabels():
    label.set_verticalalignment('bottom')
plt.grid(True, color='black', linestyle='-', linewidth=1)

# Move the x-axis ticks to the top
ax1.xaxis.set_ticks_position('top')

# Plot the heatmap with row and column labels
ax2 = plt.subplot(2, 2, 4)
plt.imshow(adv_sigma.view(10,10).cpu().detach().numpy(), cmap='plasma')
plt.colorbar()
plt.xticks(np.arange(0.5, 10.5), range(10))
plt.yticks(np.arange(0.5, 10.5), range(10))
ax2.tick_params(which='both', bottom=False, left=False, labelbottom=True, labelleft=True)
for label in ax2.get_xticklabels():
    label.set_horizontalalignment('left')
for label in ax2.get_yticklabels():
    label.set_verticalalignment('bottom')
plt.grid(True, color='black', linestyle='-', linewidth=1)

# Move the x-axis ticks to the top
ax2.xaxis.set_ticks_position('top')

plt.tight_layout()

# save the figure as a file
plt.savefig("figures/VDP_MLP_comparison.png")
