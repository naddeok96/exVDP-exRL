
def train_model(mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH):

    device = torch.device("cpu") 

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
            x = x.reshape(-1, 28*28).to(device)
            y = y.to(device)

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

def main_function(input_dim=784, hidden_dim=500, output_dim=10, batch_size=256, epochs=1, lr=0.001, kl_factor = 0.01,
                  random_noise=True, gaussian_noise_std=10000, training=False, PATH="dump/"):
    
    
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
    
    # Cutom Training Loop with Graph
    mlp_model = exVDPMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    train_model(mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH)

if __name__ == '__main__':
    main_function() 

#     Epoch:  1 / 1
# Training Progress: [##########] 99.57% Traceback (most recent call last):
#   File "torchExVDP.py", line 368, in <module>
#     main_function() 
#   File "torchExVDP.py", line 365, in main_function
#     train_model(mlp_model, train_loader, epochs, batch_size, lr, kl_factor, PATH)
#   File "torchExVDP.py", line 237, in train_model
#     log_loss = nll_gaussian(labels_one_hot, logits, sigma.clamp(min=-1e+10, max=1e+10), len(train_loader.dataset.classes), batch_size)
#   File "torchExVDP.py", line 43, in nll_gaussian
#     y_pred_sd_ns = y_pred_sd + NS
# RuntimeError: The size of tensor a (96) must match the size of tensor b (256) at non-singleton dimension 0
