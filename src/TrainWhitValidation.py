import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import ChessJSONLDataset

from Model import ChessNet
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

#Training params
num_epochs = 200
num_blocks = 15
num_filters = 128
ccrl_dir = './Training_full'
ccrl_dir_val = './Validation_full'
cuda=True
model_path = 'Newest_model_3.pt'  # Path to load pre-trained model (if exists)

def train():
    print("data loading..")
    train_ds = ChessJSONLDataset(ccrl_dir)
    print("Val data loading..")
    val_ds = ChessJSONLDataset(ccrl_dir_val)
    print("setings..")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_ds, batch_size=1096, shuffle=False, num_workers=11)
    
    if cuda:
        alphaZeroNet = ChessNet(num_blocks, num_filters).cuda()
    else:
        alphaZeroNet = ChessNet(num_blocks, num_filters)

    # Load pre-trained weights if the file exists
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')
        alphaZeroNet.load_state_dict(checkpoint)
        print(f"Loaded pre-trained model weights from {model_path}")
    else:
        print("No pre-trained model found. Starting from scratch.")
    optimizer = optim.Adam(alphaZeroNet.parameters(),weight_decay=1e-3,lr=0.001)
    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, verbose=True)

    print('Starting training')

    train_losses = []
    val_losses = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Separate plots for training and validation losses
    
    for epoch in range(num_epochs):
        alphaZeroNet.train()
        epoch_train_losses = []

        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()

            if cuda:
                position = data['position'].cuda()
                valueTarget = data['value'].cuda()
            else:
                position = data['position']
                valueTarget = data['value']

            valueLoss = alphaZeroNet(position, valueTarget=valueTarget)
            loss = valueLoss

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(float(valueLoss))
            if iter_num % 500 == 0:
                message = 'Epoch {} | Step {} / {} | Training loss {:0.5f} | Lr: {}'.format(
                    epoch, iter_num, len(train_loader), float(sum(epoch_train_losses) / len(epoch_train_losses)),optimizer.param_groups[0]['lr'])
                print(message)
            
            


        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        
        
        # Validation Step
        print(f"Starting validation on epoch {epoch}")
        alphaZeroNet.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for data in val_loader:
                if cuda:
                    position = data['position'].cuda()
                    valueTarget = data['value'].cuda()
                else:
                    position = data['position']
                    valueTarget = data['value']

                # Get the model's prediction
                value = alphaZeroNet(position)  # Do not pass valueTarget during eval

                # Calculate loss explicitly in validation
                valueLoss = (value - valueTarget).abs().mean()
                epoch_val_losses.append(valueLoss.item())
                

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)

        print(f'End of Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Lr: {optimizer.param_groups[0]['lr']}')
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Update Plots
        ax1.clear()
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.set_title('Training Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.clear()
        ax2.plot(val_losses, label='Validation Loss', color='red')
        ax2.set_title('Validation Loss Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.draw()
        plt.pause(0.01)
        if epoch % 2 == 0:
            networkFileName = 'AlphaZeroNet_{}x{}_epoch{}_4.pt'.format(num_blocks, num_filters, epoch)
            torch.save(alphaZeroNet.state_dict(), networkFileName)
            torch.save(alphaZeroNet.state_dict(), "Newest_model_4.pt")
            print('Saved model to {}'.format(networkFileName))



if __name__ == '__main__':
    train()