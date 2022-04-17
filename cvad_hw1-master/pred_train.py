import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor

import os,copy,time
from tqdm import tqdm
from ipdq import set_trace


def validate(model, dataloader):
   
    
    pass
   
    """Validate model performance on the validation dataset"""
    
    


def train(model, dataloader):
    
    
    
    # Your code here
    pass


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    print("*******Plotting the losses*****")
    my_custom_length = len(train_loss)
    my_custom_array=[]
    for g in range(0,my_custom_length):
       my_custom_array.append(g)
       
    
    my_custom_array=np.array(my_custom_array) #Converting to numpy array
    
    plt.title(" - The Epoch Values vs The Loss Values - ")
    plt.xlabel("The Epoch Values")
    plt.ylabel("The Loss Values")
   
    
    
    gr_col = "green"
    blue_col = "blue"
    train_str= "train loss values"
    validation_str = "validation loss values"
    circle_str="o"
    location_str="upper right"
    
    #Plotting by using plt.plot function of the matplotlib library of python.
    plt.plot(train_loss, my_custom_array, color = gr_col, marker = circle_str,markersize=13)
    plt.plot(val_loss, my_custom_array, color = blue_col, marker = circle_str,markersize=13)
    
    
    plt.legend([train_str, validation_str], loc = location_str)
   
    plt.show()
    
    print("*******The End of the plotting*****")
   


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = None
    val_root = None
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
