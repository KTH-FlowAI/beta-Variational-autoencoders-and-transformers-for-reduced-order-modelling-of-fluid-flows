"""
functions for training

@yuningw
"""


import torch 
import numpy as np 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False





def fit(device,
        model,
        dl,
        loss_fn,
        Epoch,
        optimizer:torch.optim.Optimizer, 
        val_dl        = None,
        scheduler:list= None,
        if_early_stop = True,patience = 10,
        ):
    
    """
    A function for training loop

    Args: 
        device      :       the device for training, which should match the model
        
        model       :       The model to be trained
        
        dl          :       A dataloader for training
        
        loss_fn     :       Loss function
        
        Epochs      :       Number of epochs 
        
        optimizer   :       The optimizer object
        
        val_dl      :       The data for validation
        
        scheduler   :       A list of traning scheduler
        

    Returns:
        history: A dict contains training loss and validation loss (if have)

    """

    from tqdm import tqdm
    
    history = {}
    history["train_loss"] = []
    
    if val_dl:
        history["val_loss"] = []
    
    model.to(device)
    print(f"INFO: The model is assigned to device: {device} ")

    if scheduler is not None:
        print(f"INFO: The following schedulers are going to be used:")
        for sch in scheduler:
            print(f"{sch.__class__}")

    print(f"INFO: Training start")

    if if_early_stop: 
        early_stopper = EarlyStopper(patience=patience,min_delta=0)
        print("INFO: Early-Stopper prepared")

    for epoch in range(Epoch):
        #####
        #Training step
        #####
        model.train()
        loss_val = 0; num_batch = 0
        for batch in tqdm(dl):
            x, y = batch
            x = x.to(device).float(); y =y.to(device).float()
            optimizer.zero_grad()
            
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()

            

            loss_val += loss.item()/x.shape[0]
            num_batch += 1

        history["train_loss"].append(loss_val/num_batch)

        if scheduler is not None:
            lr_now = 0 
            for sch in scheduler:
                sch.step()
                lr_now = sch.get_last_lr()
            print(f"INFO: Scheduler updated, LR = {lr_now} ")

        if val_dl:
        #####
        #Valdation step
        #####
            loss_val = 0 ; num_batch = 0 
            model.eval()
            for batch in (val_dl):
                x, y = batch
                x = x.to(device).float(); y =y.to(device).float()
                pred = model(x)
                loss = loss_fn(pred,y)
            
                loss_val += loss.item()/x.shape[0]
                num_batch += 1

            history["val_loss"].append(loss_val/num_batch)
        
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        print(
                f"At Epoch    = {epoch},\n"+\
                f"Train_loss  = {train_loss}\n"+\
                f"Val_loss    = {val_loss}\n"          
            )
        
        if if_early_stop:
            if early_stopper.early_stop(loss_val/num_batch):
                print("Early-stopp Triggered, Going to stop the training")
                break
    return history





