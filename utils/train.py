"""
functions for training

@yuningw
"""

import torch
import torch.nn as nn
import time
import numpy as np


##############################################
# Functions used for training beta-VAE
################################################
"""
def train_vae(
            model,
            optimizer,
            epochs,
            betaSch,
            
            ):
    
    bestloss = 1e6
    loss = 1e6
    converging = False

    for epoch in range(1, epochs + 1):

        if loss < 0.5 and not converging:
            print('Updating Wdecay, epoch: ', epoch)
            converging = True
            for param_group in optimizer.param_groups:
                optimizer.param_groups[0]["weight_decay"] = encWdecay
                optimizer.param_groups[1]["weight_decay"] = decWdecay

        model.train()
        beta = betaSch.getBeta(epoch, prints=False)
        loss, MSE, KLD, elapsed, collapsed = lib_training.train_epoch(model, dataset_train, optimizer, beta, device)
        model.eval()
        loss_test, MSE_test, KLD_test, elapsed_test = lib_training.test_epoch(model, dataset_test, beta, device)

        scheduler.step()

        lib_training.printProgress(epoch, epochs, loss, loss_test, MSE, KLD, elapsed, elapsed_test, collapsed)

        logger.add_scalar('General loss/Total', loss, epoch)
        logger.add_scalar('General loss/MSE', MSE, epoch)
        logger.add_scalar('General loss/KLD', KLD, epoch)
        logger.add_scalar('General loss/Total_test', loss_test, epoch)
        logger.add_scalar('General loss/MSE_test', MSE_test, epoch)
        logger.add_scalar('General loss/KLD_test', KLD_test, epoch)
        logger.add_scalar('Optimizer/LR', scheduler.get_last_lr()[0], epoch)

        if (loss_test < bestloss and epoch > 100):
            bestloss = loss_test
            checkpoint = {'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict()}
            ckp_file = f'./03_checkpoints/{modelname}_epoch_bestTest.pth.tar'
            lib_model.save_checkpoint(state=checkpoint, path_name=ckp_file)
            print(f'## Checkpoint. Epoch: {epoch}, test loss: {loss_test}, saving checkpoint {ckp_file}')

    checkpoint = {'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict()}
    ckp_file = f'./03_checkpoints/{modelname}_epoch_final.pth.tar'
    lib_model.save_checkpoint(state=checkpoint, path_name=ckp_file)
    print(f'Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving checkpoint {ckp_file}')



"""

class betaScheduler:
    """Schedule beta, linear growth to max value"""

    def __init__(self, startvalue, endvalue, warmup):
        self.startvalue = startvalue
        self.endvalue = endvalue
        self.warmup = warmup

    def getBeta(self, epoch, prints=False):

        if epoch < self.warmup:
            beta = self.startvalue + (self.endvalue - self.startvalue) * epoch / self.warmup
            if prints:
                print(beta)
            return beta
        else:
            return self.endvalue


def loss_function(reconstruction, data, mean, logvariance, beta):
    MSELoss = nn.MSELoss(reduction='mean').cuda()
    MSE = MSELoss(reconstruction, data)

    KLD = -0.5 * torch.mean(1 + logvariance - mean.pow(2) - logvariance.exp())

    loss = MSE + KLD * beta

    return loss, MSE, KLD

def train_epoch(model, data, optimizer, beta, device):
    start_epoch_time = time.time()

    loss_batch = [] # Store batch loss
    MSE_batch = [] # Store batch MSE
    KLD_batch = [] # Store batch KLD
    logVar_batch = [] # Store batch logVar to count collapsed modes

    for batch in data:
        #batch_noise = gaussian_noise(batch, 0.2, device)
        if not batch.is_cuda:
            batch = batch.to(device, non_blocking=True)
            #batch_noise = batch_noise.to(device, non_blocking=True)

        rec, mean, logvariance = model(batch)# + batch_noise)
        loss, MSE, KLD = loss_function(rec, batch, mean, logvariance, beta)

        loss_batch.append(loss.item())
        MSE_batch.append(MSE.item())
        KLD_batch.append(KLD.item())
        logVar_batch.append(np.exp(0.5* np.mean(logvariance.detach().cpu().numpy(), 0)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return sum(loss_batch) / len(loss_batch),\
            sum(MSE_batch) / len(MSE_batch),\
            sum(KLD_batch) / len(KLD_batch),\
            time.time() - start_epoch_time, \
            (np.mean(np.stack(logVar_batch, axis=0), 0) < 0.1).sum() # count collapsed modes

def test_epoch(model, data, beta, device):
    start_epoch_time = time.time()

    with torch.no_grad():
        loss_batch = []  # Store batch loss
        MSE_batch = []  # Store batch MSE
        KLD_batch = []  # Store batch KLD

        for batch in data:
            if not batch.is_cuda:
                batch = batch.to(device, non_blocking=True)

            rec, mean, logvariance = model(batch)
            loss, MSE, KLD = loss_function(rec, batch, mean, logvariance, beta)

            loss_batch.append(loss.item())
            MSE_batch.append(MSE.item())
            KLD_batch.append(KLD.item())

    return sum(loss_batch) / len(loss_batch),\
            sum(MSE_batch) / len(MSE_batch),\
            sum(KLD_batch) / len(KLD_batch),\
            time.time() - start_epoch_time

def gaussian_noise(x, var, device):
    return torch.normal(0.0, var, size=x.shape)

def printProgress(epoch, epochs, loss, loss_test, MSE, KLD, elapsed, elapsed_test, collapsed):
    print(f"Epoch: {epoch:3d}/{epochs:d}, Loss: {loss:2.4f}, Loss_test: {loss_test:2.4f}, MSE: {MSE:2.4f}, KLD: {KLD:2.4f}, collapsed: {collapsed:2d}, time train: {elapsed:2.3f}, time test: {elapsed_test:2.3f}")



##############################################
# Functions used for training temporal-dynamics predictor
################################################


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





