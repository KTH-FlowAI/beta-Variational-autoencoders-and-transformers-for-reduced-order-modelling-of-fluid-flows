import torch
import torch.nn as nn
import time
import numpy as np

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
