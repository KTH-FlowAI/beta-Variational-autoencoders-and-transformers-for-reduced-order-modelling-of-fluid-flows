import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path

import lib_data
import lib_model
import lib_training

# Get system info
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i))
device = torch.device('cuda')

# training parameters
delta_t = 5
batch_size = 256
lr = 2e-4
lr_end = 1e-5
epochs = 1000
beta = 0.05
beta_init = 0.1
betaSch = lib_training.betaScheduler(startvalue=beta_init, endvalue=beta, warmup=20)
latent_dim = 20
n_test = 15000//delta_t
encWdecay = 0
decWdecay = 0
DATA_TO_GPU = False

# creating directories if do not exist
Path("01_data").mkdir(exist_ok=True)
Path("02_logs").mkdir(exist_ok=True)
Path("03_checkpoints").mkdir(exist_ok=True)
Path("04_modes").mkdir(exist_ok=True)

# load data
datafile = '01_data/Re100alpha10_newData_150000.hdf5'

u_scaled, mean, std = lib_data.loadData(datafile)
u_scaled = u_scaled[::delta_t]
n_total = u_scaled.shape[0]
n_train = n_total - n_test
print(f"N train: {n_train:d}, N test: {n_test:d}, N total {n_total:d}")

if(DATA_TO_GPU):
    dataset_train = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[:n_train]).cuda(0), batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    dataset_test = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[n_train:]).cuda(0), batch_size=batch_size,
                                                shuffle=False, num_workers=0)

else:
    dataset_train = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[:n_train]), batch_size=batch_size,
                                                shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    dataset_test = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[n_train:]), batch_size=batch_size,
                                                shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
# Get model
model = lib_model.VAE(latent_dim=latent_dim).to(device)
encoder_params = list(model.encoder.parameters())
decoder_params = list(model.decoder.parameters())

# get optimizer
optimizer = torch.optim.Adam(
    [{'params': encoder_params, 'weight_decay': 0},
     {'params': decoder_params, 'weight_decay': 0}], lr=lr, weight_decay=0)
#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=lr_end/lr, total_iters=epochs)
#scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr_end, max_lr=lr, step_size_up=50, mode="triangular2", cycle_momentum=False)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs, div_factor=2, final_div_factor=lr/lr_end, pct_start=0.2)

# train loop
strdate = time.strftime("%Y%m%d_%H_%M")
modelname = f'{strdate}_smallerCNN_beta{beta}_wDecay{decWdecay}_dim{latent_dim}_lr{lr}OneCycleLR{lr_end}_bs{batch_size}_epochs{epochs}_nt{n_train}'
print(modelname)
logger = SummaryWriter(log_dir='./02_logs/' + modelname)
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

# Cleanup
logger.flush()
logger.close()
del model
del optimizer

print(modelname)
print('END')
