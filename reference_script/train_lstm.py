"""
The script for training LSTM model
"""

import torch 
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time 
# Config
from utils.config import VAE_config as vae_cfg, Name_VAE, LSTM_config as cfg, Make_LSTM_Name
from utils.model import VAE
# Plot tools
import utils.plt_rc_setup
from utils.plot_time import colorplate as cc 
from utils.NNs.RNNs import LSTMs 
from utils.datas import make_DataLoader, make_Sequence 
from utils.train import fit
from utils.pp import make_Prediction
#confirm device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Name the case and path
vae_file    =  Name_VAE(vae_cfg)
print(f"VAE case name is {vae_file}")
fileID  =  Make_LSTM_Name(cfg)
print(f"The transformer name is {fileID}")

checkpoint_save_path = "03_Checkpoints/LSTM/"
modes_data_path = "04_Modes/"
save_fig_pred   = f"05_Figs/vis_pred/dim{vae_cfg.latent_dim}/"
save_data_pred  = f"06_Preds/dim{vae_cfg.latent_dim}/"

# Load the data 
d = h5py.File(modes_data_path + vae_file + ".h5py")
data      = np.array(d['vector'])
data_test = np.array(d['vector_test'])
print(f"INFO: The training and test sequence data has been loaded "+\
      f"train={data.shape}, test={data_test.shape} ")
d.close()
print(f"INFO: Close the Data Dictionary")

# Generate the Training Data and DataLoader
X, Y = make_Sequence(cfg=cfg, data=data)

train_dl, val_dl = make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                   batch_size=cfg.Batch_size,
                                   drop_last=False, train_split=cfg.train_split)

print(f"INFO: The DataLoaders made, num of batch in train={len(train_dl)}, validation={len(val_dl)}")
## Examine the input shape 
x,y = next(iter(train_dl))
print(f"Examine the input and output shape = {x.shape, y.shape}")


# Generate a model 
model = LSTMs(
                d_input= cfg.in_dim, d_model= cfg.d_model, nmode= cfg.nmode,
                embed= cfg.embed, hidden_size= cfg.hidden_size, num_layer= cfg.num_layer,
                is_output= cfg.is_output, out_dim= cfg.next_step, out_act= cfg.out_act
                )

NumPara = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"INFO: The model has been generated, num of parameter is {NumPara}")

## Compile 
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(),lr = cfg.lr, eps=1e-7)
opt_sch = [  
            torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma= (1 - 0.01)) 
            ]
# Training 
s_t = time.time()
history = fit(device, model, train_dl, 
           loss_fn,cfg.Epoch,opt,val_dl, 
           scheduler=opt_sch,if_early_stop=cfg.early_stop,patience=cfg.patience)
e_t = time.time()
cost_time = e_t - s_t
print(f"INFO: Training ended, spend time = {np.round(cost_time)}s")
# Save Checkpoint
check_point = {"model":model.state_dict(),
               "history":history,
               "time":cost_time}
torch.save(check_point,checkpoint_save_path+fileID+".pt")
print(f"INFO: The checkpoints has been saved!")


