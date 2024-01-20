"""
The script for training transformer model
"""

import torch 
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time 
from utils.model import VAE
from utils.figs_time import colorplate as cc 
from NNs.transformer    import easyTransformerEncoder
from utils.datas        import make_DataLoader, make_Sequence 
from utils.train        import fit
import os
import argparse
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#confirm device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
parser = argparse.ArgumentParser()
parser.add_argument('-model',default="easy",type=str,help="Choose the model for time-series prediction: easy, self OR lstm")
args  = parser.parse_args()



# Name the case and path
from configs.vae import VAE_config as vae_cfg, Name_VAE
vae_file    =  Name_VAE(vae_cfg)
print(f"VAE case name is {vae_file}")

base_dir                = os.getcwd()
base_dir                += "/"
checkpoint_save_path    = base_dir +  "03_Checkpoints/Attn/"
modes_data_path         = base_dir +  "04_Modes/"
save_fig_pred           = base_dir +  f"05_Figs/vis_pred/dim{vae_cfg.latent_dim}/"
save_data_pred          = base_dir +  f"06_Preds/dim{vae_cfg.latent_dim}/"






# Generate the Training Data and DataLoader
X, Y = make_Sequence(cfg=cfg, data=data)

train_dl, val_dl = make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                    batch_size=cfg.Batch_size,
                                    drop_last=False, train_split=cfg.train_split)

print(f"INFO: The DataLoaders made, num of batch in train={len(train_dl)}, validation={len(val_dl)}")
## Examine the input shape 
x,y = next(iter(train_dl))
print(f"Examine the input and output shape = {x.shape, y.shape}")
model = easyTransformerEncoder(
                                    d_input = cfg.in_dim,
                                    d_output= cfg.next_step,
                                    seqLen  = cfg.nmode,
                                    d_proj  = cfg.time_proj,
                                    d_model = cfg.d_model,
                                    d_ff    = cfg.proj_dim,
                                    num_head = cfg.num_head,
                                    num_layer = cfg.num_block,
                                    )
NumPara = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"INFO: The model has been generated, num of parameter is {NumPara}")

## Compile 
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(),lr = cfg.lr, eps=1e-7)
opt_sch = [  
            torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma= (1 - 0.01)) 
            ]
# opt_sch = None
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

