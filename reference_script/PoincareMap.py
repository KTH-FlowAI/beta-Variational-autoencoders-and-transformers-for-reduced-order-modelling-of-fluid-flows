"""
Example Script to implement Poincare Maps on test data 
"""

#Environment
import torch 
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time 
# Config
from utils.config import VAE_config as vae_cfg, Name_VAE, Transformer_config as cfg, Make_Transformer_Name
from utils.model import VAE
# Model
from utils.NNs.Transformer import Transformer2, ResTransformer
# Postprocessing
from utils.pp import make_Prediction, Sliding_Window_Error
from utils.chaotic import Zero_Cross, Intersection, PDF
# Plot tools
import utils.plt_rc_setup
from utils.figs_time import colorplate as cc, plot_loss, plot_signal 


# ArgParse
import argparse
parser = argparse.ArgumentParser(description="Parameters for post processing")
parser.add_argument("--model","-m", default="attn",type=str,help="Type of model: attn OR lstm")
args   = parser.parse_args()

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#confirm device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Name the case and path
vae_file    =  Name_VAE(vae_cfg)
print(f"VAE case name is {vae_file}")


if args.model   == "attn":
  checkpoint_save_path =  "03_Checkpoints/Attn/"
elif args.model == "lstm":
  checkpoint_save_path =  "03_Checkpoints/LSTM/"
else:
  print(f"Warning: NOT a correct model type!")
  quit()

base_dir             =  "/mimer/NOBACKUP/groups/deepmechalvis/yuningw/2Plates/"
modes_data_path      =   base_dir + "04_Modes/"
save_fig_pred        =   base_dir + f"05_Figs/vis_pred/dim{vae_cfg.latent_dim}/"
save_fig_chaotic     =   base_dir + f"05_Figs/vis_chaotic/dim{vae_cfg.latent_dim}/"
save_loss_fig        =   base_dir + f"05_Figs/loss_fig/dim{vae_cfg.latent_dim}/"
save_data_pred       =   base_dir + f"06_Preds/dim{vae_cfg.latent_dim}/"

# Generate a model 
if args.model == "attn":
      from utils.NNs.Transformer import Transformer2, ResTransformer
      from utils.config import Transformer_config as cfg, Make_Transformer_Name
      from utils.NNs.EmbedTransformerEncoder import EmbedTransformerEncoder
      fileID  = Make_Transformer_Name(cfg)

      model = EmbedTransformerEncoder(
                                    d_input = cfg.in_dim,
                                    d_output= cfg.next_step,
                                    n_mode  = cfg.nmode,
                                    d_proj  = cfg.time_proj,
                                    d_model = cfg.d_model,
                                    d_ff    = cfg.proj_dim,
                                    num_head = cfg.num_head,
                                    num_layer = cfg.num_block,
                                    )
if args.model == 'lstm':
  from utils.NNs.RNNs import LSTMs
  from utils.config import LSTM_config as cfg, Make_LSTM_Name
  fileID = Make_LSTM_Name(cfg)
  model = LSTMs(
                d_input= cfg.in_dim, d_model= cfg.d_model, nmode= cfg.nmode,
                embed= cfg.embed, hidden_size= cfg.hidden_size, num_layer= cfg.num_layer,
                is_output= cfg.is_output, out_dim= cfg.next_step, out_act= cfg.out_act
                )

NumPara = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"INFO: The model has been generated, num of parameter is {NumPara}")

# Load State dict
print(f"INFO: The fileID: {fileID}")
ckpoint     = torch.load(checkpoint_save_path + fileID + ".pt", map_location= device)
stat_dict   = ckpoint['model']
model.load_state_dict(stat_dict)
print(f'INFO: the state dict has been loaded!')



# Load the data 
if cfg.target == "VAE":
      print(f"INFO: Going to predict {cfg.target} modes")
      d = h5py.File(modes_data_path + vae_file + ".h5py")
      data      = np.array(d['vector'])
      data_test = np.array(d['vector_test'])
      print(f"INFO: The training and test sequence data has been loaded "+\
            f"train={data.shape}, test={data_test.shape} ")
      d.close()
      print(f"INFO: Close the Data Dictionary")
elif cfg.target == "POD":
      pod_data_dir    = "08_POD/"
      R               = cfg.nmode
      datfileName     = base_dir + pod_data_dir + f"Intermediate_POD_Data_{R}modes.h5py" 
      print(f"Going to load rank= {R}, path: {datfileName}")
      try: 
            with h5py.File(datfileName, "r") as f:
                  print(f"INFO The file has been loaded, the keys are:\n{f.keys()}")
                  """
                  Description: 
                  In the file there are: 
                        Psi_train:      Spatial modes 
                        Sigma_train:    Singular Value 
                        phi_train:      Temporal Modes
                  """
                  data     =    np.array(f["Psi_train"], dtype=np.float32)
            f.close()
            print(f"The data has been loaded with shape of {data.shape}")
            data  = data[:-5000,:]
            print(f"Use last 5000 for test, training data = {data.shape}")
            data_test = data[-5000:,:]
            print(f"Test data set shape = {data_test.shape}")
      except:
            print("Warning: The dataset does not exist, ending")
            quit()



# Make prediction on test data
Preds = make_Prediction(data_test, model,device,
                        in_dim= cfg.in_dim,
                        next_step= cfg.next_step)
print(f"INFO: Prediction has been generated!")

# Compute the Intersection based on plane 0 and postive direction
## Basic setup for Pmap
planeNo      = 0 
postive_dir  = True
lim_val      = 2.5 # Limitation of x and y bound when compute joint pdf 
grid_val     = 50  # Mesh grid number for plot Pmap
Pmap_Info    = f"PMAP_{planeNo}P_{postive_dir}pos_{lim_val}lim_{grid_val}grid_"

InterSec_pred = Intersection(Preds,
                             planeNo=planeNo,postive_dir=postive_dir)
print(f"The intersection of Prediction has shape of {InterSec_pred.shape}")

InterSec_test = Intersection(data_test,
                             planeNo=planeNo,postive_dir=postive_dir)
print(f"The intersection of Test Data has shape of {InterSec_test.shape}")

# Plot all the Poincare map for each section
Nmodes   = data_test.shape[-1]
fig, axs = plt.subplots(Nmodes,Nmodes, 
                        figsize=(2.5* cfg.nmode, 2.5* cfg.nmode),
                        sharex=True,sharey=True)

for i in range(0,Nmodes):
        for j in range(0,Nmodes):
            if(i==j or j==planeNo or i==planeNo or j>i):
                axs[i,j].set_visible(False)
                continue
            
            xx,yy, pdf_test      = PDF(   InterSecX= InterSec_test[:,i],
                                          InterSecY= InterSec_test[:,j],
                                          xmin=-lim_val,xmax=lim_val,
                                          ymin=-lim_val,ymax=lim_val,
                                          x_grid=grid_val,y_grid=grid_val,
                                    )
                  

            xx,yy, pdf_pred      = PDF(   InterSecX= InterSec_pred[:,i],
                                          InterSecY= InterSec_pred[:,j],
                                          xmin=-lim_val,xmax=lim_val,
                                          ymin=-lim_val, ymax=lim_val,
                                          x_grid=grid_val,y_grid=grid_val,
                                    )


            
            axs[i,j].contour(xx,yy,pdf_test,colors=cc.black)
            axs[i,j].contour(xx,yy,pdf_pred,colors=cc.blue)
            axs[i,j].set_xlim(-lim_val,lim_val)
            axs[i,j].set_xlabel(f"r{i+1}",fontsize='large')
            axs[i,j].set_ylabel(f"r{j+1}",fontsize='large')
            axs[i,j].set_aspect('equal',"box")
            axs[i,j].grid(visible=True,markevery=1,color='gainsboro', zorder=1)
 
plt.savefig(save_fig_chaotic + Pmap_Info + fileID + ".jpg", bbox_inches="tight")
print(f"The image has been saved as:\n {save_fig_chaotic + Pmap_Info + fileID}")