"""
Script for post-processing the model and predictions
"""

import torch 
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time 
# Config
from utils.config import VAE_config as vae_cfg, Name_VAE
from utils.model import VAE
# Plot tools
import utils.plt_rc_setup
from utils.plot_time import colorplate as cc, plot_loss, plot_signal 
from utils.NNs.Transformer import Transformer2, ResTransformer 
from utils.NNs.EmbedTransformerEncoder import EmbedTransformerEncoder
from utils.pp import make_Prediction, Sliding_Window_Error,l2Norm_Error
# ArgParse
import argparse
parser = argparse.ArgumentParser(description="Parameters for post processing")
parser.add_argument("--model","-m", default="attn",type=str,help="Type of model: attn OR lstm")
parser.add_argument("--window","-w", action="store_true",help="Compute sliding window error")
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

base_dir             =   os.getcwd()
base_dir            +=   "/"

modes_data_path      =  base_dir +  "04_Modes/"
save_fig_pred        =  base_dir +  f"05_Figs/vis_pred/dim{vae_cfg.latent_dim}/"
save_fig_chaotic     =  base_dir +  f"05_Figs/vis_chaotic/dim{vae_cfg.latent_dim}/"
save_loss_fig        =  base_dir +  f"05_Figs/loss_fig/dim{vae_cfg.latent_dim}/"
save_data_pred       =  base_dir +  f"06_Preds/dim{vae_cfg.latent_dim}/"



# Generate a model 
if args.model == "attn":
      from utils.NNs.Transformer import Transformer2
      from utils.config import Transformer_config as cfg, Make_Transformer_Name
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



## Regularisation
if (cfg.out_act is not None) and (cfg.out_act == 'tanh'):
      print(f"INFO: Find the output activation using {cfg.out_act}, Regularise data")
      data_max = data_test.max(0)
      data_min = data_test.min(0)
      data_test     = 1 - 2*(data_test- data_min)/(data_max - data_min)
      print(f"INFO: After regularisation, the min = {data_test.min()}, max = {data_test.max()}")


# Load State dict
ckpoint     = torch.load(checkpoint_save_path + fileID + ".pt", map_location= device)
stat_dict   = ckpoint['model']
model.load_state_dict(stat_dict)
print(f'INFO: the state dict has been loaded!')

# Make prediction on training data
evalLen    = 100
Pred_Train = make_Prediction(data[:evalLen,:], model,device,
                             in_dim= cfg.in_dim,
                              next_step= cfg.next_step)
print(f"INFO: Prediction on TRAIN has been generated!")
# Compute l2-norm error 
error_train    = l2Norm_Error(Pred=Pred_Train, Truth=data[:evalLen,:])
print(f"The error on train data for each mode is:{np.round(error_train)}%")


# Make prediction on test data
Preds = make_Prediction(data_test, model,device,
                        in_dim= cfg.in_dim,
                        next_step= cfg.next_step)
print(f"INFO: Prediction on TEST has been generated!")
plot_signal(test_data= data_test, Preds= Preds, 
            save_file= save_fig_pred +\
                    "Pred_" + fileID+ ".jpg",)

error_test    = l2Norm_Error(Pred=Preds, Truth=data_test)
print(f"The error on test data for each mode is:{np.round(error_test)}%")


# Compute the window error if needed
if args.window:
  print(f"Begin to compute the sliding window error")
  window_error = Sliding_Window_Error(data_test, model, device, in_dim= cfg.in_dim)

  plt.figure()
  plt.plot(window_error,lw =2 , c = cc.red)
  plt.ylabel("L2_Norm error")
  plt.xlabel("Window size")
  plt.savefig(save_fig_pred + "Window_" + fileID + ".jpg",bbox_inches="tight")
else: 
  window_error = np.nan

# Save the prediction and the test data.
np.savez_compressed(
                  save_data_pred + fileID + ".npz",
                    p = Preds, 
                    g = data_test,
                    e = window_error
                    )
print(f"INFO: The prediction has been saved!")


# Plot loss evolution and and save fig
plot_loss(history=ckpoint['history'], save_file=save_loss_fig + fileID + ".jpg")
print(f"INFO: Loss evolution has been plotted")



plt.show()