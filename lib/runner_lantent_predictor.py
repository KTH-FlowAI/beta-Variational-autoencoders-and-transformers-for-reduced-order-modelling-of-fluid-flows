"""
A runner for the temporal-dynamic prediction in latent space 
@yuningw
"""

import os 
import time
import h5py
import numpy as np
import torch 
from torch          import nn 
from utils.model    import get_predictors
from utils.train    import fit
from utils.datas    import make_DataLoader, make_Sequence
from utils.pp       import make_Prediction, Sliding_Window_Error
from utils.chaotic  import Intersection

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
data_path   = 'data/'

model_path  = 'models/'; 
if not os.path.exists(model_path): os.makedirs(model_path)

res_path    = 'results/preds/'
if not os.path.exists(res_path): os.makedirs(res_path)

fig_path    = 'results/figs/'
if not os.path.exists(fig_path): os.makedirs(fig_path)



class latentRunner(nn.Module): 
    def __init__(self,name,device):
        """
        A runner for latent space temporal-dynmaics prediction

        Args:

            name            :       (str) The model choosed for temporal-dynamics prediction 

            device          :       (Str) The device going to use
            
        """

        super(latentRunner,self).__init__()
        print("#"*30)
        print(f"INIT temporal predictor: {name}, device: {device}")
        self.device = device
        self.model, self.filename, self.config = get_predictors(name)
        self.NumPara = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"INFO: The model has been generated, num of parameter is {self.NumPara}")


    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        
        self.compile()
        
        self.run()


    def get_data(self):
        """
        Get the latent space variable data for training and validation
        """ 
        try: 
            # hdf5 = h5py.File("data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5")
            hdf5 = h5py.File(data_path + "latent_data.h5py")
            data   = np.array(hdf5['vector'])
        except:
            print(f"Error: DataBase not found, please check path or keys")

        X,Y = make_Sequence(self.config,data=data)
        self.train_dl, self.val_dl =make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                                    batch_size=self.config.Batch_size,
                                                    drop_last=False, 
                                                    train_split=self.config.train_split)
        print(f"INFO: DataLoader Generated!")
        del data, X, Y

    def compile(self): 
        """
        Compile the model with optimizer, scheduler and loss function
        """
        self.loss_fn =   torch.nn.MSELoss()
        self.opt     =   torch.optim.Adam(self.model.parameters(),lr = self.config.lr, eps=1e-7)
        self.opt_sch =  [  
                        torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma= (1 - 0.01)) 
                        ]
    
    def run(self): 
        """
        Training Model, we use the fit() function 
        """

        s_t = time.time()
        history = fit(      self.device, 
                            self.model,
                            self.train_dl, 
                            self.loss_fn,
                            self.config.Epoch,
                            self.opt,
                            self.val_dl, 
                            scheduler=self.opt_sch,
                            if_early_stop=self.config.early_stop,
                            patience=self.config.patience)
        e_t = time.time()
        cost_time = e_t - s_t
        
        print(f"INFO: Training FINISH, Cost Time: {cost_time:.2f}s")
        
        check_point = { "model":self.model.state_dict(),
                        "history":history,
                        "time":cost_time}
        
        torch.save(check_point,model_path + self.filename+".pt")
        print(f"INFO: The checkpoints has been saved!")


    def load_pretrain_model(self):
        try:
            ckpoint = torch.load(model_path + self.filename + ".pt", map_location= self.device)
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['model']
        self.model.load_state_dict(stat_dict)
        print(f'INFO: the state dict has been loaded!')


    def post_process(self,if_window=True,if_pmap=True):
        """
        Post Processing of the temporal-dynamics predcition 
        Args:
            
            if_window   :   (bool) If compute the sliding-window error 

            if_pmap     :   (bool) If compute the Poincare Map 
        """ 

        try: 
            # hdf5 = h5py.File("data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5")
            hdf5        = h5py.File(data_path + "latent_data.h5py")
            test_data   = np.array(hdf5['vector_test'])
        except:
            print(f"Error: DataBase not found, please check path or keys")

        print(f"INFO: Test data loaded, SIZE = {test_data.shape}")
        Preds = make_Prediction(test_data   = test_data, 
                                model       = self.model,
                                device      = self.device,
                                in_dim      = self.config.in_dim,
                                next_step   = self.config.next_step)
        
        if if_window: 
            print(f"Begin to compute the sliding window error")
            window_error = Sliding_Window_Error(test_data, 
                                                self.model, 
                                                self.device, 
                                                self.config.in_dim)
        else: 
            window_error = np.nan
        
        
        if if_pmap:
            planeNo      = 0 
            postive_dir  = True
            lim_val      = 2.5 # Limitation of x and y bound when compute joint pdf 
            grid_val     = 50
            InterSec_pred = Intersection(Preds,     planeNo=planeNo,postive_dir=postive_dir)
            InterSec_test = Intersection(test_data, planeNo=planeNo,postive_dir=postive_dir)
        else:
            InterSec_pred = np.nan
            InterSec_test = np.nan
        
        
        np.savez_compressed(
                            res_path + self.filename + ".npz",
                            p = Preds, 
                            g = test_data,
                            e = window_error,
                            pmap_g = InterSec_test,
                            pmap_p = InterSec_pred
                            )