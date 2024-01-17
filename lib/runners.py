"""
Runners for the VAE and temporal-dynamic prediction in latent space 
@yuningw
"""

import os 
import time
from pathlib import Path
import h5py
import numpy as np
import torch 
from torch          import nn

from lib.init       import pathsBib
from lib.train      import * 
from lib.model      import * 
from lib.pp_time    import * 
from lib.datas      import * 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



####################################################
### RUNNER for beta-VAE
####################################################

class vaeRunner(nn.Module):
    def __init__(self, device, datafile) -> None:
        """
        A runner for beta-VAE

        Args:

            device          :       (Str) The device going to use
            
            datafile        :       (Str) Path of training data
        """
        
        from configs.vae import VAE_config as cfg 
        from configs.nomenclature import Name_VAE
        
        super(vaeRunner,self).__init__()
        print("#"*30)

        self.config     = cfg
        self.filename   = Name_VAE(self.config)
        
        self.datafile   = datafile

        self.device     = device

        self.model      = get_vae(self.config.latent_dim)
        
        self.model.to(device)

        print(f"INIT betaVAE, device: {device}")
        print(f"Case Name:\n {self.filename}")
#-------------------------------------------------
    def run(self):
        self.train()
        self.infer()

#-------------------------------------------------
    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self):
        print("#"*30)
        print("INFO: Start post-processing")
        self.load_pretrain_model()


        print(f"INFO: Inference ended!")
        print("#"*30)


#-------------------------------------------------

    def get_data(self): 
        """
        
        Generate the DataLoader for training 

        """
        
        # datafile = 
        
        u_scaled, self.mean, self.std = loadData(self.datafile)
        ## Down-Sample the data with frequency
        ## since we already down-sampled it for database, we skip it here
        
        u_scaled            = u_scaled[::self.config.downsample]
        n_total             = u_scaled.shape[0]
        self.n_train             = n_total - self.config.n_test
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")
        
        self.train_dl, self.val_dl = get_vae_DataLoader(  d_train=u_scaled[:self.n_train],
                                                d_val=u_scaled[self.n_train:],
                                                device= self.device,
                                                batch_size= self.config.batch_size)
        print( f"INFO: Dataloader generated, Num train batch = {len(self.train_dl)} \n" +\
                f"Num val batch = {len(self.val_dl)}")
        
#-------------------------------------------------
    def compile(self):
        """
        
        Compile the optimiser, schedulers and loss function for training

        
        """

        from torch.optim import lr_scheduler
        
        print("#"*30)
        print(f"INFO: Start Compiling")

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        # get optimizer
        self.opt = torch.optim.Adam(
            [   {'params': encoder_params, 'weight_decay': self.config.encWdecay},
                {'params': decoder_params, 'weight_decay': self.config.decWdecay}], 
                lr=self.config.lr, weight_decay=0)
        
        self.opt_sch = lr_scheduler.OneCycleLR(self.opt, 
                                            max_lr=self.config.lr,
                                            total_steps=self.config.epochs, 
                                            div_factor=2, 
                                            final_div_factor=self.config.lr/self.config.lr_end, 
                                            pct_start=0.2)

        self.beta_sch = betaScheduler(  startvalue =self.config.beta_init,
                                        endvalue      =self.config.beta,
                                        warmup        =self.config.beta_warmup)

        print(f"INFO: Compiling Finished!")


#-------------------------------------------------

    def fit(self):
        """

        Training beta-VAE
        
        """
        from torch.utils.tensorboard import SummaryWriter

        print(f"Training {self.filename}")
        logger = SummaryWriter(log_dir=pathsBib.log_path + self.filename)

        bestloss = 1e6
        loss = 1e6

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            beta = self.beta_sch.getBeta(epoch, prints=False)
            loss, MSE, KLD, elapsed, collapsed = train_epoch(model=self.model,
                                                                        data=self.train_dl,
                                                                        optimizer=self.opt,
                                                                        beta=beta,
                                                                        device=self.device)
            self.model.eval()
            loss_test, MSE_test, KLD_test, elapsed_test = test_epoch(model=self.model,
                                                                                data=self.val_dl,
                                                                                beta=beta,
                                                                                device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                                    epochs=self.config.epochs,
                                    loss=loss,
                                    loss_test=loss_test,
                                    MSE=MSE,
                                    KLD=KLD,
                                    elapsed=elapsed,
                                    elapsed_test=elapsed_test,
                                    collapsed=collapsed)

            logger.add_scalar('General loss/Total', loss, epoch)
            logger.add_scalar('General loss/MSE', MSE, epoch)
            logger.add_scalar('General loss/KLD', KLD, epoch)
            logger.add_scalar('General loss/Total_test', loss_test, epoch)
            logger.add_scalar('General loss/MSE_test', MSE_test, epoch)
            logger.add_scalar('General loss/KLD_test', KLD_test, epoch)
            logger.add_scalar('Optimizer/LR', self.opt_sch.get_last_lr()[0], epoch)

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
                ckp_file = f'{pathsBib.chekp_path}/{self.filename}_epoch_bestTest.pth.tar'
                save_checkpoint(state=checkpoint, path_name=ckp_file)
                print(f'## Checkpoint. Epoch: {epoch}, test loss: {loss_test}, saving checkpoint {ckp_file}')

        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
        ckp_file = f'{pathsBib.chekp_path}/{self.filename}_epoch_final.pth.tar'
        save_checkpoint(state=checkpoint, path_name=ckp_file)
        print(f'Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving checkpoint {ckp_file}')


#-------------------------------------------------

    def load_pretrain_model(self):
        """

        Load the pretrained model for beta VAE

        """
        
        try:
            ckpoint = torch.load(pathsBib.pretrain_path + self.filename + ".pth.tar", map_location= self.device)
            
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['state_dict']
        self.model.load_state_dict(stat_dict)
        print(f'INFO: the state dict has been loaded!')
        print(self.model.eval)



####################################################
### RUNNER for Temporal-dynamics Prediction
####################################################


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
        print(f"Case Name:\n {self.filename}")


#-------------------------------------------------

    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, if_window, if_pmap):
        print("#"*30)
        print("INFO: Start post-processing")
        self.load_pretrain_model()
        self.post_process(if_window,if_pmap)
        print(f"INFO: Inference ended!")
        print("#"*30)

#-------------------------------------------------


    def get_data(self):
        """
        Get the latent space variable data for training and validation
        """ 
        try: 
            hdf5 = h5py.File(pathsBib.data_path + "latent_data.h5py")
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

#-------------------------------------------------

    def compile(self): 
        """
        Compile the model with optimizer, scheduler and loss function
        """
        self.loss_fn =   torch.nn.MSELoss()
        self.opt     =   torch.optim.Adam(self.model.parameters(),lr = self.config.lr, eps=1e-7)
        self.opt_sch =  [  
                        torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma= (1 - 0.01)) 
                        ]

#-------------------------------------------------

    def fit(self): 
        """
        Training Model, we use the fit() function 
        """

        s_t = time.time()
        history = fitting(  self.device, 
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
        
        torch.save(check_point,pathsBib.model_path + self.filename+".pt")
        print(f"INFO: The checkpoints has been saved!")

#-------------------------------------------------


    def load_pretrain_model(self):
        try:
            ckpoint = torch.load(pathsBib.model_path + self.filename + ".pt", map_location= self.device)
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['model']
        self.model.load_state_dict(stat_dict)
        print(f'INFO: the state dict has been loaded!')


#-------------------------------------------------

    def post_process(self,if_window=True,if_pmap=True):
        """
        Post Processing of the temporal-dynamics predcition 
        Args:
            
            if_window   :   (bool) If compute the sliding-window error 

            if_pmap     :   (bool) If compute the Poincare Map 
        """ 

        try: 
            hdf5        = h5py.File(pathsBib.data_path + "latent_data.h5py")
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
                            pathsBib.res_path + self.filename + ".npz",
                            p = Preds, 
                            g = test_data,
                            e = window_error,
                            pmap_g = InterSec_test,
                            pmap_p = InterSec_pred
                            )