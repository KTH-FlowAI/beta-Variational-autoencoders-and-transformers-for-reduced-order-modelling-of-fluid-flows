"""
A solver for Reduced-Order model as the second part of work in the paper 
@ author yuningw
"""


import  torch
from    torch.optim                         import  Adam
from    torch.utils.data                    import  TensorDataset, DataLoader, random_split
import  torch.nn                            as      nn
import  numpy                               as      np
import  math
import  time
from    copy                                import  deepcopy
# import predictor 
from    utils.VAE.AutoEncoder               import  BetaVAE
from    utils.NNs.EmbedTransformerEncoder   import  EmbedTransformerEncoder 
from    utils.NNs.easyAttns                 import  easyTransformerEncoder
from    utils.NNs.RNNs                      import  LSTMs

from    utils.configs                       import Name_Costum_VAE, Name_ROM
from    utils.ROM.tools                     import EarlyStopper, Frozen_model
# Import configs

class VAE_ROM(nn.Module):
    def __init__(self, 
                 ROM_config,
                 device,
                 ):
        """
        The Reduced-order model which using $\beta$-VAE for model decomposition 
        and transformer for temporal-dynamics prediction in latent space 

        Args:

            ROM_config      :       (Class) The configuration of ROM 

            device          :       (Str) The device going to use
            
        """


        super(VAE_ROM,self).__init__()
        print("#"*30)
        print(f"INIT the ROM solver")
        if ROM_config.if_pretrain:
                self.forzen_enc =   ROM_config.froze_enc
                self.forzen_dec =   ROM_config.froze_dec
        else:
                self.forzen_enc =   False
                self.forzen_dec =   False

        self.if_pretrain            =   ROM_config.if_pretrain
        self.device                 =   device

        self.ROM_config             =   ROM_config
        self.vae_config             =   ROM_config.mdcp
        self.predictor_config       =   ROM_config.tssp
        
        self.Z                      =   self.vae_config.latent_dim
        # Prepare the VAE
        self.VAE        =   BetaVAE(    zdim         = self.vae_config.latent_dim, 
                                        knsize       = self.vae_config.knsize, 
                                        beta         = self.vae_config.beta, 
                                        filters      = self.vae_config.filters,
                                        block_type   = self.vae_config.block_type,
                                        lineardim    = self.vae_config.linear_dim,
                                        act_conv     = self.vae_config.act_conv,
                                        act_linear   = self.vae_config.act_linear)
        
        
        # Prepare the Seqence model
        if self.ROM_config.predictor_model == "TFSelf":
            assert self.ROM_config.predictor_model == self.predictor_config.model_type, "The input string does not match the config!"
            print(f"Creating Self attention transformer model")
            
            self.Predictor  =  EmbedTransformerEncoder(
                                                        d_input     = self.predictor_config.in_dim,
                                                        d_output    = self.predictor_config.next_step,
                                                        n_mode      = self.predictor_config.nmode,
                                                        d_proj      = self.predictor_config.time_proj,
                                                        d_model     = self.predictor_config.d_model,
                                                        d_ff        = self.predictor_config.proj_dim,
                                                        num_head    = self.predictor_config.num_head,
                                                        num_layer   = self.predictor_config.num_block,
                                                        )
        if self.ROM_config.predictor_model == "TFEasy":
                    assert self.ROM_config.predictor_model == self.predictor_config.model_type, "The input string does not match the config!"
                    print(f"Creating Self attention transformer model")
                    
                    self.Predictor  = easyTransformerEncoder(
                                                        d_input     =   self.predictor_config.in_dim,
                                                        d_output    =   self.predictor_config.next_step,
                                                        seqLen      =   self.predictor_config.nmode,
                                                        d_proj      =   self.predictor_config.time_proj,
                                                        d_model     =   self.predictor_config.d_model,
                                                        d_ff        =   self.predictor_config.proj_dim,
                                                        num_head    =   self.predictor_config.num_head,
                                                        num_layer   =   self.predictor_config.num_block,
                                                        )
            
        if self.ROM_config.predictor_model == "LSTM":
            assert self.ROM_config.predictor_model == self.predictor_config.model_type, "The input string does not match the config!"
            print(f"Creating LSTM model")
            self.Predictor  =   LSTMs(
                                        d_input     = self.predictor_config.in_dim, 
                                        d_model     = self.predictor_config.d_model, 
                                        nmode       = self.predictor_config.nmode,
                                        embed       = self.predictor_config.embed, 
                                        hidden_size = self.predictor_config.hidden_size, 
                                        num_layer   = self.predictor_config.num_layer,
                                        is_output   = self.predictor_config.is_output, 
                                        out_dim     = self.predictor_config.next_step, 
                                        out_act     = self.predictor_config.out_act
                                        )
        
        print("INFO: The ROM has been initialised!")
        print("#"*30)
    def Load_VAE_weight(self,base_dir, nt):

        """
        The function for load the state dict of VAE 

        Args:

            base_dir    :   (Str) The main diretory

            nt          :   (Int) The number of training data has been used 

        Returns:

            The VAE with correct W&B
        """
        
        self.vae_config.filters.reverse()
        case_name       =   Name_Costum_VAE(self.vae_config,nt)
        
        
        load_model_path =   base_dir + "02_Checkpoints/" + case_name + ".pt"
        print(f"INFO: Going to read the case:\n{load_model_path}")

        try:
            ckpoint         =   torch.load(load_model_path,map_location=self.device)
        except:
            print("Error: The file can NOT be loaded")
            print("#"*30)
            quit()

        stat_dic        =   ckpoint['model']
        
        try:
            self.VAE.load_state_dict(state_dict=stat_dic)
        except:
            print("Error: The W&B does NOT match")
            print("#"*30)
            quit()

        print(f'INFO: The case has been loaded successfully!')


        if self.forzen_enc: 
            Frozen_model(self.VAE.encoder)
            print(f"The encoder layers have been frozen")
    
        if self.forzen_dec:
            Frozen_model(self.VAE.decoder)
            print(f"The decoder layers have been frozen!")

        print("#"*30)

    def GenDataLoader(self,data):
        """
        Generate the trainig / validation DataLoader from the given dataset

        Args:

            data    :   (NumPy.Array)  The flow snapshots with shape of [Nt, Nc, Nx, Nz]
        
        Returns:

            self.train_dl    :   The DataLoader for training data

            self.val_dl      :   The DataLoader for validation data. 
        

        """

    
        assert len(data.shape) == 4, "The database should has three dimension"

        print("#"*30)
        print("INFO: Start Generate data for ROM")
        Nt, Nc ,Nx, Ny  =   data.shape
        # The time-delay dimension
        seqLen      =   self.predictor_config.in_dim 
        next_step   =   self.predictor_config.next_step 

        # Generate the tensor for training ROM 
        X   =   torch.empty(size=(Nt, seqLen,   Nc ,  Nx, Ny))
        Y   =   torch.empty(size=(Nt, next_step,Nc  , Nx, Ny))
        
        k   =   0
        for s in range(Nt - seqLen - next_step):
            X[k,:,:,:,:]  =   torch.from_numpy(data[s:s+seqLen, :, :]).float()
            Y[k,:,:,:,:]  =   torch.from_numpy(data[s+seqLen+next_step, :, :, :]).float()
            k           =   k + 1
        
        del data 
        print(f"Delete the data object")
        print(f"The tensor data has been generated as X = {X.shape}, Y = {Y.shape}")
        database        =  TensorDataset(X,Y)
        del X, Y 
        
        lenData         =   len(database)
        valid_size      =   int(lenData * self.predictor_config.val_split)
        train_size      =   int(lenData - valid_size)
        batch_size      =   self.predictor_config.Batch_size
        print(f"Use {train_size} for training and {valid_size} for validation, batch size = {batch_size}")
        
        if valid_size != 0:
            train_d , val_d = random_split(database,[train_size, valid_size])
            
            self.train_dl = DataLoader(train_d,
                                batch_size=batch_size,shuffle=True)
            self.val_dl   = DataLoader(val_d,    
                                batch_size=batch_size,shuffle=True)
            
            del train_d, val_d
            print(f"INFO: The DataLoaders have been generated!")
            x,y     =   next(iter(self.train_dl))
            print(f"One sample from training dataloader has shape of x = {x.shape}, y = {y.shape}")
            del x, y 
            print("#"*30)
            
        else:
            self.train_dl = DataLoader(database,
                                batch_size=batch_size,shuffle=True)
            
            self.val_dl  =  None
            del train_d
            print(f"INFO: The DataLoaders have been generated!")
            x,y     =   next(iter(self.train_dl))
            print(f"One sample from training dataloader has shape of x = {x.shape}, y = {y.shape}")
            del x, y 
            
            print("#"*30)


    def Complie(self, 
                latent_loss_weight, 
                rec_loss_weight, 
                opt_scheduler   , 
                ):
        """
        Compiling the optimizer, loss functions and schedulers for training. 
        
        We use Adam as optimizer and MSE_Loss in both latent space and physical space 

        Args:

            predictor_config    :   (Class) The configuration of predictor

            latent_loss_weight  :   (Float) The weight of loss in latent space 
            
            rec_loss_weight     :   (Float) The weight of loss in physics space 

            opt_scheduler        :   (List) Usage of the learning-rate schedulers or Give it None 

            
        Returns:

            self.optimizer          :   The optimizer object 

            self.latent_loss        :   The loss func in latent space 
            
            self.latent_loss_weight :   The weight of latent loss
            
            self.rec_loss           :   The loss func in physical space  

            self.rec_loss_weigt     :   The weight of reconstruction loss      
        """   
        print("#"*30)
        print(f"Compling of optimizer and loss func")     
        # Optimizer                     
        self.optimizer          =   Adam(self.Predictor.parameters(), 
                                         lr             =   self.predictor_config.lr,
                                         eps            =   1e-7, 
                                         weight_decay   =   self.predictor_config.wdecay
                                         )
        # Loss Func
        self.latent_loss        =   nn.MSELoss()
        self.latent_loss_weight =   latent_loss_weight
        self.rec_loss           =   nn.MSELoss()
        self.rec_loss_weight    =   rec_loss_weight
        
        self.opt_scheduler       =   opt_scheduler
        
        if self.predictor_config.early_stop:
             
            self.early_stopper  =   EarlyStopper(patience= self.predictor_config.patience) 

            print(f"INFO: Early stopping prepared, the patient is {self.predictor_config.patience} Epochs")

        else:
            self.early_stopper  =   None
        
        print(f"Finish compiling!")
        print(f"#"*30)
        
    
    def Solve_ROM(self):
        """
        The function used for training the leverage the Rec_Loss from VAE to penalise the predictor model 


            
        Returns:

            self.history         :   (Dict) The loss evolution     
        """

        print("#"*30)
        print(f"Solving ROM")
        start_time                              = time.time()
        self.history                            = {}
        self.history["train_tot_loss"]          = []
        self.history["val_tot_loss"]            = []
        self.history["train_latent_loss"]       = []
        self.history["val_latent_loss"]         = []
        self.history["train_rec_loss"]          = []
        self.history["val_rec_loss"]            = []

        device  =   self.device
        epochs  =   self.predictor_config.Epoch
        self.Predictor.to(device)
        print(f"INFO: The model is assigned to device: {device} ")
        
        if self.opt_scheduler is not None:
            print(f"INFO: The following schedulers are going to be used:")
            for sch in self.opt_scheduler:
                print(f"{sch.__class__}")


        print(f"Start training loop, totally {epochs} Epochs")

        for epoch in range(epochs):
            tot_loss_epoch = 0;     latent_loss_epoch = 0;      rec_loss_epoch = 0  
            val_tot_loss_epoch = 0; val_latent_loss_epoch = 0;  val_rec_loss_epoch = 0  
            stp = 0  
            print(f"INFO: Training")
            self.Predictor.to(self.device)
            self.VAE.to(self.device)
            self.Predictor.train()
            for x, y in (self.train_dl):
                # Assign the data to the device and make sure they are float32
                x = x.float().to(device)
                y = y.float().to(device)
                stp +=1 

                B, T, C, H, W = x.shape
                _, N, _, _, _ = y.shape
            
                
                src_z_vector            = torch.cat ( [  self.VAE.reparameterize((
                                                                                    self.VAE.encoder(x[:,t,:,:,:])[0].view(B,1,self.Z),
                                                                                    self.VAE.encoder(x[:,t,:,:,:])[1].view(B,1,self.Z)
                                                                                  
                                                                                  )).view(B,1,self.Z) for t in range(T)], dim = 1).to(self.device).float()

                
        
                # Make prediction
                pre_z_vector            =   self.Predictor(src_z_vector)

                tgt_z_vector            =   torch.cat ( [  self.VAE.reparameterize((
                                                                                    self.VAE.encoder(y[:,t,:,:,:])[0].view(B,1,self.Z),
                                                                                    self.VAE.encoder(y[:,t,:,:,:])[1].view(B,1,self.Z)
                                                                                    )).view(B,1,self.Z)  for t in range(N)], dim = 1).to(self.device).float()

                
                latent_loss_            =   self.latent_loss(pre_z_vector, tgt_z_vector)
                
                rec                     =   torch.cat([self.VAE.decoder(pre_z_vector[:,n,:]).view(B, 1, C, H, W) for n in range(N)], 
                                                      dim = 1).to(self.device).float()
                
                rec_loss_               =   self.rec_loss(rec,y)    

                del rec

                tot_loss                =   latent_loss_    * self.latent_loss_weight +\
                                            rec_loss_       * self.rec_loss_weight
                
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()

                tot_loss_epoch      += tot_loss.item()
                latent_loss_epoch   += latent_loss_.item()
                rec_loss_epoch      += rec_loss_.item()

            self.history["train_tot_loss"].append(tot_loss_epoch/stp)
            self.history["train_latent_loss"].append(latent_loss_epoch/stp)
            self.history["train_rec_loss"].append(rec_loss_epoch/stp)
            print(  f"Epoch             = {epoch},\n"+\
                    f"train_tot_loss    = {tot_loss_epoch/stp},\n"+\
                    f"train_rec_loss    = {rec_loss_epoch/stp},\n"+\
                    f"train_latent_loss = {latent_loss_epoch/stp}\n")

            if self.opt_scheduler is not None:
                lr_now = 0 
                for sch in self.opt_scheduler:
                    sch.step()
                    lr_now = sch.get_last_lr()
                print(f"INFO: Scheduler updated, LR = {lr_now} ")


            if self.val_dl != None:

                self.Predictor.eval()
                print("INFO: Validating")
                stp =  0
                for x,y in (self.val_dl):
                    stp +=1 

                    x,  y                   = x.float().to(self.device), y.float().to(self.device)
                    
                    B, T, C, H, W = x.shape
                    _, N, _, _, _ = y.shape
                    
                    src_z_vector            = torch.cat ( [  self.VAE.reparameterize((  self.VAE.encoder(x[:,t,:,:,:])[0].view(B,1,self.Z),
                                                                                        self.VAE.encoder(x[:,t,:,:,:])[1].view(B,1,self.Z)              
                                                                                                      )).view(B,1,self.Z)  for t in range(T)], dim = 1).to(self.device).float()

                 
        
                # Make prediction
                    pre_z_vector                =   self.Predictor(src_z_vector)

                    tgt_z_vector                =   torch.cat ( [  self.VAE.reparameterize((  
                                                                                            self.VAE.encoder(y[:,t,:,:,:])[0].view(B,1,self.Z),
                                                                                            self.VAE.encoder(y[:,t,:,:,:])[1].view(B,1,self.Z)              
                                                                                                      )).view(B,1,self.Z)  for t in range(N)], dim = 1).to(self.device).float()

                
                    val_latent_loss_            =   self.latent_loss(pre_z_vector, tgt_z_vector)
                    
                    rec                         =   torch.cat([self.VAE.decoder(pre_z_vector[:,n,:]).view(B, 1, C, H, W) for n in range(N)], dim = 1).to(self.device).float()
                    val_rec_loss_               =   self.rec_loss(rec,y)         
                    del rec

                    val_tot_loss                =   val_latent_loss_    * self.latent_loss_weight +\
                                                    val_rec_loss_       * self.rec_loss_weight
                    
                    val_tot_loss_epoch          +=  val_tot_loss.item()
                    val_latent_loss_epoch       +=  val_latent_loss_.item()
                    val_rec_loss_epoch          +=  val_rec_loss_.item()

                self.history["val_tot_loss"].append(val_tot_loss_epoch/stp)
                self.history["val_latent_loss"].append(val_latent_loss_epoch/stp)
                self.history["val_rec_loss"].append(val_rec_loss_epoch/stp)
                if self.early_stopper is not None:
                        if self.early_stopper.early_stop(val_tot_loss_epoch/stp):
                            print("Early-stopp Triggered, Going to stop the training")
                            end_time        =   time.time()
                            self.cost_time  =   end_time - start_time
                            
                            print(f"INFO: End Training, total time = {np.round(self.cost_time)}s")         
                            print("#"*30)

                            break
                print(  f"val_tot_loss      = {val_tot_loss_epoch/stp},\n"+\
                        f"val_rec_loss      = {val_rec_loss_epoch/stp},\n"+\
                        f"val_latent_loss   = {val_latent_loss_epoch/stp}\n")

        end_time        =   time.time()
        self.cost_time  =   end_time - start_time 

        print(f"INFO: End Training, total time = {np.round(self.cost_time)}s")         
        print("#"*30)

        return self.history


    def Save_checkpoint(self, base_dir):

        """
        Save the model checkpoints to the folder

        Args:

            base_dir                :   (Str) The main directors used for the code 

            rom_config              :   (Class) The configuration of ROM 
            
            save_predictor_only     :   (Bool) If only save the predictor or the whole ROM

        Returns:

            The saved .pt file in the folder /yourbasedir/06_ROM/VAEPredictor     
                                
            
        """

        print("#"*30)
        case_name           =   Name_ROM(self.ROM_config)

        save_checkpoint_to  =   base_dir + "06_ROM/VAEPredictor/CheckPoints/" + case_name


        print(f"We save the checkpoint as file:\n{save_checkpoint_to}")
        
        
        if self.ROM_config.save_predictor_only: 
            print(f"INFO: Only save the predictor model")
            ckPoint             =   {
                                        "model"         :   self.Predictor.state_dict(), 
                                        "history"       :   self.history,
                                        "time"          :   self.cost_time,
                                        "w_latent"      :   self.latent_loss_weight,
                                        "w_rec"         :   self.rec_loss_weight,

                                            
                                    }

        else:
            
            ckPoint             =   {
                                        "model"         :   self.state_dict(), 
                                        "history"       :   self.history,
                                        "time"          :   self.cost_time,
                                        "w_latent"      :   self.latent_loss_weight,
                                        "w_rec"         :   self.rec_loss_weight,

                                    }

        
        torch.save(ckPoint, save_checkpoint_to + ".pt")


        print(f"INFO: Successfully save the check point!")
        print("#"*30)

    def Load_ROM_state(self, base_dir):

        """
        Load the state dict for the ROM

        Args:
            
            base_dir    :   The base directory for the project

        Returns:

            The loaded module 
        
        """

        print("#"*30)
        
        case_name           =   Name_ROM(self.ROM_config)
        load_checkpoint_as  =   base_dir + "06_ROM/VAEPredictor/CheckPoints/" + case_name + ".pt"
        print(f"INFO: Load ROM state dict from:\n{load_checkpoint_as}")

        ckPoint             =   torch.load(load_checkpoint_as, map_location= self.device)

        if self.ROM_config.save_predictor_only:
            print(f"INFO: Only Load the predictor model")
            try:
                self.Predictor.load_state_dict(ckPoint["model"])
            except:
                print("Error: The file does not match!")

        else:
            print(f"INFO: Load FULL ROM")
            try:
                self.load_state_dict(ckPoint["model"])
            except:
                print("Error: The file does not match!")
        
        print("INFO: The model has been loaded!")
        print("#"*30)


    def GenPredicton(self, test_data):
        """
        Generate the prediction using trained-ROM 

        Args:
            
            test_data:  (NumPy.Array) test data with shape of [Nt, C, H, W]

        Returns:

            Latent_predictions  :   The prediction in latent space

            Physic_predictions  :   The prediction of snapshot        
        """
        assert len(test_data.shape) == 4, "The input data size should be 4D!"

        print("#"*30)
        print("INFO: Start prediction on the test data!")

        self.VAE.to(self.device)
        self.VAE.eval()
        self.Predictor.to(self.device)
        self.Predictor.eval()
        
        Nt, C, H, W             = test_data.shape
        
        seqLen                  =  self.predictor_config.in_dim
        next_step               =  self.predictor_config.next_step
        latent_dim              =  self.vae_config.latent_dim

        u_t     = torch.from_numpy(test_data)
        del test_data
        u       = TensorDataset(u_t, u_t)
        del u_t
        dl      = DataLoader(u, batch_size = 1)
        
        Latent_ref             =    []
        for x, _ in (dl):
            x   = x.float().to(self.device)
            z_mean, z_var = self.VAE.encoder(x)
            z_sample = self.VAE.reparameterize((z_mean, z_var))

            Latent_ref.append(z_sample.detach().cpu().numpy()) 
        
        del dl, x, z_mean, z_var, z_sample 

        print(f"INFO: Generate the reference latent space variables")

        Latent_ref              =   np.concatenate(Latent_ref,0)
        print(f"The generated latent variables are {Latent_ref.shape}")
        Latent_predictions      =   deepcopy(Latent_ref)
        Physic_predictions      =   []
        
        for i  in  range(seqLen, Nt - seqLen - next_step):

            src     =   Latent_predictions[None, i-seqLen:i, :]
          
            src     =   torch.from_numpy(src).float().to(self.device)
          
            pred    =   self.Predictor(src)   

            snap    =   self.VAE.decoder(pred)
          

            Latent_predictions[i:i+next_step, :]                = pred.detach().cpu().numpy()
            
            Physic_predictions.append(snap.detach().cpu().numpy())

        Physic_predictions  =   np.concatenate(Physic_predictions, 0)
        
        print(f"INFO: Prediction finished!")
        
        return Latent_predictions, Physic_predictions
