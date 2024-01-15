"""
The main class object used for bulidng the ROM with beta-VAE and transformer

@yuningw
"""

from    torch.optim                         import  Adam
from    torch.utils.data                    import  TensorDataset, DataLoader, random_split
import  torch.nn                            as      nn
import  numpy                               as      np
import  math
import  time
from    copy                                import  deepcopy


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