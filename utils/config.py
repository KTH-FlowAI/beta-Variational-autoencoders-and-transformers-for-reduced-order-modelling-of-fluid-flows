class VAE_config:
    """
    A class of configuration of vae model
    """
    Re          = 40 
    ModelType   = "smallerCNN" # smallerCNN  OR ResNET0
    beta        = 0.005 # 0.005 or  0.05 or 0.1
    latent_dim  = 2 # 10 or 15 or 20 
    
    lr          = 0.0002 
    batch_size  = 256 
    epoch       = 1000 # 1000 or 800 

    enc_Wdecay  = 0 
    dec_Wdecay  = 0 # 0.0003 or 0 

    num_train   = 27000 #  135000 or 27000

class Transformer_config:
    """
    A class of configuration of Transformer predictor 
    """
    from utils.config import VAE_config 
    
    target      = "VAE" # POD or VAE
    
    in_dim      = 64
    out_dim     = 64 # The output sequence length
    d_model     = 64

    time_proj   = 64 # The projection on time, which is used for new embedding stragtegy 
    next_step   = 1
    if target == "VAE":
        nmode       = VAE_config.latent_dim  # Should be consistent as the modes
    elif target == "POD":
        nmode       = 10  # Choose from [10, 15, 20] 


    num_head    = 4
    attn_type   = "selfconv" # self or selfconv or easy
    # attn_type   = "easy" # self or selfconv or easy

    embed       = "time" # sin / cos/ posenc
    num_block   = 4   # Number of layer 

    is_res_attn = True
    is_res_proj = True
    proj_dim    = 128

    act_proj    = "relu"
    is_output   = True
    out_act     = None

    Epoch       = 100
    Batch_size  = 256
    lr          = 1e-3

    train_split = 0.8 
    num_train   = 135000

    early_stop  = True

    if early_stop == True:
        patience  = 50 # 30 or 50
    else:
        patience  = 0 



class LSTM_config:
    """
    A class of config for LSTM Predictor
    """
    from utils.config import VAE_config 
    target      = "VAE" # POD or VAE    
    in_dim      = 64
    d_model     = 64
    next_step   = 1
    nmode       = VAE_config.latent_dim

    num_layer   = 4
    embed       = None
    
    hidden_size = 128

    is_output   = True
    out_act     = None

    Epoch       = 100
    Batch_size  = 256
    lr          = 1e-3

    train_split = 0.8 
    val_split   = 0.2 
    num_train   = 135000

    early_stop = False

    if early_stop == True:
        patience  = 30
    else:
        patience  = 0 






class Data_config:
    lookback    = 10 
    step        = 1 
    batch_size  = 64 
    n_test      = 1024 
    is_shuffle  = True    


def Name_VAE(cfg):
    
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for VAE name
    """

    name = f"Re{cfg.Re}_{cfg.ModelType}_beta{cfg.beta}_dim{cfg.latent_dim}_lr{cfg.lr}"+\
            f"OneCycleLR1e-05_bs{cfg.batch_size}_epochs{cfg.epoch}_"+\
            f"encWdecay{cfg.enc_Wdecay}_decWdecay{cfg.dec_Wdecay}_nt{cfg.num_train}"
    
    return name


def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for Transformer model
    """

    case_name = f"{cfg.target}_" +\
                f"{cfg.attn_type}Attn"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff"+\
                f"_{cfg.act_proj}act_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.num_train}N_{cfg.early_stop}ES_{cfg.patience}P"
    
    return case_name


def Make_LSTM_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.hidden_size}hideen_{cfg.num_layer}nlayer_"+\
                f"_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.num_train}N_{cfg.early_stop}ES_{cfg.patience}P"
    
    return case_name
