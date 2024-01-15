class lstm_config:
    """
    A class of config for LSTM Predictor
    """
    from configs.vae import VAE_config 
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
