class lstm_config:
    """
    A class of config for LSTM Predictor
    """
    from configs.vae import VAE_config 

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

    early_stop = True

    if early_stop == True:
        patience  = 50
    else:
        patience  = 0 




