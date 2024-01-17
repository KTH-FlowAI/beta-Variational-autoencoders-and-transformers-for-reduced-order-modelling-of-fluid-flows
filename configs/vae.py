class VAE_config:
    """
    A class of configuration of vae model
    """
    latent_dim = 2
    
    delta_t = 1
    
    batch_size = 256
    
    lr = 2e-4
    
    lr_end = 1e-5
    
    epochs = 1000
    
    beta = 0.001
    
    beta_init = 0.001

    beta_warmup = 20
    
    n_test = 100
    
    encWdecay = 0
    
    decWdecay = 0
    
    DATA_TO_GPU = False

