class VAE_config:
    """
    A class of configuration of vae model
    """
    latent_dim = 10
    
    delta_t = 5
    
    batch_size = 32
    
    lr = 2e-4
    
    lr_end = 1e-5
    
    epochs = 1000
    
    beta = 0.05
    
    beta_init = 0.1
    
    n_test = 200 
    
    encWdecay = 0
    
    decWdecay = 0
    
    DATA_TO_GPU = False

