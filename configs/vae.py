class VAE_config:
    """
    A class of configuration of vae model
    """

    Re          = 40 # Reynolds number for the case

    latent_dim  = 2
    
    delta_t     = 1
    
    batch_size  = 256
    
    lr          = 2e-4
    
    lr_end      = 1e-5
    
    epochs      = 1000
    
    beta        = 0.005
    
    beta_init   = 0.001

    downsample  = 1 # We set = 1, change if you need 

    beta_warmup = 20
    
    n_test      = 200
    
    encWdecay   = 0
    
    decWdecay   = 0
    
    DATA_TO_GPU = False
    
