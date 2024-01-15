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
