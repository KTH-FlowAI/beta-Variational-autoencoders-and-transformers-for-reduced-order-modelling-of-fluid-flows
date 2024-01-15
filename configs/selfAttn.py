class selfAttn_config:
    """
    A class of configuration of Transformer predictor 
    """
    from configs.vae import VAE_config 
    
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