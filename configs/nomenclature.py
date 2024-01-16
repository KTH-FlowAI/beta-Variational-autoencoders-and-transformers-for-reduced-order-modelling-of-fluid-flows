"""
Functions for naming the models in the present study
@yuningw 
"""


def Name_VAE(cfg):
    
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for VAE name
    """

    name =  f'smallerCNN_beta{cfg.beta}_' + \
            f'wDecay{cfg.decWdecay}_'+\
            f'dim{cfg.latent_dim}_'+\
            f'lr{cfg.lr}OneCycleLR{cfg.lr_end}_'+\
            f'bs{cfg.batch_size}_'+\
            f'epochs{cfg.epochs}_'+\
            f'nt{cfg.n_train}'
    return name



def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of Transorfmer model configuration
    
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
    A function to name the LSTM checkpoint 

    Args: 
        cfg: A class of LSTM model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.hidden_size}hideen_{cfg.num_layer}nlayer_"+\
                f"_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.num_train}N_{cfg.early_stop}ES_{cfg.patience}P"
    
    return case_name



