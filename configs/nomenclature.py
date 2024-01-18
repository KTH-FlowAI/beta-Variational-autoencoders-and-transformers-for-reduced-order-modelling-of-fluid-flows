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

    name =  f"Re{cfg.Re}_" +\
            f'smallerCNN_'+\
            f'beta{cfg.beta}_' + \
            f'wDecay{cfg.decWdecay}_'+\
            f'dim{cfg.latent_dim}_'+\
            f'lr{cfg.lr}OneCycleLR{cfg.lr_end}_'+\
            f'bs{cfg.batch_size}_'+\
            f'epochs{cfg.epochs}'
    
    return name



def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of Transorfmer model configuration
    
    Returns:
        name: A string for Transformer model
    """
    
    case_name = f"{cfg.attn_type}Attn_"+\
                f"{cfg.in_dim}in_"      +\
                f'{cfg.d_model}dmodel_'  +\
                f'{cfg.next_step}next_'+\
                f'{cfg.nmode}dim_'+\
                f"{cfg.embed}emb_"+\
                f"{cfg.num_head}h_"+\
                f"{cfg.num_block}nb_"+\
                f"{cfg.proj_dim}ff_"+\
                f"{cfg.act_proj}act_"+\
                f"{cfg.out_act}outact_"+\
                f"{cfg.Epoch}Epoch_"+\
                f"{cfg.num_train}N_"+\
                f"{cfg.early_stop}ES_"+\
                f"{cfg.patience}P"
    return case_name



def Make_LSTM_Name(cfg):
    """
    A function to name the LSTM checkpoint 

    Args: 
        cfg: A class of LSTM model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM_"+\
                f"{cfg.in_dim}in_"+\
                f"{cfg.d_model}dmodel_"+\
                f"{cfg.next_step}next_"+\
                f"{cfg.nmode}dim_"+\
                f"{cfg.embed}emb_"+\
                f"{cfg.hidden_size}hideen_"+\
                f"{cfg.num_layer}nlayer_"+\
                f"{cfg.out_act}outact_"+\
                f"{cfg.Epoch}Epoch_"+\
                f"{cfg.num_train}N_"+\
                f"{cfg.early_stop}ES_"+\
                f"{cfg.patience}P"
    
    return case_name



