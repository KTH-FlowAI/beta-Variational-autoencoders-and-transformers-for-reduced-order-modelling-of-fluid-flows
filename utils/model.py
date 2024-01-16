"""
Function for obtain the models from nns library

@yuningw 
"""

import torch 
from torch import nn 


def get_predictors(name):
    """
    A function to obatin the model for temporal dynamics predictor in latent space 

    Args:
        name    : (str) Name between; attn, self and lstm 
    
    Returns:
        model       : (torch.nn.Module) The model
        
        filename    : (str) The filename to save the model 

        config      : (class) The configuration of the model 
    """
    assert(name=="self" or name == "easy" or name == "lstm"), print("ERROR: The Name is not Valid!")

    if name == "easy":
        from configs.easyAttn import easyAttn_config as cfg, Make_Transformer_Name
        from nns.transformer import easyTransformerEncoder
        try:
            model = easyTransformerEncoder( d_input = cfg.in_dim,
                                        d_output= cfg.next_step,
                                        seqLen  = cfg.nmode,
                                        d_proj  = cfg.time_proj,
                                        d_model = cfg.d_model,
                                        d_ff    = cfg.proj_dim,
                                        num_head = cfg.num_head,
                                        num_layer = cfg.num_block,)
        except: 
            print("ERROR: Parameter NOT MATCHED!")
            exit()

        filename = Make_Transformer_Name(cfg)
        print(f"Easy-Attention-based Transformer has been generated")
        print(f"FileName: {filename}")
        return model, filename, cfg
    
    if name == "easy":
        from configs.selfAttn import selfAttn_config as cfg, Make_Transformer_Name
        from nns.transformer import EmbedTransformerEncoder
        try:
            model = EmbedTransformerEncoder( d_input = cfg.in_dim,
                                        d_output= cfg.next_step,
                                        seqLen  = cfg.nmode,
                                        d_proj  = cfg.time_proj,
                                        d_model = cfg.d_model,
                                        d_ff    = cfg.proj_dim,
                                        num_head = cfg.num_head,
                                        num_layer = cfg.num_block,)
        except: 
            print("ERROR: Parameter NOT MATCHED!")
            exit()

        filename = Make_Transformer_Name(cfg)
        print(f"Self-Attention-based Transformer has been generated")
        print(f"FileName: {filename}")
        return model, filename, cfg
    
    
    if name =="lstm":
        from configs.lstm import lstm_config as cfg, Make_LSTM_Name
        from nns.RNNs import LSTMs
        try:
            model = LSTMs(
                d_input= cfg.in_dim, d_model= cfg.d_model, nmode= cfg.nmode,
                embed= cfg.embed, hidden_size= cfg.hidden_size, num_layer= cfg.num_layer,
                is_output= cfg.is_output, out_dim= cfg.next_step, out_act= cfg.out_act
                )
        except: 
            print("ERROR: Parameter NOT MATCHED!")
            exit()
        
        filename = Make_LSTM_Name(cfg)
        print(f"LSTM has been generated")
        print(f"FileName: {filename}")
        return model, filename, cfg
    
