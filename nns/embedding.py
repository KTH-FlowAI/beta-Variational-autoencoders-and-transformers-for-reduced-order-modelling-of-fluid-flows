import torch
import torch.nn as nn 
from torch.nn import Module

######################################################
#### Time2Vec architecture
######################################################

def t2v(tau, f, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)

    
    v2 = torch.matmul(tau, w0) + b0
  
    return torch.cat([v1, v2], -1)

class SineActivation(Module):
    def __init__(self, in_features,nmodes, out_features):
        super(SineActivation, self).__init__()
       
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,nmodes))
        self.b0 = nn.parameter.Parameter(torch.randn(nmodes))
        
        self.w = nn.parameter.Parameter(torch.randn(in_features,out_features-nmodes))
        self.b = nn.parameter.Parameter(torch.randn(out_features-nmodes))
        
        self.f = torch.sin

        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w)
    

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

class CosineActivation(Module):
    def __init__(self, in_features,nmodes, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,nmodes))
        self.b0 = nn.parameter.Parameter(torch.randn(nmodes))
        
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-nmodes))
        self.b = nn.parameter.Parameter(torch.randn(out_features-nmodes))
        
        self.f = torch.cos
    
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w)

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)





import math


class PosEncoding(Module):
    def __init__(self,in_features,nmodes,out_features ) -> None:
        super(PosEncoding,self).__init__()
        
        self.proj = nn.Linear(nmodes,out_features)
       
        pe = torch.zeros((in_features, out_features))
        
        position = torch.arange(0, in_features).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, out_features, 2, dtype=torch.float) *
                            -(math.log(10000.0) / out_features)))
        
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe

        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self,x):    
        x_emb = self.proj(x)
       
        return x_emb + self.pe[:, :x_emb.size(-1)].to(x.device)


def Positional_Encoding(Batch,d_model, nmode):
    """
    :param d_model: dimension of the model
    :param nmode: nmode of positions
    :return: nmode*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(Batch,nmode, d_model)
    position = torch.arange(0, nmode).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:,:, 0::2] = torch.sin(position.float() * div_term)
    pe[:,:, 1::2] = torch.cos(position.float() * div_term)

    return pe




"""

Create a new embedding strategy for time and space embedding

@ yuningw

"""

import torch

from torch import nn 

import numpy as np


class TimeSpaceEmbedding(nn.Module):

    """"

    A embedding module based on both time and space
    Args:

    d_input : The input size of timedelay embedding

    n_mode : The number of modes/dynamics in the time series 

    d_expand : The projection along the time

    d_model : The projection along the space 

    """

    def __init__(self, d_input, n_mode,
                d_expand,d_model ):

        super(TimeSpaceEmbedding, self).__init__()

        self.spac_proj      = nn.Linear(n_mode,d_model)

        self.time_proj      = nn.Conv1d(d_input, d_expand,1)

        self.time_avgpool   = nn.AvgPool1d(2)
        self.time_maxpool   = nn.MaxPool1d(2)
        self.time_compress  = nn.Linear(d_model, d_model)
        self.act            = nn.Identity()


        nn.init.xavier_uniform_(self.spac_proj.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.time_compress.weight)

    def forward(self, x):
        
        # Along space projection
        x       = self.spac_proj(x)
        
        # Along the time embedding 
        x       = self.time_proj(x)
        timeavg = self.time_avgpool(x)
        timemax = self.time_maxpool(x)
        tau     = torch.cat([timeavg, timemax],-1)
        out     = self.act(self.time_compress(tau))
        return out
