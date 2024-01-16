"""
Recurrent network model in torch.nn.Module

@author Yuning Wang
"""
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm
from nns.embedding import SineActivation, CosineActivation, PosEncoding
import torch.nn as nn 

class LSTMs(nn.Module):
    def __init__(self,  d_input,
                        d_model,
                        nmode,
                        embed       = None,
                        hidden_size = 64,
                        num_layer   = 1,
                        is_output   = True, 
                        out_dim     = 1,
                        out_act     = "tanh", 
                        is_output_bias = True) -> None:
        """
        Module for Long Short-term Memory architecture 
        
        Note that by default, we do not use embedding layer for RNN model since it is able to recognize the sequence data
        
        Args:
            d_input         : (Int) The input dimension

            d_model         : (Int) The projection dimension
            
            nmode           : (Int) Number of mode
            
            embed           : (Str) Embed layer, if not use embedding then args is None
            
            hidden_size     : (Str) Number of hidden cells in RNN
            
            num_layer       : (Str) Number of layers used in RNN
            
            is_output       : (Bool) If use output layer
            
            out_act         : (Str) The activation function used for output
            
            is_out_put_bias : (Bool) If use bias for output layer 



        
        """

        super(LSTMs,self).__init__()

        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.is_output = is_output
        self.out_dim = out_dim
        if embed == "sin":
            self.embed = SineActivation(nmode,nmode,d_model)
        if embed == "cos":
            self.embed = CosineActivation(nmode,nmode,d_model)
        if embed == "posenc":
            self.embed = PosEncoding(d_input,nmode,d_model)
        
        if embed == None:
            try: d_model == d_input
            except: print("INFO: NO Embedding! \nThe input dimension should be same as the d_model")
            self.embed = None
        
        
        self.lstms = nn.LSTM(nmode,hidden_size,num_layer,batch_first=True )
        for name, param in self.lstms.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
                print(f"INFO: {name} has been initilised")
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
                print(f"INFO: {name} has been initilised")
                    

        if is_output:
            self.out = nn.Linear(hidden_size,nmode,
                                bias=is_output_bias)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            if out_act =="tanh":
                self.out_act = nn.Tanh() 
            elif out_act =="elu":
                self.out_act = nn.ELU()
            else:
                self.out_act = None 
            
    def forward(self,x):

        if self.embed != None:
            x            =   self.embed(x)
        
        hidden, cell =   self.init_hidden(x.shape[0],device=x.device)
        
        x, (hn,cn) = self.lstms(x,(hidden.detach(),cell.detach()))
        
    
        if self.is_output:
            if self.out_act is not None:
                return self.out_act(self.out(x[:,-self.out_dim:,:]))
            else:    
                return self.out(x[:,-self.out_dim:,:])
    def init_hidden(self,batch_size,device):
        hidden = torch.zeros(self.num_layer,
                                batch_size,
                                self.hidden_size).to(device)
                    
        cell  =  torch.zeros(self.num_layer,
                                batch_size,
                                self.hidden_size).to(device) 
                    
                    
        return hidden, cell