"""
Attention module used in Transformer 

Here we use the easy-attention mechanism for time-series prediction in latent space 
We use a stack of encoders only for the prediction, leveraging time-delay embedding 
"""
import torch
import torch.nn as nn 
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-head attention with mask 
        
        Args:   
            
            d_model     : The dimension of the embedding 
            
            num_heads   : Number of heads to be split 

        Funcs:
            __init__ 

            scaled_dot_product_attention(self, Q, K, V, mask): Implement the scaled-dot attention for each head 

            split_heads()                                    : Split the sequnce by the number of heads
            
            combine_heads()                                    : Combine the sequnce by the number of heads

            forward()                                          : Forward prop for the module                  


        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Basic data dimension
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.d_k            = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Implement the scaled dot attention 

        Args: 
            Q       :   Query projected by WQ   
            K       :   Key projected by WK   
            V       :   Value projected by WV   
            mask    :   Mask that implmented on attention score 

        Returns:
            output  :   The sequence that be encoded by attention mechanism 
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
        
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
     
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        Split the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S, N//H]
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        
        
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward prop of the attention module

        Args:
            Q, K, V     :   The input tensor used as Query, Key and Value
            mask        :   The mask used on attention score
        
        Returns:
            output      :   The encoded sequence by attention module

        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output




class DenseEasyAttn(nn.Module):
    
    def __init__(self,  d_model, 
                        seqLen,
                        num_head,) -> None:
        """
        Dense Easy attention mechansim used in transformer model for the time-series prediction and reconstruction
        
        Args:

            d_model     :   The embedding dimension for the input tensor 
            
            seqLen      :   The length of the sequence 

            num_head    :   The number of head to be used for multi-head attention
    
        """
        super(DenseEasyAttn,self).__init__()
     
        assert d_model % num_head == 0, "dmodel must be divible by number of heads"

        self.d_model    =   d_model
        self.d_k        =   d_model // num_head
        self.num_heads  =   num_head
        # Create the tensors
        self.Alpha      = nn.Parameter(torch.randn(size=(num_head,seqLen,seqLen),dtype=torch.float),requires_grad=True)            
        self.WV         = nn.Parameter(torch.randn(size=(d_model,d_model)       ,dtype=torch.float),requires_grad=True)               
        # Initialisation
        nn.init.xavier_uniform_(self.Alpha)
        nn.init.xavier_uniform_(self.WV)
    
    def split_heads(self, x):
        """
        Split the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S, N//H]
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
 
    def forward(self,x:torch.Tensor):   
        """
        Forward prop for the easyattention module 
        
        Following the expression:  x_hat    =   Alpha @ Wv @ x 

        Args:  
            
            self    :   The self objects

            x       :   A tensor of Input data
        
        Returns:
            
            x       :   The tensor be encoded by the moudle
        
        """
        # Obtain the value of batch size 
        B,_,_   =   x.shape
        # We repeat the tensor into same number of batch size that input has 
        Wv      =   self.WV.repeat(B,1,1)    
        # Implement matmul along the batch 
        V       =   torch.bmm(Wv,x)
        # Split heads for value
        V_h     =   self.split_heads(V)
        # Implement the learnable attention tensor 
        Alpha   =   self.Alpha.repeat(B,1,1,1)
        # Gain Attention by matrix multiplication
        x       =   Alpha @ V_h
        # Combine the output back to the original size
        x       =   self.combine_heads(x)
        
        return x





