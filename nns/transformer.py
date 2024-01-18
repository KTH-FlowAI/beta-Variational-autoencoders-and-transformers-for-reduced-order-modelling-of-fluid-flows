"""
The transfomer encoders using new embeding layer

"""

from nns.layers           import EncoderLayer, easyEncoderLayer
from nns.embedding        import TimeSpaceEmbedding
from    torch                   import nn

class EmbedTransformerEncoder(nn.Module):


    def __init__(self, d_input, d_output, n_mode, 
                        d_proj, d_model, d_ff,
                        num_head,num_layer,
                        act_proj= "relu",
                        dropout= 1e-5) -> None:
        """
    A transformer-based architecture using temporal-spatial embedding and a stack of encoder 
    here we use self attention as the attention mechanism 
        """
        super(EmbedTransformerEncoder,self).__init__()

        self.embed      =   TimeSpaceEmbedding(d_input, n_mode, d_proj, d_model)

        self.encoders   =   nn.ModuleList([ EncoderLayer(d_model= d_model, num_heads=num_head ,d_ff = d_ff,act_proj=act_proj,dropout=dropout ) for _ in range(num_layer)]) 

        self.cf         =   nn.Conv1d(d_proj, d_output,1)
        
        self.of         =   nn.Linear(d_model,n_mode)

        nn.init.xavier_uniform_(self.cf.weight)
        nn.init.xavier_uniform_(self.of.weight)
        nn.init.zeros_(self.cf.bias)
        nn.init.zeros_(self.of.bias)

    def forward(self, src):
        enc_input   = self.embed(src)
        # Leave the residual for forward porp
        enc_res     = 0
        for enc_layer in self.encoders:
            enc_input   = enc_layer(enc_input+enc_res,None)  
            enc_res     = enc_input  

        x   =   self.cf(enc_input)
        x   =   self.of(x)

        return x 



class easyTransformerEncoder(nn.Module):
    """
    A transformer-based architecture using temporal-spatial embedding and a stack of encoder 
    here we use self attention as the attention mechanism
    """

    def __init__(self, d_input, d_output, seqLen, 
                        d_proj, d_model, d_ff,
                        num_head,num_layer,
                        act_proj= "relu",
                        dropout= 1e-5) -> None:
        super(easyTransformerEncoder,self).__init__()

        self.embed      =   TimeSpaceEmbedding(d_input, seqLen, d_proj, d_model)

        self.encoders   =   nn.ModuleList([ easyEncoderLayer(d_model   =   d_model, 
                                                            seqLen     =   d_proj,
                                                            num_heads  =   num_head ,
                                                            d_ff       =   d_ff,
                                                            act_proj   =   act_proj,
                                                            dropout    =   dropout ) for _ in range(num_layer)]) 

        self.cf         =   nn.Conv1d(d_proj, d_output,1)
        
        self.of         =   nn.Linear(d_model,seqLen)

        nn.init.xavier_uniform_(self.cf.weight)
        nn.init.xavier_uniform_(self.of.weight)
        nn.init.zeros_(self.cf.bias)
        nn.init.zeros_(self.of.bias)

    def forward(self, src):
        enc_input   = self.embed(src)
        # print(enc_input.shape)
        # Leave the residual for forward porp

        for enc_layer in self.encoders:
            enc_input   = enc_layer(enc_input)  

        x   =   self.cf(enc_input)
        # print(x.shape)
        x   =   self.of(x)

        return x 
