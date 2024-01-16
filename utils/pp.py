"""
Post-processing and evaluation for time series prediction
"""
def make_Prediction(test_data, model,
                    device, in_dim, next_step):
    """
    Function for generat the prediction data 
    
    Args:
        test_data   :  A numpy array of test data, with shape of [Ntime, Nmode]
        model       :  A torch.nn.Module object as model
        device      :  String of name of device    
        in_dim      :  Integar of TimeDelay size
        next_step   :  Future time step to predict
    
    Returns:
        preds    : A numpy array of the prediction  
    """
    from copy import deepcopy
    import torch 
    from tqdm import tqdm


    
    model.eval()
    model.to(device)
    Preds  = deepcopy(test_data)
    seq_len = max([Preds.shape[0],Preds.shape[1]])
    print(f"The sequence length = {seq_len}")

    for i in tqdm(range(in_dim,seq_len-next_step)):
        
        feature = Preds[None,i-in_dim:i,:]

        x = torch.from_numpy(feature)
        x = x.float().to(device)
        pred = model(x)

        pred = pred.cpu().detach().numpy()

        Preds[i:i+next_step,:] = pred[0,:,:]

    return Preds


def Sliding_Window_Error(test_data,
                        model, device,
                        in_dim,window = 100):
    """
    Compute the sliding window error on test dataset
    Args:
        test_data   : A numpy array of test data [Ntime, Nmode]
        model       : A torch.nn.Module as model 
        device      : String of name of device
        in_dim      : Integar of input dimension
        window      : The size of window for evaluation, default = 100 
    
    Returns:
        error_l2    : A numpy arrary of sliding window error, shape = [window,]
    
    """
    import torch
    import copy
    from tqdm import tqdm
    import numpy as np 

    def l2norm(predictions, targets):
        return np.sqrt( np.sum( (predictions - targets) ** 2, axis=1 ) )
    
    model.to(device)
    model.eval()

    SeqLen = test_data.shape[0]
    error = None
    for init in tqdm(range(in_dim,SeqLen-window,2)):
        temporalModes_pred = copy.deepcopy(test_data)

        for timestep in range(init, init+window):

            data    = temporalModes_pred[None, (timestep-in_dim):timestep, :]
            feature = torch.from_numpy(data).float().to(device)
            pred    = model(feature)
            temporalModes_pred[timestep,:] = pred[0].detach().cpu().numpy()

        if error is None:
            error = l2norm(temporalModes_pred[init:init+window,:], test_data[init:init+window,:])
            n = 1
        else:
            error = error + l2norm(temporalModes_pred[init:init+window,:], test_data[init:init+window,:])
            n += 1

    print(n)
    error_l2 = error / n 
    print(error.shape)

    return error_l2


def l2Norm_Error(Pred, Truth):
    """
    Compute the l2-norm proportional error for the prediction of temporal evolution
    Args:
        Pred    :   Numpy array has shape of [Ntime, Nmode] 
        Truth   :   Numpy array has shape of [Ntime, Nmode]
    
    Returns:
        error   :   A vector of error for each mode
    """

    import sys
    import numpy as np 
    from numpy.linalg import norm
    
    
    if Pred.shape != Truth.shape:
        print("The size of array does not match!")
        sys.exit()

    error = norm((Pred-Truth),axis=1)\
                            /(norm(Truth,axis=1))
    
    return error.mean()*100