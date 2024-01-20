"""
Post-processing and evaluation for time series prediction
"""

#####################################
# Assessment of temporal dynamic prediction 
####################################

#--------------------------------------------------------
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

#--------------------------------------------------------
def Sliding_Window_Error(test_data,
                        model, device,
                        in_dim,window = 5):
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
    for init in tqdm(range(in_dim,SeqLen-window,1)):
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

#--------------------------------------------------------
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


#--------------------------------------------------------
def make_physical_prediction(vae,
                            pred_latent,
                            true_latent,
                            device):
    """
    Reconstruct the latent-space prediction to the physical space
    
    Args: 

            vae         : (lib.runners.vaeRunner) The module for employed VAE     

            pred_latent : (Numpyrray) The latent-space prediction 

            true_latent : (NumpyArray) The latent-space ground truth from VAE

            device      : (str) The device for computation 

    Returns:

            rec_vae     : (NumpyArray) Reconstruction from reference latent variable

            rec_pred    : (NumpyArray) Reconstruction from the predicted latent variable
    """
    import numpy as np
    from lib.pp_space import decode

    rec_vae     =   decode(vae.model,true_latent,device)

    rec_pred    =   decode(vae.model,pred_latent,device)

    return rec_vae, rec_pred




#####################################
# Poincare Map for chaotic nature analysis 
####################################

"""
Implement Poincare Maps on test data 

@yuningw
"""

#--------------------------------------------------------
def Zero_Cross(data,postive_dir = True):    
    """
    Function to find the cross section betwee positive and negative of a 1D Vector
    Args:
        data        : 1D numpy arrary, object data
        postive_dir : bool, Find the positive direction or negative
    
    Returns:
        cross_idx   : Indices of the cross-section points 
        x_values    : Estimation of the position of the zero crossing 
                        within the interval between crossings_idx 
                        and crossings_idx_next

    """
    import numpy as np
    zero_crossings = np.where( np.diff(np.signbit(data)) )
    if (postive_dir):
        wherePos = np.where( np.diff(data) > 0 )
    else:
        wherePos = np.where( np.diff(data) < 0 )

    cross_idx      = np.intersect1d(zero_crossings, wherePos)
    cross_idx_next = cross_idx +1

    x_values       = cross_idx - data[list(cross_idx)]/\
                                        (data[list(cross_idx_next)] - data[list(cross_idx)])


    return cross_idx, x_values



#--------------------------------------------------------
def Intersection(data,planeNo = 0,postive_dir = True):
    """
    Compute the intersections of time-series data w.r.t each temporal mode

    Args:
        data        :   A 2D numpy array has shape of [Ntime, Nmodes]
        planeNo     :   Integar, the No. of plane to compute the intersections
        postive_dir :   bool, choose which direction     

    Returns:
        InterSec    : The intersection data in numpy array
    """
    import numpy as np 
    import sys
    if len(data.shape) !=2:
        print("The data should have 2 dimensions")
        sys.exit()

    SeqLen, Nmode               = data.shape[0],data.shape[-1]
    zero_cross, x_value = Zero_Cross(data        = data[:,planeNo], 
                                    postive_dir = postive_dir)

    # Create InterSec to store the results
    InterSec = np.zeros((zero_cross.shape[0],Nmode))
    for mode in range(0,Nmode):
        InterSec[:,mode] = np.interp(x_value, np.arange(SeqLen), data[:,mode])

    return InterSec



#--------------------------------------------------------
def PDF(InterSecX,InterSecY,
        xmin = -1,xmax = 1,x_grid = 50,
        ymin = -1,ymax = 1,y_grid = 50):

    """
    Compute the joint PDF of X and Y 
    Args:
        InterSecX   : numpy array of data 1
        InterSecY   : numpy array of data 2

        xmin, xmax, x_grid  :   The limitation of InterSecX and number of grid to be plot for contour 
        ymin, ymax, y_grid  :   The limitation of InterSecY and number of grid to be plot for contour 

    Returns:
        xx, yy: The meshgrid of InterSecX and InterSecY according to the limitation and number of grids
        pdf   : The joint pdf of InterSecX and InterSecY 
    """
    import numpy as np 
    import scipy.stats as st 
    # Create meshgrid acorrding 
    xx, yy = np.mgrid[xmin:xmax:1j*x_grid, ymin:ymax:1j*y_grid]

    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([InterSecX, InterSecY])
    kernel = st.gaussian_kde(values)
    pdf = np.reshape(kernel(positions).T, xx.shape)

    return xx,yy,pdf