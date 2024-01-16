"""
Implement Poincare Maps on test data 

@yuningw
"""

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