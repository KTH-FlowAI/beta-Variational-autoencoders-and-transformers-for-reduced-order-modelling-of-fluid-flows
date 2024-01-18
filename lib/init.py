"""
Initialisation and setup before running 
@yuningw
"""

class pathsBib: 
    
    data_path       = 'data/'
    model_path      = 'models/'; 
    res_path        = 'res/'
    fig_path        = 'figs/'
    log_path        = 'logs/'
    chekp_path      =  model_path + 'checkpoints/'
    pretrain_path   =  model_path + "pretrained/"

#-------------------------------------------------
def init_env(Re=40):
    """
    A function for initialise the path and data 

    Args:
        Re  :   The corresponding Reynolds number of the case   

    Returns:

        data_file   : (str)  File name
    
    """
    from configs.vae import VAE_config
    assert(Re == VAE_config.Re), print(f"ERROR: Please match the config of vae {VAE_config.Re}!")
    
    is_init_path = init_path()
    
    datafile = None

    if is_init_path:
        is_acquired, datafile = acquire_data(Re)
    else: 
        print(f"ERROR: Init Path failed!")
    
    return datafile


#-------------------------------------------------
def init_path():
    """
    Initialisation of all the paths 

    Returns:
        is_init_path    :   (bool) if initialise success
    """
    import os 
    from pathlib import Path
    
    is_init_path = False
    try:
        print("#"*30)
        print(f"Start initialisation of paths")
        path_list =[i for _,i in pathsBib.__dict__.items() if type(i)==str and "/" in i]
        print(path_list)
        for pth in path_list:
            # if "/" in pth:
            Path(pth).mkdir(exist_ok=True)
            print(f"INIT:\t{pth}\tDONE")
        print("#"*30)
        is_init_path = True
    except:
        print(f"Error: FAILD to inital path, Please check setup for your path!")

    return is_init_path


#-------------------------------------------------
def acquire_data(Re=40):

    """
    Acquisition of dataset from zendo
    
    Args:
        Re  :   The corresponding Reynolds number of the case   

    Returns:
        is_acquired : (bool) A flag for results 
        data_file   : (str)  File name
    """

    import urllib.request
    import os 
    import time
    is_acuqired = False
    datfile     = None
    if Re == 40:
        datfile = pathsBib.data_path + "Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5"
    else:
        print(f"Error: Data might be too large to download, please get it manually")

    try:
        if not os.path.exists(datfile):
            try:
                print(f"{datfile}")
                print(f"INFP: Not found, trying to download example dataset")
                st = time.time()
                urllib.request.urlretrieve('https://zenodo.org/records/10501216/files/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5?download=1', 
                                        datfile)
                et = time.time()
                ct = et - st 
                print(f"File downloaded successfully to {datfile}, Time Cost: {ct:.2f}s")
                is_acuqired = True
            
            except Exception as e:
                    print(f"Failed to download sample dataset. Error: {e}")
        else:
            print(f"INFO: Data exists in local!")
            
            is_acuqired = True

    except: 
        print(f"Error: Failed loading data")

    print("#"*30)
    
    return is_acuqired, datfile

