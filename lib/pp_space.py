"""
Post-processing and analysis algorithm for beta-VAE in latent space and physic space

author :    @alsora
editting:   @yuningw

"""

import torch
import numpy as np
import h5py
from lib.init import pathsBib

################################
### Main programme for spatial analysis
###############################
def spatial_Mode(   fname,
                    model, 
                    latent_dim, 
                    train_data,
                    test_data,
                    dataset_train,
                    dataset_test,
                    mean, std, 
                    device,
                    if_order    = True,
                    if_nlmode   = True,
                    if_Ecumt    = True,
                    if_Ek_t     = True,
                ): 
    """
    The main function for spatial mode analysis and generate the dataset 
        
    Args:

        fname           :   (str) The file name

        latent_dim      :   (int) The latent dimension 

        train_data      :   (NumpyArray) Dataset for training
        
        test_data       :   (NumpyArray) Dataset for test

        dataset_train   :   (dataloader) DataLoader for training data
        
        dataset_test    :   (dataloader) DataLoader for test data

        mean            :   (NumpyArray) The mean of flow database
        
        std             :   (NumpyArray) The std of flow database

        device          :   (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
        Ecum_test       : (NumpyArray) accumlative Energy obtained for each mode

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode

        Ek_t            : (List) A list of enery of each snapshot in dataset

        if_order        : (bool) IF ranking the mode

        if_nlmode       : (bool) IF generate non-linear mode

        if_Ecumt        : (bool) IF compute accumulative energy 
        
        if_Ek_t         : (bool) IF compute evolution of energy 
    
    Returns: 

        if_save         : (bool) If successfully save file

    """
    
    print(f"INFO: Start spatial mode generating")
    if if_order:
        order, Ecum = get_order(model, latent_dim, 
                                        train_data, 
                                        dataset_train, 
                                        std, device)
        print(f"INFO: RANKING DONE")        
    else:
        order = None
        Ecum  = None
    

    if if_nlmode:
        NLvalues, NLmodes = getNLmodes(model, order[0], latent_dim, device)
        print("INFO: Non-linear mode generated")
    else:
        NLmodes = None
        NLvalues = None
    
    if if_Ecumt: 
        
        Ecum_test = get_EcumTest(model, latent_dim, test_data, dataset_test, std, device, order)
        print('INFO: Test E_cum generated')
    else: 
        Ecum_test = None
    
    if if_Ek_t: 
        Ek_t = get_Ek_t(model=model, data=test_data, device=device)

    else:
        Ek_t = None
    
    is_save = createModesFile(fname, model, latent_dim, 
                            dataset_train, dataset_test,
                            mean, std, device,
                            order, Ecum, Ecum_test,
                            NLvalues, NLmodes,Ek_t)
    
    if is_save: print("INFO: Successfuly DONE!")

    return is_save

################################
### Basic function for using VAE
###############################

#--------------------------------------------------------
def encode(model, data, device):
    """
    Use encoder to compress flow field into latent space 
    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        data        :   (DataLoader) DataLoader of data to be encoded 

        device      : (str) The device for the computation

    Returns: 

        means       : (NumpyArray) The mu obtained in latent space     
        
        logvars     : (NumpyArray) The sigma obtained in latent space
    """
    mean_list = []
    logvar_list = []
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device, non_blocking=True)
            mean, logvariance = torch.chunk(model.encoder(batch), 2, dim=1)

            mean_list.append(mean.cpu().numpy())
            logvar_list.append(logvariance.cpu().numpy())

    means = np.concatenate(mean_list, axis=0)
    logvars = np.concatenate(logvar_list, axis=0)

    return means, logvars


#--------------------------------------------------------
def decode(model, data, device):
    """
    Use decoder to reconstruct flow field back to physical space 

    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        data        :   (NumpyArray) The latent vectors required to be reconstructed 

        device      :   (str) The device for the computation

    Returns: 

        rec         :   (NumpyArray) The reconstruction of the flow fields. 

    """
    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=512,
                                        shuffle=False, num_workers=2)
    rec_list = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            rec_list.append(model.decoder(batch).cpu().numpy())

    return np.concatenate(rec_list, axis=0)



def get_samples(model, dataset_train, dataset_test, device):
    """
    
    A function for quickly obtain a restructed flow field for the propose of testing or visualisation

    We obtain snapshot through training and test data, respectively 

    Args: 
        model                :   (nn.Module) Pytorch module for beta-VA
        
        dataset_train        :   (DataLoader) DataLoader of training data 
        
        dataset_test         :   (DataLoader) DataLoader of test data 

        device               :   (str) The device for the computation

    Returns: 

        rec_train            :   (NumpyArray) The reconstruction from training dataset. 
        
        rec_test             :   (NumpyArray) The reconstruction from test dataset. 
        
        true_train           :   (NumpyArray) The corresponding ground truth from training dataset. 
        
        true_test            :   (NumpyArray) The corresponding ground truth from test dataset. 
    """
    with torch.no_grad():
        if dataset_train != None:
            for batch_train in dataset_train:
                batch_train = batch_train.to(device, non_blocking=True)
                rec_train, _, _ = model(batch_train)
        
                rec_train = rec_train.cpu().numpy()[-1]
                true_train = batch_train.cpu().numpy()[-1]
        
                break
        else: 
            rec_train   = None
            true_train  = None
        
        if dataset_test != None:
            for batch_test in dataset_test:
                batch_test = batch_test.to(device, non_blocking=True)
                rec_test, _, _ = model(batch_test)

                rec_test = rec_test.cpu().numpy()[-1]
                true_test = batch_test.cpu().numpy()[-1]

                break
        else: 
            rec_test    = None
            true_test   = None

        return rec_train, rec_test, true_train, true_test


################################
### Spatial-mode generate and analysis
###############################

#--------------------------------------------------------
def calcmode(model, latent_dim, mode, device):
    """
        
    Generate the non-linear mode with unit vector 

    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        latent_dim  :   (int) Latent-dimension adpot for beta-VA
        
        mode        :   (int) The indice of the mode zi

    Returns: 

        mode        : (NumpyArray) The spatial mode for zi     
    """

    z_sample = np.zeros((1, latent_dim), dtype=np.float32)

    z_sample[:, mode] = 1

    with torch.no_grad():
        mode = model.decoder(torch.from_numpy(z_sample).to(device)).cpu().numpy()
        
    return mode


#--------------------------------------------------------
def get_spatial_modes(model, latent_dim, device):

    """
    Algorithm for optain the spatial mode from beta-VAE Decoder. 
    For latent variable i, We use the unit vector v (where vi = 1) as the input to obtain the spatial mode for each latent variables
    Also, we compute the spatial mode use zeros vectors as input 

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        latent_dim      : (int) The latent dimension we employed

        device          : (str) The device for the computation
    
    Returns:

        modes           : The
        
    """


    with torch.no_grad():
        zero_output = model.decoder(torch.from_numpy(np.zeros((1, latent_dim), dtype=np.float32)).to(device)).cpu().numpy()

    modes = np.zeros((latent_dim, zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]))

    for mode in range(latent_dim):
        modes[mode, :, :, :] = calcmode(model, latent_dim, mode, device)

    return zero_output, modes


#--------------------------------------------------------
def getNLmodes(model, mode, latent_dim, device):
    """
    Algorithm for optain single spatial mode from beta-VAE Decoder. 
    
    For latent variable i, We use the vector v (where vi = 1)  with a value within a range
    as the input to obtain the spatial mode for each latent variables

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        mode            : (int) The indice of the mode zi

        latent_dim      : (int) The latent dimension we employed

        device          : (str) The device for the computation
    
    Returns:

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode
        
    """

    zero_output = decode(model, np.zeros((1, latent_dim), dtype=np.float32), device)
    NLvalues = np.arange(-2, 2.1, .1)
    NLmodes = np.zeros((NLvalues.shape[0], zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]),
                        dtype=np.float32)
    
    for idx, value in enumerate(NLvalues):
        latent = np.zeros((1, latent_dim), dtype=np.float32)
        latent[0, mode] = value
        NLmodes[idx,:,:,:] = decode(model, latent, device)

    return NLvalues, NLmodes


#--------------------------------------------------------
def get_order(  model, latent_dim, 
                data, dataset, 
                std, device):
    """
    Algorithm for ranking the obtained spatial modes according to the yield accumlated energy level
    For more detail please check the paper

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        
        latent_dim      : (int) The latent dimension we employed

        data            : (NumpyArray) The flow database 

        dataset         : (torch.Dataloader) The dataloader of the flow data

        std             : (NumpyArray) The std of flow database

        device          : (str) The device for the computation
    
    Returns:

        m               : (NumpyArray) The ranking result (order) of each mode

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
    """
    import time
    import numpy as np 
    print('#'*30)
    print('Ordering modes')

    modes, _ = encode(model, dataset, device)

    print(modes.shape)
    u = data[:, :, :, :] * std[:, :, :, :]
    m = np.zeros(latent_dim, dtype=int)
    n = np.arange(latent_dim)

    Ecum = []
    partialModes = np.zeros_like(modes, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        for j in n:  # for mode in remaining modes
            start = time.time()
            print(m[:i], j, end="")
            partialModes *= 0
            partialModes[:, m[:i]] = modes[:, m[:i]]
            partialModes[:, j] = modes[:, j]
            u_pred = decode(model, partialModes, device) * std
            Eks.append(get_Ek(u, u_pred))
            elapsed = time.time() - start
            print(f' : Ek={Eks[-1]:.4f}, elapsed: {elapsed:.2f}s')
        Eks = np.array(Eks).squeeze()
        ind = n[np.argmax(Eks)]
        m[i] = ind
        n = np.delete(n, np.argmax(Eks))
        Ecum.append(np.max(Eks))
        print('Adding: ', ind, ', Ek: ', np.max(Eks))
        print('#'*30)
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {m}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(m), Ecum



################################
### Assessment on Energy
###############################
#--------------------------------------------------------
def get_Ek(original, rec):
    
    """
    Calculate energy percentage reconstructed
    
    Args:   
            original : (NumpyArray) The ground truth 

            rec      : (NumpyArray) The reconstruction from decoder

    Returns:  

            The energy percentage for construction. Note that it is the Ek/100 !!
    """

    import numpy as np 

    TKE_real = original[:, 0, :, :] ** 2 + original[:, 1, :, :] ** 2

    u_rec = rec[:, 0, :, :]
    v_rec = rec[:, 1, :, :]

    return 1 - np.sum((original[:, 0, :, :] - u_rec) ** 2 + (original[:, 1, :, :] - v_rec) ** 2) / np.sum(TKE_real)



#--------------------------------------------------------
def get_Ek_t(model, data, device):
    """
    
    Get the Reconstructed energy for snapshots

    Args:

        model           : (torch.nn.Module) The beta-VAE model 
        
        data            : (NumpyArray) The flow database 

        device          : (str) The device for the computation
    
    Returns:

        Ek_t            : (List) A list of enery of each snapshot in dataset
    
    
    """

    dataloader = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), 
                                            batch_size=1,
                                            shuffle=False, 
                                            pin_memory=True, 
                                            num_workers=2)

    rec_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            rec, _, _ = model(batch)
            rec_list.append(rec.cpu().numpy())

    rec = np.concatenate(rec_list, axis=0)

    print(rec.shape)
    print(data.shape)

    Ek_t = np. zeros((rec.shape[0]))
    for i in range(rec.shape[0]):
        Ek_t[i] = get_Ek(data[np.newaxis, i], rec[np.newaxis, i])

    return Ek_t





#--------------------------------------------------------
def get_EcumTest(model, latent_dim, data, dataset, std, device, order):

    """
    Get the accumlative energy of test database 

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        
        latent_dim      : (int) The latent dimension we employed

        data            : (NumpyArray) The flow database 

        dataset         : (torch.Dataloader) The dataloader of the flow data

        std             : (NumpyArray) The std of flow database

        device          : (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

    Returns:

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
    """

    modes, _ = encode(model, dataset, device)
    print(modes.shape)
    u = data[:, :, :, :] * std[:, :, :, :]

    Ecum = []
    for i in range(latent_dim):
        partialModes = np.zeros_like(modes, dtype=np.float32)
        partialModes[:, order[:i+1]] = modes[:, order[:i+1]]
        u_pred = decode(model, partialModes, device)
        u_pred *= std[:, :, :, :]
        Ecum.append(get_Ek(u, u_pred))
        print(order[:i+1], Ecum[-1])

    return np.array(Ecum)





################################
### I/O 
###############################
#--------------------------------------------------------
def createModesFile(fname, 
                    model, 
                    latent_dim, 
                    dataset_train, dataset_test, 
                    mean, std, 
                    device, 
                    order, 
                    Ecum, Ecum_test, 
                    NLvalues, NLmodes, 
                    Ek_t):
    """
    
    Function for integrating all the obtained results and save it as fname

    Args: 

        fname           :   (str) The file name

        latent_dim      :   (int) The latent dimension 

        dataset_train   :   (dataloader) DataLoader for training data
        
        dataset_test    :   (dataloader) DataLoader for test data

        mean            :   (NumpyArray) The mean of flow database
        
        std             :   (NumpyArray) The std of flow database

        device          :   (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
        Ecum_test       : (NumpyArray) accumlative Energy obtained for each mode

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode

        Ek_t            : (List) A list of enery of each snapshot in dataset

    Returns:

        is_save         : (bool)

    """
    
    if_save = False

    print(f"Start post-processing")


    means_train, stds_train  =   encode(model, dataset_train, device)
    print('INFO: Latent Variable Train Generated')
    means_test, stds_test    =   encode(model, dataset_test, device)
    print('INFO: Latent Variable Test Generated')
    
    zero_output, modes = get_spatial_modes(model, latent_dim, device)
    print('INFO: Spatial mode generated')

    if order is None:
        order = np.arange(latent_dim)


    with h5py.File(fname + ".hdf5", 'w') as f:

            # Mean velocity of flow data 
            f.create_dataset('mean', data=mean)
            # std of flow data
            f.create_dataset('std', data=std)

            # mu
            f.create_dataset('vector', data=means_train)
            f.create_dataset('vector_test', data=means_test)
            
            # Sigma
            f.create_dataset('stds_vector', data=stds_train)
            f.create_dataset('stds_vector_test', data=stds_test)

            # Spatial modes: Unit vector input 
            f.create_dataset('modes', data=modes)
            # Sptail modes: zeros vector input 
            f.create_dataset('zero_output', data=zero_output)
            # Spatial modes: Non-linear value adopted 
            f.create_dataset('NLvalues', data=NLvalues)
            # Spatial modes: Non-linear modes
            f.create_dataset('NLmodes', data=NLmodes)

            # Ranking results for spatial modes
            f.create_dataset('order', data=order)

            #Assessments: Accumlative energy 
            f.create_dataset('Ecum', data=Ecum)
            #Assessments: Accumlative energy on test dataset
            f.create_dataset('Ecum_test', data=Ecum_test)   
            #Assessments: E_k for each single snapshots 
            f.create_dataset('Ek_t', data=Ek_t)

    f.close()

    # Extra file for time-series prediction.
    with h5py.File(pathsBib.data_path + 'latent_data' + ".h5py", 'w') as f:
            # mu
            f.create_dataset('vector', data=means_train)
            f.create_dataset('vector_test', data=means_test)
    f.close()


    if_save = True

    
    print(f"INFO: Post-processing results has been saved as dataset: {fname}")

    return if_save