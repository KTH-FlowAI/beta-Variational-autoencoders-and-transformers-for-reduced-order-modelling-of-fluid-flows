import torch
import numpy as np
import h5py



#--------------------------------------------------------
def encode(model, data, device):
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

    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=512,
                                        shuffle=False, num_workers=2)
    rec_list = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            rec_list.append(model.decoder(batch).cpu().numpy())

    return np.concatenate(rec_list, axis=0)



#--------------------------------------------------------
def get_spatial_modes(model, latent_dim, device):

    """
    Algorithm for optain the spatial mode from beta-VAE Decoder. 
    For latent variable i, We use the unit vector v (where vi = 1) as the input to obtain the spatial mode for each latent variables
    
    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        latent_dim      : (int) The latent dimension we employed

        device          : (str) The device for the computation
    
    Returns:

        
        
    """
    def calcmode(model, latent_dim, mode):
        """
        
        Generate the non-linear mode with unit vector


        
        """
        
        z_sample = np.zeros((1, latent_dim), dtype=np.float32)
        z_sample[:, mode] = 1
        with torch.no_grad():
            mode = model.decoder(torch.from_numpy(z_sample).to(device)).cpu().numpy()
        
        return mode

    with torch.no_grad():
        zero_output = model.decoder(torch.from_numpy(np.zeros((1, latent_dim), dtype=np.float32)).to(device)).cpu().numpy()

    modes = np.zeros((latent_dim, zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]))

    for mode in range(latent_dim):
        modes[mode, :, :, :] = calcmode(model, latent_dim, mode)

    return zero_output, modes



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
def get_order(model, latent_dim, data, dataset, std, device):
    """
    

    
    """
    
    
    import time
    import numpy as np 


    print('############################################')
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
        print('############################################')
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {m}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(m), Ecum




#--------------------------------------------------------
def get_EcumTest(model, latent_dim, data, dataset, std, device, order):
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




#--------------------------------------------------------
def createModesFile(fname, model, latent_dim, dataset_train, dataset_test, mean, std, device, order, Ecum, Ecum_test, NLvalues, NLmodes, Ek_t):
    # encode
    means_train, stds_train = encode(model, dataset_train, device)
    means_test, stds_test = encode(model, dataset_test, device)
    print(f'Train: {means_train.shape}, test: {means_test.shape}')

    zero_output, modes = get_spatial_modes(model, latent_dim, mean, std, device)

    print(fname)

    if order is None:
        order = np.arange(latent_dim)

    with h5py.File(fname, 'w') as f:
        # mu
        f.create_dataset('vector', data=means_train)
        f.create_dataset('vector_test', data=means_test)
        
        # Sigma
        f.create_dataset('stds_vector', data=stds_train)
        f.create_dataset('stds_vector_test', data=stds_test)

        #
        f.create_dataset('modes', data=modes)
        f.create_dataset('zero_output', data=zero_output)
        
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        
        f.create_dataset('order', data=order)


        f.create_dataset('Ecum', data=Ecum)
        f.create_dataset('Ecum_test', data=Ecum_test)
        
        f.create_dataset('NLvalues', data=NLvalues)
        f.create_dataset('NLmodes', data=NLmodes)
        
        f.create_dataset('Ek_t', data=Ek_t)



def get_samples(model, dataset_train, dataset_test, device):

    with torch.no_grad():

        for batch_train in dataset_train:
            batch_train = batch_train.to(device, non_blocking=True)
            rec_train, _, _ = model(batch_train)
            break
        for batch_test in dataset_test:
            batch_test = batch_test.to(device, non_blocking=True)
            rec_test, _, _ = model(batch_test)
            break

        rec_train = rec_train.cpu().numpy()[-1]
        rec_test = rec_test.cpu().numpy()[-1]
        true_train = batch_train.cpu().numpy()[-1]
        true_test = batch_test.cpu().numpy()[-1]

        return rec_train, rec_test, true_train, true_test

def get_Ek_t(model, data, device):

    dataloader = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=batch_size,
                                            shuffle=False, pin_memory=True, num_workers=2)

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

def getNLmodes(model, mode, latent_dim, device):


    zero_output = decode(model, np.zeros((1, latent_dim), dtype=np.float32), device)
    #print(zero_output.shape)

    NLvalues = np.arange(-2, 2.1, .1)

    NLmodes = np.zeros((NLvalues.shape[0], zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]),
                        dtype=np.float32)
    #print(NLmodes.shape)

    for idx, value in enumerate(NLvalues):
        #print(value, idx)
        latent = np.zeros((1, latent_dim), dtype=np.float32)
        latent[0, mode] = value
        NLmodes[idx,:,:,:] = decode(model, latent, device)

    return NLvalues, NLmodes
