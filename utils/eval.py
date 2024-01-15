import torch
import numpy as np
import h5py

import data  as lib_data
import model as lib_model
import fig   as  lib_figures

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


def decode(model, data, device):

    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=512,
                                          shuffle=False, num_workers=2)
    rec_list = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            rec_list.append(model.decoder(batch).cpu().numpy())

    return np.concatenate(rec_list, axis=0)


def get_spatial_modes(model, latent_dim, mean, std, device):

    def calcmode(model, latent_dim, mode):
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


def get_Ek(original, rec):
    """Calculate energy percentage reconstructed"""
    #u_real = original[:, 0, :, :]
    #v_real = original[:, 1, :, :]
    #TKE_real = u_real ** 2 + v_real ** 2
    TKE_real = original[:, 0, :, :] ** 2 + original[:, 1, :, :] ** 2

    u_rec = rec[:, 0, :, :]
    v_rec = rec[:, 1, :, :]

    return 1 - np.sum((original[:, 0, :, :] - u_rec) ** 2 + (original[:, 1, :, :] - v_rec) ** 2) / np.sum(TKE_real)


def get_order(model, latent_dim, data, dataset, std, device):
    import time

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


def get_EcumTest(model, latent_dim, data, dataset, std, device, order):
    modes, _ = encode(model, dataset, device)
    print(modes.shape)
    u = data[:, :, :, :] * std[:, :, :, :]

    Ecum = []
    for i in range(latent_dim):
        partialModes = np.zeros_like(modes, dtype=np.float32)
        partialModes[:, order[:i]] = modes[:, order[:i]]
        u_pred = decode(model, partialModes, device)
        u_pred *= std[:, :, :, :]
        Ecum.append(get_Ek(u, u_pred))
        print(order[:i+1], Ecum[-1])

    return np.array(Ecum)

def createModesFile(fname, model, latent_dim, dataset_train, dataset_test, mean, std, device, order, Ecum, Ecum_test, NLvalues, NLmodes):
    # encode
    means_train, _ = encode(model, dataset_train, device)
    means_test, _ = encode(model, dataset_test, device)
    print(f'Train: {means_train.shape}, test: {means_test.shape}')

    zero_output, modes = get_spatial_modes(model, latent_dim, mean, std, device)

    print(fname)

    if order is None:
        order = np.arange(latent_dim)

    with h5py.File(fname, 'w') as f:
        f.create_dataset('vector', data=means_train)
        f.create_dataset('vector_test', data=means_test)
        f.create_dataset('modes', data=modes)
        f.create_dataset('zero_output', data=zero_output)
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        f.create_dataset('order', data=order)
        f.create_dataset('Ecum', data=Ecum)
        f.create_dataset('Ecum_test', data=Ecum_test)
        f.create_dataset('NLvalues', data=NLvalues)
        f.create_dataset('NLmodes', data=NLmodes)

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
    print('Calc NL modes')
    # decode(model, data, device)

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



if __name__ == "__main__":

    # weights file
    weights_files_list = ['03_checkpoints/20230718_11_31_beta0.05_dim20_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_bestTest.pth.tar']

    # Get system info
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))
    device = torch.device('cuda')

    # load data
    # datafile = '01_data/Re100alpha10_newData_v1.hdf5'
    datafile = '01_data/Re100alpha10_newData_85000.hdf5'

    # parameters
    batch_size = 256
    n_test = 5000

    u_scaled, mean, std = lib_data.loadData(datafile)
    n_total = u_scaled.shape[0]
    n_train = n_total - n_test
    print(f"N train: {n_train:d}, N test: {n_test:d}, N total {n_total:d}")

    dataset_train = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[:n_train]), batch_size=batch_size,
                                                shuffle=False, pin_memory=True, num_workers=2)
    dataset_test = torch.utils.data.DataLoader(dataset=torch.from_numpy(u_scaled[n_train:]), batch_size=batch_size,
                                               shuffle=False, pin_memory=True, num_workers=2)

    for weights_file in weights_files_list:
        out_name = weights_file.split('.pth')[0].split('/')[-1]
        out_name = '04_modes/' + out_name + '_modes.h5py'

        latent_dim = int(weights_file.split('dim')[-1].split('_')[0])
        print(f'Latent dim: {latent_dim:2d}')

        # Get model
        model = lib_model.VAE(latent_dim=latent_dim).to(device)
        model.eval()

        # Load weights
        lib_model.load_checkpoint(model=model, path_name=weights_file)

        #create samples
        #rec_train, rec_test, true_train, true_test = get_samples(model, dataset_train, dataset_test, device)
        #lib_figures.plotSamples(rec_train, rec_test, true_train, true_test)

        # get Ek evolution in test
        #Ek_test = get_Ek_t(model, u_scaled[n_train:], device)
        #lib_figures.plotEk_t(Ek_test)

        try:
            with h5py.File(out_name, 'r') as f:
                order = f['order'][:]
                Ecum = f['Ecum'][:]
            print('Loaded order from ', out_name)

        except:
            print('Order not found, generating: ', out_name)
            # sort modes
            order, Ecum = get_order(model, latent_dim, u_scaled[:n_train], dataset_train, std, device)

        print('NL mode: ', order[0])
        NLvalues, NLmodes = getNLmodes(model, order[0], latent_dim, device)

        Ecum_test = get_EcumTest(model, latent_dim, u_scaled[n_train:], dataset_test, std, device, order)

        # Create modes file
        createModesFile(out_name, model, latent_dim, dataset_train, dataset_test, mean, std, device, order, Ecum, Ecum_test, NLvalues, NLmodes)