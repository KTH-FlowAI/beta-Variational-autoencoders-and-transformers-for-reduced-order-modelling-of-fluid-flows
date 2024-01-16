import numpy as np
import time
import h5py
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
# from scipy.linalg import svd
import lib_data
import torch.nn as nn
import torch
#%%
#######################################################
########         PARAMETERS           #################
#######################################################

case = 'twoPlatesRe100_tc1'
DEBUG_MODE = 0
n_modes = 80
Load_original_data = 1

if case == 'wake_v2':
    datafile = '01_data/airfoilLES_midspan_v2_16000.hdf5'
    dt = 1
    n_test = 2000//dt #//dt; means integer division. to be consistent with dt.
elif case == 'TBL':
    datafile = '01_data/BL_DNS1_3D_2000f_64p.hdf5'
    dt = 1
    n_test = 10000//dt #//dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_tc1':
    datafile = '01_data/Re100alpha10_newData_150000.hdf5'
    dt = 5 # 5 means 1 tc time step
    n_test = 15000//dt #//dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_tc0.2':
    datafile = '01_data/Re100alpha10_newData_150000.hdf5'
    dt = 1  # 5 means 1 tc time step
    n_test = 15000 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe40':
    datafile = '01_data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5'
    dt = 1
    n_test = 100//dt #//dt; means integer division. to be consistent with dt.
elif case == 'expTBL600':
    datafile = '01_data/BLexp_Re600_6170.hdf5'
    dt = 1
    n_test = 1000 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_reconstructed':
    datafile = '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_reconstruction.h5py'
    dt = 1
    n_test = 15000 // dt  # //dt; means integer division. to be consistent with dt.

#%%
#######################################################
########         FUNCTIONS            #################
#######################################################

# snapshot_POD_overTime:   
# Use when number of time points is much smaller 
# than the resolution of a single snapshot
# (Np>>Nt)
# takes:
    #           u: U velocity field complete component
    #           v: V velocity field complete component
    #           n_modes: energy modes to be computed/extracted
# returns:
#           Phi_x.T: spatial modes of U, up only to n_modes
#           Phi_y.T: spatial modes of V, up only to n_modes
#           psi: temporal modes of both U and V times Sigma, up only to n_modes
#           eigVal: eigenvalues of the modes, up only to n_modes
def snapshot_POD_overTime(u, v, n_modes):
    u_t = u.T
    v_t = v.T

    print(f'u shape: {u.shape}')
    print(f'v shape: {v.shape}')

    # C matrix
    print('Calc C matrix = X*X.T', end="")
    start = time.time()
    C = u.dot(u_t) + v.dot(v_t)
    C = C / (n_train - 1)
    print(f': {(time.time() - start):.1f}s')
    print(f'C shape: {C.shape}')

    # SVD
    print('Calc SVD', end="")
    start = time.time()
    psi, eigVal2, _ = randomized_svd(C, n_components=n_modes, random_state=0)
    # psi, eigVal, _ = svd(C)
    print(f': {(time.time() - start):.1f}s')

    # print(f'psi shape: {psi.shape}')
    # print(f'eigVal2 shape: {eigVal2.shape}')

    Phi_x = u_t.dot(psi)
    Phi_y = v_t.dot(psi)

    return Phi_x, Phi_y, psi, np.sqrt(eigVal2)

# snapshot_POD_overSpace:
# Use when the resolution of a single snapshot
# is much smaller than the number of time points
# (Nt>>Np)
# takes:
    #           u: U velocity field complete component
    #           v: V velocity field complete component
    #           n_modes: energy modes to be computed/extracted
# returns:
#           Phi_x.T: spatial modes of U, up only to n_modes
#           Phi_y.T: spatial modes of V, up only to n_modes
#           psi: temporal modes of both U and V times Sigma, up only to n_modes
#           eigVal: eigenvalues of the modes, up only to n_modes
def snapshot_POD_overSpace(u, v, n_modes):

    print(f'u shape: {u.shape}')
    print(f'v shape: {v.shape}')

    X = np.concatenate((u, v), axis=1)
    print(f'X shape: {X.shape}')
    # C matrix
    print('Calc C matrix = X.T*X', end="")
    start = time.time()
    C = (X.T).dot(X)
    C = C / (n_train - 1)
    print(f': {(time.time() - start):.1f}s')
    print(f'C shape: {C.shape}')

    # SVD
    print('Calc SVD', end="")
    start = time.time()
    phi, eigVal2, _ = randomized_svd(C, n_components=n_modes, random_state=0)
    # psi, eigVal, _ = svd(C)
    print(f': {(time.time() - start):.1f}s')

    # print(f'phi shape: {phi.shape}')
    # print(f'eigVal2 shape: {eigVal2.shape}')

    Phi_x, Phi_y = np.vsplit(phi, 2)
    
    psi = X.dot(phi).dot(np.linalg.inv(np.diag(np.sqrt(eigVal2)))) # psi =  V * Sigma = X * U^

    return Phi_x, Phi_y, psi, np.sqrt(eigVal2)

# test_spatial_POD:
# Project the velocity components onto
# a given spatial basis and returns the temporal modes
# returns:
#           Phi_x_test.T: spatial modes of U_test
#           Phi_y_test.T: spatial modes of V_test
def test_spatial_POD(u_test, v_test, phi_train, eigVal_train):
    # print(f'u_test shape: {u_test.shape}')
    # print(f'v_test shape: {v_test.shape}')
    Sigma_i = np.linalg.inv(np.diag(np.sqrt(eigVal_train)))

    X_test = np.concatenate((u_test, v_test), axis=1)
    psi_test = X_test.dot(phi_train.T)# X*U=V*Sigma; X=V*Sigma*U^T
    # print(f'Psi_test shape: {psi_test.shape}')
    

    phi_test = Sigma_i.dot((psi_test.T).dot(X_test))# X*U=V*Sigma; X=V*Sigma*U^T
    Phi_x_test, Phi_y_test = np.vsplit(phi_test.T, 2)

    return Phi_x_test.T, Phi_y_test.T, psi_test

# test_spatial_POD_solera:
# Project the velocity components onto
# a given spatial basis and returns the temporal modes
# returns:
#           X_reconstructed
def test_reconstruction_POD_test(u_test, v_test, phi_proyect):

    X_test = np.concatenate((u_test, v_test), axis=1)
    
    # print(f'X_test shape: {X_test.shape}')
    # print(f'phi_train shape: {phi_train.shape}')
    
    # psi_test = X_test.dot(phi_train)# X*U=V*Sigma; X=V*Sigma*U^T
    # print(f'Psi_test shape: {psi_test.shape}')
    
    # psi_test_red=psi_test[:,:20]
    # phi_train_red=phi_train[:,:20]
    # print(f'psi_test_red shape: {psi_test_red.shape}')
    #Phi_x_test, Phi_y_test = np.vsplit(phi_test.T, 2)
    #X_reconstructed = psi_test_red.dot(phi_train_red.T)
    P = phi_proyect.dot(phi_proyect.T)
    print(f'P shape: {P.shape}')
    X_reconstructed = X_test.dot(P)
    # print(f'X_reconstructed shape: {X_reconstructed.shape}')
    return X_reconstructed
# test_proyection_POD:
# Reconstruct the data as X=V*S*U^T
# used only the fisrt n_max_modes 
# takes:
    #           phi_total: spatial modes of U and V
    #           Sigma_total: temporal modes of U and V
    #           psi_total: energy modes of U and V
    #           n_max_modes: maxi modes to be proyected back. 
    # n_max_modes should be equal or less than the original modes of the data
# returns:
#           u_recon: reconstruction u component
#           v_recon: reconstruction v component

def test_proyection_POD(phi_total, Sigma_total, psi_total, n_max_modes):

 X_recon = (phi_total[:,0:n_max_modes].dot(Sigma_total[0:n_max_modes,0:n_max_modes])).dot(psi_total.T[0:n_max_modes,:])
 u_recon, v_recon = np.vsplit(X_recon, 2)
     #POD_benchmark[n_recon_modes] = get_TEK(u, v, u_recon.T, v_recon.T)
     #print(f'POD_benchmark: {POD_benchmark}')
 # print(f'u_recon shape: {u_recon.shape}')
 # print(f'v_recon shape: {v_recon.shape}')
 return u_recon.T, v_recon.T

# test_reconstruction_POD_batch:
# Reconstruct the data as X=V*S*U^T
# used only the fisrt n_max_modes 
# takes:
    #           phi_total: spatial modes of U and V
    #           Sigma_total: temporal modes of U and V
    #           psi_total: energy modes of U and V
    #           n_max_modes: maxi modes to be proyected back. 
    # n_max_modes should be equal or less than the original modes of the data
# returns:
#           u_recon: last n_modes(most modes) reconstruction u component
#           v_recon: last n_modes(most modes) reconstruction v component

def test_reconstruction_POD_batch(u, v, n_max_modes):
    from tqdm import tqdm

    X = np.concatenate((u, v), axis=1)
    C = (X.T).dot(X)
    C = C / (n_train - 1)
    phi_total, eigVal2, _ = randomized_svd(C, n_components=n_max_modes, random_state=0)
    psi_total = X.dot(phi_total).dot(np.linalg.inv(np.diag(np.sqrt(eigVal2))))  # X=V*S*U^T;
    # psi =  V = X * U*sigma^-1
    Sigma_total = np.diag(np.sqrt(eigVal2))
    Phi_x_total, Phi_y_total = np.vsplit(phi_total, 2)
    # print(f'Phi_x_total shape: {Phi_x_total.shape}')
    # print(f'Phi_total shape: {Phi_y_total.shape}')
    # print(f'Sigma_total shape: {Sigma_total.shape}')
    # print(f'psi_total shape: {psi_total.shape}')
    #u_recon = np.zeros((1650,85000,n_max_modes))
    #v_recon = np.zeros((1650,85000,n_max_modes))
    POD_benchmark_TKE = np.zeros((n_max_modes+1,1))
    POD_benchmark_MSE = np.zeros((n_max_modes+1,1))
    for n_recon_modes in tqdm(range(n_max_modes)):
        n_recon_modes=n_recon_modes+1
        # print(f'n_recon_modes: {n_recon_modes}')
   
        X_recon = (phi_total[:,0:n_recon_modes].dot(Sigma_total[0:n_recon_modes,0:n_recon_modes])).dot(psi_total.T[0:n_recon_modes,:])
        u_recon, v_recon = np.vsplit(X_recon, 2)
        POD_benchmark_TKE[n_recon_modes] = get_TEK(u, v, u_recon.T, v_recon.T)
        POD_benchmark_MSE[n_recon_modes] = get_MSE(u, v, u_recon.T, v_recon.T)
        #print(f'POD_benchmark_TKE: {POD_benchmark_TKE}')
        #print(f'POD_benchmark_MSE: {POD_benchmark_MSE}')
    # print(f'u_recon shape: {u_recon.shape}')
    # print(f'v_recon shape: {v_recon.shape}')

    return u_recon.T, v_recon.T, phi_total, Sigma_total, psi_total, POD_benchmark_TKE,  POD_benchmark_MSE


# test_reconstruction_POD:
# Reconstruct the data as X=V*S*U^T
# used only the fisrt n_max_modes
# takes:
#           phi_total: spatial modes of U and V
#           Sigma_total: temporal modes of U and V
#           psi_total: energy modes of U and V
#           n_max_modes: maxi modes to be proyected back.
# n_max_modes should be equal or less than the original modes of the data
# returns:
#           u_recon: last n_modes(most modes) reconstruction u component
#           v_recon: last n_modes(most modes) reconstruction v component

def test_reconstruction_POD(u, v, n_max_modes):
    X = np.concatenate((u, v), axis=1)
    C = (X.T).dot(X)
    C = C / (n_train - 1)
    phi_total, eigVal2, _ = randomized_svd(C, n_components=n_max_modes, random_state=0)
    psi_total = X.dot(phi_total).dot(np.linalg.inv(np.diag(np.sqrt(eigVal2))))  # X=V*S*U^T;
    # psi =  V = X * U*sigma^-1
    Sigma_total = np.diag(np.sqrt(eigVal2))
    Phi_x_total, Phi_y_total = np.vsplit(phi_total, 2)
    # print(f'Phi_x_total shape: {Phi_x_total.shape}')
    # print(f'Phi_total shape: {Phi_y_total.shape}')
    # print(f'Sigma_total shape: {Sigma_total.shape}')
    # print(f'psi_total shape: {psi_total.shape}')

    X_recon = (phi_total[:, 0:n_max_modes].dot(Sigma_total[0:n_max_modes, 0:n_max_modes])).dot(psi_total.T[0:n_max_modes, :])
    u_recon, v_recon = np.vsplit(X_recon, 2)
    POD_benchmark_TKE = get_TEK(u, v, u_recon.T, v_recon.T)
    POD_benchmark_MSE = get_MSE(u, v, u_recon.T, v_recon.T)
    # print(f'u_recon shape: {u_recon.shape}')
    # print(f'v_recon shape: {v_recon.shape}')

    return u_recon.T, v_recon.T, phi_total, Sigma_total, psi_total, POD_benchmark_TKE, POD_benchmark_MSE

# get_TEK:
# Calculate energy percentage reconstructed
# takes:
    #       u_original,v_original : velocity field taked as reference
    #       u_rec,v_rec : velocity field reconstructed
# returns:
#        TKE difference between the two velocity fields introduced
def get_TEK(u_original,v_original, u_rec,v_rec):
    """Calculate energy percentage reconstructed"""
    #u_real = original[:, 0, :, :]
    #v_real = original[:, 1, :, :]
    #TKE_real = u_real ** 2 + v_real ** 2
    TKE_real = u_original ** 2 + v_original ** 2

    return 1 - np.sum((u_original - u_rec) ** 2 + (v_original - v_rec) ** 2) / np.sum(TKE_real)

# get_MSE:
# Calculate energy percentage reconstructed
# takes:
    #       u_original,v_original : velocity field taked as reference
    #       u_rec,v_rec : velocity field reconstructed
# returns:
#        MSE difference between the two velocity fields introduced
def get_MSE(u_original,v_original, u_rec,v_rec):
    """Calculate energy percentage reconstructed"""

    loss = nn.MSELoss(reduction='mean').cuda()
    X_original = np.concatenate((u_original, v_original), axis=1)
    X_rec = np.concatenate((u_rec, v_rec), axis=1)
    X_original_torch = torch.from_numpy(X_original) #
    X_rec_torch = torch.from_numpy(X_rec) #

    MSE = loss(X_original_torch, X_rec_torch) 
    return MSE

# Time_marching_prediction (DUMMY):
# Moves the Temporal modes into de future time_to instants
# takes:
#       Psi_time_present : temporal modes of the present moment
#       time_to : time to move the modes
# returns:
#       Psi_time_future : temporal modes of the future
def Time_marching_prediction(Psi_time_present, time_to):
    Psi_time_future = Psi_time_present
    return Psi_time_future

#%%
#######################################################
########         MAIN PROGRAM         #################
#######################################################


# creating directories if do not exist
Path("01_data").mkdir(exist_ok=True)
Path("04_modes").mkdir(exist_ok=True)
Path("08_POD").mkdir(exist_ok=True)

# load data
if DEBUG_MODE:
    print('DEBUG MODE ENABLED')
if not Load_original_data:
    print('Load_original_data DISABLED')

if Load_original_data:
    print('Loading data')
    u_scaled, mean, std = lib_data.loadData(datafile)

    u_scaled = u_scaled[::dt]

    # ONLY for some cases
    if u_scaled.shape[2] == 2:
        u_scaled = np.moveaxis(u_scaled, 1, -1)
        mean = np.moveaxis(mean, 1, -1)
        std = np.moveaxis(std, 1, -1)

    print(f'u shape: {u_scaled.shape}, mean: {mean.shape}, std: {std.shape}')

    if u_scaled.ndim > 4:
        u_scaled = u_scaled.reshape((-1, 2, u_scaled.shape[3], u_scaled.shape[4]))
        print(f'u reshaped to {u_scaled.shape}')

    if DEBUG_MODE:
        u_scaled = u_scaled[:,:,::4,::4]
        mean = mean[:,:,::4,::4]

    n_total = u_scaled.shape[0]
    data_dim_1 = u_scaled.shape[2]
    data_dim_2 = u_scaled.shape[3]
    
    n_train = n_total - n_test
    print(f"N train: {n_train:d}, N test: {n_test:d}, N total {n_total:d}")

    u_train = u_scaled[:n_train, 0, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 0, 0, 0]
    v_train = u_scaled[:n_train, 1, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 1, 0, 0]
    u_test = u_scaled[n_train:, 0, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 0, 0, 0]
    v_test = u_scaled[n_train:, 1, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 1, 0, 0]
    del u_scaled
    
    print(f'Npx2: {data_dim_1*data_dim_2*2}')
    print(f'Nt: {n_train//dt}')
    

## POD RECONSTRUCTION ENERGY 
    ## Proyect train data into n_modes most energetic modes
    Phi_x, Phi_y, Psi_train, eigVal_train = snapshot_POD_overSpace(u_train, v_train, n_modes)
    phi_train = np.concatenate((Phi_x, Phi_y), axis=0)
    Sigma_train = np.diag(eigVal_train) 
    print(f'psi_train shape: {Psi_train.shape}')
    print(f'Sigma_train shape: {Sigma_train.shape}')
    print(f'phi_train shape: {phi_train.shape}')

    X_tr_rec = (Psi_train.dot(Sigma_train)).dot(phi_train.T)
    u_tr_rec, v_tr_rec = np.hsplit(X_tr_rec, 2)
    TKE_train = get_TEK(u_train, v_train, u_tr_rec, v_tr_rec)
    print(f'TKE_train: {TKE_train}')
    print(f'-------------------------------')
    #Phi_x_test, Phi_y_test, psi_test = test_spatial_POD(u_test, v_test, phi_train, eigVal_train)
    print(f'u_test shape: {u_test.shape}')
    print(f'v_test shape: {v_test.shape}')
    X_test_rec = test_reconstruction_POD_test(u_test, v_test, phi_train)
    u_rec, v_rec = np.hsplit(X_test_rec, 2)

    TKE_test = get_TEK(u_test, v_test, u_rec, v_rec)
    print(f'TKE_test: {TKE_test}')
   
    import matplotlib.pyplot as plt
    #plotting the array
    Ulim = max(abs(u_test[-1].flatten()))
    fig, ax = plt.subplots(2, 2, sharex=True)

    axs = ax[0, 0]
    im = axs.imshow(u_test[-1].reshape(data_dim_1, data_dim_2), cmap="RdBu_r", origin='lower', aspect='equal', vmin=-Ulim, vmax=Ulim)
    axs.set_title('True test u')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[0, 1]
    im = axs.imshow(u_rec[-1].reshape(data_dim_1, data_dim_2), cmap="RdBu_r", origin='lower', aspect='equal', vmin=-Ulim, vmax=Ulim)
    axs.set_title(f'POD test u, {n_modes} modes')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[1, 0]
    im = axs.imshow(v_test[-1].reshape(data_dim_1, data_dim_2), cmap="RdBu_r", origin='lower', aspect='equal', vmin=-Ulim, vmax=Ulim)
    axs.set_title('True test v')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[1, 1]
    im = axs.imshow(v_rec[-1].reshape(data_dim_1, data_dim_2), cmap="RdBu_r", origin='lower', aspect='equal', vmin=-Ulim, vmax=Ulim)
    axs.set_title(f'POD test v, {n_modes} modes')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    plt.show()
else:
    file = '08_POD/Intermediate_POD_Data_20modes.h5py'
    print('Loading INTERMEDIATE data only')
    with h5py.File(file, 'r') as f:
        Psi_train = f['Psi_train'][:]
        phi_train = f['phi_train'][:]
        Sigma_train = f['Sigma_train'][:]

## Take only the 20 most energetic modes and move them foward-time t_instants (DUMMY)
#t_instants = 42;
#Psi_time_future = Time_marching_prediction(Psi_train[:,0:20], t_instants)


##--------------------------------------------------------------------

## Reconstruct that data taking into account only the n_modes most energetic modes
# NOTE: 
    # test_reconstruction_POD makes ONE reconstruction over the n_modes selected
    # test_reconstruction_POD_batch makes N_MODES reconstruction from 1 to n_modes
    
#u_reconstruct, v_reconstruct = test_proyection_POD(phi_train.T, Sigma_train, Psi_train, n_modes)
u_reconstruct, v_reconstruct, phi_total, Sigma_total, psi_total, POD_benchmark_TKE, POD_benchmark_MSE = test_reconstruction_POD_batch(u_train, v_train, n_modes)

if Load_original_data:
    ## PERFORM reconstruction test TKE metric
    POD_TKE_benchmark = get_TEK(u_train, v_train, u_reconstruct, v_reconstruct)
    # print(f'POD_benchmark_TKE: {POD_TKE_benchmark}')
    
    ## PERFORM reconstruction test MSE metric
    
    POD_MSE_benchmark = get_MSE(u_train, v_train, u_reconstruct, v_reconstruct)
    # print(f'POD_benchmark_MSE: {POD_MSE_benchmark}')
##--------------------------------------------------------------------
    
## Reconstruct and save the data
if Load_original_data:
    if DEBUG_MODE:
        Phi_x_shaped = np.reshape(Phi_x, (-1, data_dim_1, data_dim_2))
        Phi_y_shaped = np.reshape(Phi_y, (-1, data_dim_1, data_dim_2))
        u_reconstruct_shaped = np.reshape(u_reconstruct, (-1, data_dim_1, data_dim_2))
        v_reconstruct_shaped = np.reshape(v_reconstruct, (-1, data_dim_1, data_dim_2))
        print('DEBUG MODE; nothing saved')
    else:
        Phi_x_shaped = np.reshape(Phi_x, (-1, data_dim_1, data_dim_2))
        Phi_y_shaped = np.reshape(Phi_y, (-1, data_dim_1, data_dim_2))
        u_reconstruct_shaped = np.reshape(u_reconstruct, (-1, data_dim_1, data_dim_2))
        v_reconstruct_shaped = np.reshape(v_reconstruct, (-1, data_dim_1, data_dim_2))
    # save the data
    caseName = datafile.split('/')[-1].split('.')[0]
    fname = f'08_POD/POD_{case}_Data.h5py'
    print('saving intermediate data')
    with h5py.File(fname, 'w') as f:
        f.create_dataset('Psi_train', data=Psi_train)
        f.create_dataset('phi_train', data=phi_train)
        f.create_dataset('Sigma_train', data=Sigma_train)
        #f.create_dataset('Phi_x_total', data=Phi_x_shaped)
        #f.create_dataset('Phi_y_total', data=Phi_y_shaped)
        f.create_dataset('TKE', data=POD_benchmark_TKE)
        f.create_dataset('MSE', data=POD_benchmark_MSE)
        #f.create_dataset('u_test_reconstruct', data=u_reconstruct_shaped)
        #f.create_dataset('v_test_reconstruct', data=v_reconstruct_shaped)

    #   f.create_dataset('eigVal', data=eigVal)
else:
    print('nothing saved')
    #%%
    n_modes = psi_total.shape[1]

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,n_modes+1,n_modes+1), POD_benchmark_TKE)

    ax.set(xlabel='Number of modes considered', ylabel='Cumulative Ek',
           title='Re=100; tc = 1; train data')
    ax.grid()

    fig.savefig("Re100_tc0.2.png")
    plt.show()
       
    
