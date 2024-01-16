import numpy as np
import time
import h5py
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
import lib_data

#######################################################
# #######         PARAMETERS           ################
#######################################################

case = 'twoPlatesRe100_predicted'
DEBUG_MODE = 0
n_modes = 20
Load_original_data = 1

if case == 'twoPlatesRe100_tc1':
    datafile = '01_data/Re100alpha10_newData_150000.hdf5'
    dt = 5  # 5 means 1 tc time step
    n_test = 15000 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_tc0.2':
    datafile = '01_data/Re100alpha10_newData_150000.hdf5'
    dt = 1  # 5 means 1 tc time step
    n_test = 15000 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe40':
    datafile = '01_data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5'
    dt = 1
    n_test = 100 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_reconstructed':
    datafile = ('04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_'
                'nt27000_epoch_final_reconstruction.h5py')
    dt = 5
    n_test = 15000 // dt  # //dt; means integer division. to be consistent with dt.
elif case == 'twoPlatesRe100_predicted':
    datafile = ('06_Preds/dim20/VAE_easyAttn_64in_64dmodel_1next_20dim_timeemb_4h_4nb_128ff_reluact_Noneoutact_'
                '1000Epoch_135000N_FalseES_0P_5Interval_pred_VAE.h5py')
    dt = 1 # data is already 1 tc
    n_test = 3000


# %%
#######################################################
# #######         FUNCTIONS            ################
#######################################################

def snapshot_POD_overSpace(u, v, n_modes):
    """    snapshot_POD_overSpace:
    Use when the resolution of a single snapshot
    is much smaller than the number of time points
    (Nt>>Np)
    takes:
              u: U velocity field complete component
              v: V velocity field complete component
              n_modes: energy modes to be computed/extracted
    returns:
              Phi_x.T: spatial modes of U, up only to n_modes
              Phi_y.T: spatial modes of V, up only to n_modes
              psi: temporal modes of both U and V times Sigma, up only to n_modes
              eigVal: eigenvalues of the modes, up only to n_modes
              """

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
    print(f': {(time.time() - start):.1f}s')

    Phi_x, Phi_y = np.vsplit(phi, 2)

    psi = X.dot(phi).dot(np.linalg.inv(np.diag(np.sqrt(eigVal2))))  # psi =  V * Sigma = X * U^

    return Phi_x, Phi_y, psi, np.sqrt(eigVal2)


#######################################################
# #######         MAIN PROGRAM         ################
#######################################################

# creating directories if do not exist
Path("01_data").mkdir(exist_ok=True)
Path("04_modes").mkdir(exist_ok=True)
Path("08_POD").mkdir(exist_ok=True)

# load data
if DEBUG_MODE:
    print('DEBUG MODE ENABLED')

print('Loading data')
u_scaled, mean, std = lib_data.loadData(datafile)
u_scaled = u_scaled[::dt]

# ONLY for some cases
if u_scaled.shape[2] == 2:
    u_scaled = np.moveaxis(u_scaled, 1, -1)
    mean = np.moveaxis(mean, 1, -1)
    std = np.moveaxis(std, 1, -1)

print(f'u shape: {u_scaled.shape}, mean: {mean.shape}, std: {std.shape}')

if DEBUG_MODE:
    u_scaled = u_scaled[:, :, ::4, ::4]
    mean = mean[:, :, ::4, ::4]

n_total = u_scaled.shape[0]
data_dim_1 = u_scaled.shape[2]
data_dim_2 = u_scaled.shape[3]

n_train = n_total - n_test
print(f"N train: {n_train:d}, N test: {n_test:d}, N total {n_total:d}")

u_test = u_scaled[n_train:, 0, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 0, 0, 0]
v_test = u_scaled[n_train:, 1, :, :].reshape((-1, data_dim_1 * data_dim_2)) * std[0, 1, 0, 0]
del u_scaled

print(f'Npx2: {data_dim_1 * data_dim_2 * 2}')
print(f'Nt: {n_train // dt}')

# Project train data into n_modes most energetic modes
Phi_x, Phi_y, Psi_train, eigVal_train = snapshot_POD_overSpace(u_test, v_test, n_modes)
phi_train = np.concatenate((Phi_x, Phi_y), axis=0)
Sigma_train = np.diag(eigVal_train)
print(f'psi_train shape: {Psi_train.shape}')
print(f'Sigma_train shape: {Sigma_train.shape}')
print(f'phi_train shape: {phi_train.shape}')

if not DEBUG_MODE:
    # save the data
    # caseName = datafile.split('/')[-1].split('.')[0]
    fname = f'08_POD/POD_{case}_OnlyTest.h5py'
    print(f'saving data: {fname}')
    with h5py.File(fname, 'w') as f:
        f.create_dataset('Psi_train', data=Psi_train)
        f.create_dataset('phi_train', data=phi_train)
        f.create_dataset('Sigma_train', data=Sigma_train)
