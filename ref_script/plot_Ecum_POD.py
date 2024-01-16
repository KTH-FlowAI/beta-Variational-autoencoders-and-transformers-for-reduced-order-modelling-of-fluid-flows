import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230808_13_59_ResNET0_beta0.1_dim20_lr0.0001linear0.0001_bs256_epochs1000_encWdecay0_decWdecay0.0003_nt90000_epoch_bestTest_modes.h5py']

names = ['20 modes, nt 90000']

nmodes = np.zeros(len(files), dtype=np.int16)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        Ecum = f['Ecum'][:]
        Ecum_test = f['Ecum_test'][:]

    ax.plot(np.arange(1, Ecum.shape[0]+1), Ecum, '-', color=colors[idx], label=names[idx])
    ax.plot(np.arange(1, Ecum_test.shape[0]+1), Ecum_test, '--', color=colors[idx], label=f'{names[idx]}, test  data')
    ax.set_xlabel('Number of modes considered')
    ax.set_ylabel('Cumulative Ek')
    ax.set_ylim(0,1)

file = '08_POD/Re100alpha10_newData_100000_dt1_nt90000.h5py'
with h5py.File(file, 'r') as f:
    # Phi_x = f['Phi_x'][:]
    # Phi_y = f['Phi_y'][:]
    # phi = f['phi'][:]
    eigVal = f['eigVal'][:]

n_modes = eigVal.shape[0]
plt.plot(np.arange(1, n_modes + 1), (np.cumsum(eigVal) / np.sum(eigVal)), color='r',
         label='POD, nt 90000, train data')

file = '08_POD/Re100alpha10_newData_150000_dt1_nt135000.h5py'
with h5py.File(file, 'r') as f:
    #Phi_x = f['Phi_x'][:]
    #Phi_y = f['Phi_y'][:]
    #phi = f['phi'][:]
    eigVal = f['eigVal'][:]

n_modes = eigVal.shape[0]
plt.plot(np.arange(1, n_modes+1), (np.cumsum(eigVal)/np.sum(eigVal)), color='k', label='POD, nt 135000, train data')

plt.xlim(1,50)
ax.grid()
ax.legend()
fig.set_tight_layout(True)
plt.show()