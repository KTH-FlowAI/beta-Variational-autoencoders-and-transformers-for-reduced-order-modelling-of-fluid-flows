import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230719_21_55_beta0.025_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_final_modes.h5py',
         '04_modes/20230717_16_22_beta0.05_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_bestTest_modes.h5py',
         '04_modes/20230719_14_25_beta0.1_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_final_modes.h5py']

names = ['30 modes, beta 0.025',
         '30 modes, beta 0.05',
         '30 modes, beta 0.01']

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

ax.grid()
ax.legend()
fig.set_tight_layout(True)
plt.show()