import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230712_12_36_beta0.05_dim20_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py',
         '04_modes/20230712_09_30_beta0.05_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py',
         '04_modes/20230712_20_34_beta0.05_dim40_lr0.0001_bs256_epochs1000_Wdecay0.0005_epoch_final_modes.h5py',
         '04_modes/20230718_11_31_beta0.05_dim20_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_bestTest_modes.h5py',
         '04_modes/20230717_16_22_beta0.05_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_bestTest_modes.h5py',
         '04_modes/20230717_09_33_beta0.05_dim40_lr0.0001_bs256_epochs1000_Wdecay0.0005_nt80000_epoch_bestTest_modes.h5py']

names = ['20 modes, 45k train',
         '30 modes, 45k train',
         '40 modes, 45k train',
         '20 modes, 80k train',
         '30 modes, 80k train',
         '40 modes, 80k train']

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