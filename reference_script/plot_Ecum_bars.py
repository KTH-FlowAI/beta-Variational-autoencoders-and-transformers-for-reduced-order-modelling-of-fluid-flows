import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230809_18_19_ResNET0_beta0.1_dim20_lr0.0001linear0.0001_bs256_epochs1000_encWdecay0_decWdecay0.0003_nt135000_epoch_bestTest_modes.h5py',
         '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
         '04_modes/20230915_14_32_smallerCNN_beta0.05_wDecay0_dim15_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
         '04_modes/20230914_18_41_smallerCNN_beta0.2_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
         '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py']

labels = ['Baseline normalized',
          'smallCNN, tc1, 10m, b0.05',
          'smallCNN, tc1, 15m, b0.05',
          'smallCNN, tc1, 20m, b0.2',
          'smallCNN, tc1, 20m, b0.05']

nmodes = np.zeros(len(files), dtype=np.int16)


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for idx, file in enumerate(files):
    # nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        Ecum = f['Ecum'][-1]
        Ecum_test = f['Ecum_test'][-1]

    p = ax.bar(labels[idx], Ecum_test, label='Test', color='tab:blue')
    ax.bar_label(p)
    p = ax.bar(labels[idx], Ecum-Ecum_test, bottom=Ecum_test, label='Train', color='tab:orange')
    ax.bar_label(p)
    ax.set_ylabel('Cumulative Ek')

ax.grid()
#ax.legend()
fig.set_tight_layout(True)
plt.show()