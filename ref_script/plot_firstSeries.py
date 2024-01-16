import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py']

firstN = 6
seriesLen = 1000
Ulim = 3
fig, ax = plt.subplots(firstN, len(files), figsize=(16, 10), sharex='col', sharey='row')

nmodes = np.zeros(len(files), dtype=np.int16)

# plot u
for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        modes = f['vector'][:seriesLen]
        order = f['order'][:]

        print(modes.shape)

    for i in range(firstN):
        ax[i, idx].plot(modes[:, order[i]])
        ax[i, idx].set_title(f'Mode {i+1}, {nmodes[idx]} modes')
        ax[i, idx].set_xlabel('idx')
        ax[i, idx].grid()
        ax[i, idx].set_xlim(0, seriesLen)

fig.set_tight_layout(True)
plt.show()