import matplotlib.pyplot as plt
import h5py
import numpy as np

files = [
    '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py']

firstN = 6
Ulim = 3
fig, ax = plt.subplots(firstN, len(files), figsize=(8, 10), sharex='col', sharey='row')

nmodes = np.zeros(len(files), dtype=np.int16)

# plot u
for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        modes = f['modes'][:]
        order = f['order'][:]

    for i in range(firstN):
        Ulim = np.std(modes[order[i], 0, :, :].flatten()) * 4
        im = ax[i, idx].imshow(modes[order[i], 0, :, :], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim, extent=[-9, 87, -14, 14],
                               origin='lower')
        ax[i, idx].set_title(f'Mode {i + 1}, {nmodes[idx]} modes')
        ax[i, idx].set_xlabel('x/c')
        ax[i, idx].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, idx], shrink=0.8, aspect=10)

fig.set_tight_layout(True)
plt.show()

# plot v
fig, ax = plt.subplots(firstN, len(files), figsize=(8, 10), sharex='col', sharey='row')
for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        print("Keys: %s" % f.keys())
        modes = f['modes'][:]
        order = f['order'][:]

    for i in range(firstN):
        Ulim = np.std(modes[order[i], 1, :, :].flatten()) * 4
        im = ax[i, idx].imshow(modes[order[i], 1, :, :], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim, extent=[-9, 87, -14, 14],
                               origin='lower')
        ax[i, idx].set_title(f'Mode {i + 1}, {nmodes[idx]} modes')
        ax[i, idx].set_xlabel('x/c')
        ax[i, idx].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, idx], shrink=0.8, aspect=10)

fig.set_tight_layout(True)
plt.show()
