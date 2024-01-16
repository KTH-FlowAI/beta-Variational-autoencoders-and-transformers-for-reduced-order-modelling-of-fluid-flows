import matplotlib.pyplot as plt
import h5py
import numpy as np

files = ['04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
         '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py']

nmodes = np.zeros(len(files), dtype=np.int16)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])

    with h5py.File(file, 'r') as f:
        Ecum = f['Ecum'][:]
        Ecum_test = f['Ecum_test'][:]

    ax.plot(np.arange(1, Ecum.shape[0]+1), Ecum*100, '-', color=colors[idx], label=f'VAE {Ecum.shape[0]} modes, train  data')
    ax.plot(np.arange(1, Ecum_test.shape[0]+1), Ecum_test*100, '--', color=colors[idx], label=f'VAE {Ecum.shape[0]} modes, test  data')
    ax.set_xlabel('Number of modes considered')
    ax.set_xticks(np.arange(0, 80, step=10))
    ax.set_yticks(np.arange(0, 101, step=10))
    ax.set_ylabel('Cumulative E (%)')
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 101)

filePOD = '08_POD/POD_twoPlatesRe100_tc1_Data.h5py'
with h5py.File(filePOD, 'r') as f:
    POD_E = f['TKE'][:]

ax.plot(POD_E*100, '-', color='k', label=f'POD, train  data')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(color='whitesmoke', zorder=1)
ax.legend()
fig.set_tight_layout(True)
plt.savefig("05_Figs/Ecum_Re100.png", bbox_inches="tight")
plt.savefig("05_Figs/Ecum_Re100.eps", format='eps', bbox_inches="tight")
plt.show()