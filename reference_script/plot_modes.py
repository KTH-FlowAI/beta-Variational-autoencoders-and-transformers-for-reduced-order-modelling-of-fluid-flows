import matplotlib.pyplot as plt
import h5py
import numpy as np
from lib_figures import plotCompleteModes


files = [
    '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    #'04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    '04_modes/20231025_21_59_Re40_smallerCNN_beta0.001_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py',
    #'04_modes/20230928_13_04_Re40_smallerCNN_beta0.005_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py'
    ]

Nplot = 6


for file in files:
    with h5py.File(file, 'r') as f:
        print("Keys: %s" % f.keys())
        temporalModes = f['vector'][:]
        order = f['order'][:]
        modes = f['modes'][:, :]

    print(temporalModes.shape)
    nmodes = temporalModes.shape[1]

    if nmodes == 2:
        fs = 1
        xytext = (0.54,0.96)
    else:
        fs = 5
        xytext = (0.94,0.96)

    fig, ax = plotCompleteModes(modes, temporalModes, modes.shape[0], order, min(Nplot, nmodes), fs, xytext)
    plt.savefig(f'05_Figs/modes_{nmodes}.png', bbox_inches = "tight")
    plt.savefig(f'05_Figs/modes_{nmodes}.eps', format='eps', bbox_inches = "tight")
plt.show()
