import matplotlib.pyplot as plt
import h5py
import numpy as np
from lib_figures import plotNLmodeField


files = [
    '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    # '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    # '04_modes/20231025_21_59_Re40_smallerCNN_beta0.001_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py',
    # '04_modes/20230928_13_04_Re40_smallerCNN_beta0.005_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py'
    ]

Nplot = 6


for file in files:
    with h5py.File(file, 'r') as f:
        print("Keys: %s" % f.keys())
        order = f['order'][:]
        NLvalues = f['NLvalues'][:]
        NLmodes = f['NLmodes'][:]

    nmodes = order.shape[0]

    fig, ax = plotNLmodeField(NLmodes, NLvalues)
    #plt.savefig(f'05_Figs/NLmodes_{nmodes}.png', bbox_inches = "tight")
    plt.savefig(f'05_Figs/NLmodes_{nmodes}.eps', format='eps', bbox_inches = "tight")
#plt.show()
