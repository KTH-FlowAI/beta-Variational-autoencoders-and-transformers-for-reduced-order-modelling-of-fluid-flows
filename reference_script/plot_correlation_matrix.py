import matplotlib.pyplot as plt
import h5py
import numpy as np
# Standard imports
import pandas as pd
# For this example we'll use Seaborn, which has some nice built in plots
import seaborn as sns


files = [
    '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    '04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
    #'04_modes/20231025_21_59_Re40_smallerCNN_beta0.001_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py',
    #'04_modes/20230928_13_04_Re40_smallerCNN_beta0.005_wDecay0_dim2_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt901_epoch_final_modes.h5py'
    ]


for file in files:
    with h5py.File(file, 'r') as f:
        print("Keys: %s" % f.keys())
        temporalModes = f['vector'][:]
        order = f['order'][:]
        print(temporalModes.shape)

    df = pd.DataFrame(temporalModes[:, order])

    # Create the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 3))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    cmap = sns.color_palette("icefire", as_cmap=True)
    # cmap = sns.diverging_palette(100, 200, s=75, l=50, sep=1, n=6, center='light', as_cmap=True)

    n_modes = temporalModes.shape[1]

    axis_labels = np.arange(1, n_modes + 1).tolist()
    axis_labels = list(map(str, axis_labels))
    if n_modes >= 20:
        for i in range(0, n_modes, 2):
            axis_labels[i] = ''

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        np.abs(corr),  # The data to plot
        # mask=mask,     # Mask some cells
        cmap=cmap,  # What colors to plot the heatmap as
        annot=False,  # Should the values be plotted in the cells?
        fmt=".2f",
        vmax=1,  # The maximum value of the legend. All higher vals will be same color
        vmin=0,  # The minimum value of the legend. All lower vals will be same color
        # center=0.75,      # The center value of the legend. With divergent cmap, where white is
        square=True,  # Force cells to be square
        linewidths=0.5,  # Width of lines that divide cells
        # cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
        # norm=LogNorm()
        xticklabels=axis_labels,
        yticklabels=axis_labels,
    )
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(f'05_Figs/corrMat_{temporalModes.shape[1]}.png', bbox_inches = "tight")
    plt.savefig(f'05_Figs/corrMat_{temporalModes.shape[1]}.eps', format='eps', bbox_inches = "tight")
    plt.show()
