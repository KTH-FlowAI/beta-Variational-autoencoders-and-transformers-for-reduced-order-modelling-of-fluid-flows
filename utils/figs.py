"""
Function supports visualisation

Author: @yuningw & @alsolera

Editting: @yuningw 
"""

import matplotlib.pyplot as plt
from lib.init import pathsBib


# --------------------------------------------------------
def plot_training(logFile):
    """Plot from tensorboard file"""
    from tbparse import SummaryReader

    def loadLog(logFile):
        df = SummaryReader(logFile, pivot=True).scalars
        df = df[['step', 'General loss/KLD', 'General loss/KLD_test', 'General loss/MSE',
                 'General loss/MSE_test', 'General loss/Total',
                 'General loss/Total_test']]
        list(df.columns)
        return df['step'].to_numpy(), df['General loss/KLD'].to_numpy(), df['General loss/MSE'].to_numpy(), df[
            'General loss/MSE_test'].to_numpy(), df['General loss/KLD_test'].to_numpy()

    step, KL, MSE, MSE_t, KL_t = loadLog(logFile)

    fig, ax1 = plt.subplots(figsize=(5, 2))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax1.plot(step, KL, '-', label='KL', color='lightseagreen')
    ax2.plot(step, MSE, '-', label='MSE', color='saddlebrown')
    ax1.plot(step, KL_t, ':', label='KL test', color='lightseagreen')
    ax2.plot(step, MSE_t, ':', label='MSE test', color='saddlebrown')

    ax1.set_xlabel('Train step')
    # ax1.axis(xmin=0, xmax=1000)
    # ax1.axis(ymin=1)
    ax1.set_ylabel('KL loss', color='lightseagreen')
    ax1.tick_params(axis='y', labelcolor='lightseagreen')

    ax2.set_ylabel('Reconstruction loss', color='saddlebrown')
    ax2.tick_params(axis='y', labelcolor='saddlebrown')
    # ax2.axis(ymax=0.5)
    # plt.ylabel('Average error (L2)')
    # plt.legend()

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(color='whitesmoke', zorder=1)

    plt.legend()

    plt.savefig(pathsBib.fig_path + 'bVAE/training.png', format='png', bbox_inches="tight")


# --------------------------------------------------------

def annot_max(x, y, ax=None):
    import numpy as np

    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "$f/f_c={:.3f}$".format(xmax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


# --------------------------------------------------------

def plotCompleteModes(modes, temporalModes, numberModes, fs, order):
    from scipy import signal
    import numpy as np

    fig, ax = plt.subplots(numberModes, 3, figsize=(16, numberModes * 1.6), sharex='col')

    # for mode in range(numberModes):

    for i, mode in np.ndenumerate(order):
        i = i[0]
        print(i, mode)
        Uplot = modes[mode, 0, :, :]
        Vplot = modes[mode, 1, :, :]

        Ulim = max(abs(Uplot.flatten()))
        Vlim = max(abs(Vplot.flatten()))

        im = ax[i, 0].imshow(Uplot, cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                             extent=[-9, 87, -14, 14])
        # ax[mode,0].set_title('Mode ' + str(mode) + ', u')
        ax[i, 0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, 0], shrink=0.8, aspect=10)

        im = ax[i, 1].imshow(Vplot, cmap="RdBu_r", vmin=-Vlim, vmax=Vlim,
                             extent=[-9, 87, -14, 14])
        ax[i, 1].set_title('Mode ' + str(i + 1))
        # ax[mode,1].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, 1], shrink=0.8, aspect=10)

        f, Pxx_den = signal.welch(temporalModes[:, mode], fs, nperseg=512 * 4)
        ax[i, 2].plot(f, Pxx_den, color='lightseagreen')
        ax[i, 2].axis(xmin=0, xmax=.15)
        annot_max(f, Pxx_den, ax=ax[i, 2])
        ax[i, 2].grid(color='whitesmoke', zorder=1)
        ax[i, 2].spines['top'].set_visible(False)
        ax[i, 2].spines['right'].set_visible(False)
        if mode == (numberModes - 1):
            ax[i, 0].set_xlabel('$x/c$')
            ax[i, 1].set_xlabel('$x/c$')
            ax[i, 2].set_xlabel('$f/f_c$')

    plt.savefig(pathsBib.fig_path + 'bVAE/modes.png', format='png', bbox_inches="tight")


# --------------------------------------------------------

def correlationMatrix(temporalModes, order):
    import pandas as pd
    import seaborn as sns
    import numpy as np

    df = pd.DataFrame(temporalModes[:, order])

    # Create the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 3))

    # Generate a colormap
    cmap = sns.color_palette("icefire", as_cmap=True)

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

    plt.savefig(pathsBib.fig_path + 'bVAE/matrix.png', format='png', bbox_inches="tight")


# --------------------------------------------------------

def plotTemporalSeries(modes):
    latent_dim = modes.shape[1]
    fig, ax = plt.subplots(latent_dim, 1, figsize=(16, latent_dim * 1.6), sharex='col')

    for i in range(modes.shape[1]):
        ax[i].plot(modes[:1000, i], label='Mode ' + str(i))
        ax[i].axis(ymin=-2.5, ymax=2.5)

        ax[i].legend()
        ax[i].grid()

    plt.savefig(pathsBib.fig_path + 'bVAE/series.png', format='png', bbox_inches="tight")


# --------------------------------------------------------
def plotEcum(Ecum):
    import numpy as np

    fig = plt.figure()
    plt.plot(np.arange(1, Ecum.shape[0] + 1), Ecum)
    plt.xlabel('Number of modes')
    plt.ylabel('Cumulative Ek')
    plt.grid()

    plt.savefig(pathsBib.fig_path + 'bVAE/Ecum.png', format='png', bbox_inches="tight")


# --------------------------------------------------------
def plotNLmodeField(modes, values):
    import numpy as np

    valuesToPlot = np.array([-2., -1., 0., 1., 2.])

    # Find the indices of values_array elements in main_array
    indices = []
    for value in valuesToPlot:
        matching_indices = np.where(np.abs(values - value) < 1e-3)[0]
        indices.extend(matching_indices)

    indices_array = np.array(indices)
    values = values[indices_array]

    Ulim = max(abs(modes[indices_array, 0, :, :].flatten()))
    Vlim = max(abs(modes[indices_array, 1, :, :].flatten()))

    fig, ax = plt.subplots(len(indices_array), 2, figsize=(10, 8), sharex='col', sharey='row')

    for idx, value in enumerate(values):
        print(idx, value)
        Uplot = modes[indices_array[idx], 0, :, :]
        Vplot = modes[indices_array[idx], 1, :, :]

        im = ax[idx, 0].imshow(Uplot, cmap="RdBu_r", vmin=-Ulim, vmax=Ulim, extent=[-9, 87, -14, 14])
        # ax[idx, 0].set_title('Value {}'.format(round(value, 1)))
        ax[idx, 0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[idx, 0], shrink=0.7, aspect=10)

        im = ax[idx, 1].imshow(Vplot, cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
        # ax[idx,1].set_title('Mode {}, value {}, v'.format(mode, round(NLvalues[value],2)))
        fig.colorbar(im, ax=ax[idx, 1], shrink=0.7, aspect=10)

        fig.text(0.48, 0.965 - idx * .184, '$s_1 = {}$'.format(round(value, 1)), fontsize=11, ha="center")

    ax[idx, 0].set_xlabel('x/c')
    ax[idx, 1].set_xlabel('x/c')

    fig.set_tight_layout(True)

    plt.savefig(pathsBib.fig_path + 'bVAE/NLfield.png', format='png', bbox_inches="tight")


def vis_bvae(modes_file, training_file):
    import h5py
    from pathlib import Path

    Path(pathsBib.fig_path + 'bVAE/').mkdir(exist_ok=True)

    try:
        plot_training(training_file)
    except:
        print(f'Warning: {training_file} could not be loaded')

    with h5py.File(modes_file, 'r') as f:
        print("Keys: %s" % f.keys())
        temporalModes = f['vector'][:, :]
        temporalModes_test = f['vector_test'][:, :]
        modes = f['modes'][:, :]
        order = f['order'][:]
        Ecum = f['Ecum'][:]
        NLvalues = f['NLvalues'][:]
        NLmodes = f['NLmodes'][:]

    # Re40 case is sampled at 1tc, Re100 case is sampled at tc/5
    if 'Re40' in modes_file:
        fs = 1
    else:
        fs = 5

    plotNLmodeField(NLmodes, NLvalues)
    plotCompleteModes(modes, temporalModes, modes.shape[0], fs, order)
    plotTemporalSeries(temporalModes)
    correlationMatrix(temporalModes, order)
    plotEcum(Ecum)
