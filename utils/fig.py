import matplotlib.pyplot as plt


def plot_training(logFile):
    """Plot from tensorboard file"""
    from tbparse import SummaryReader
    import matplotlib.pyplot as plt
    def loadLog(logFile):
        df = SummaryReader(logFile, pivot=True).scalars
        df = df[['step', 'General loss/KLD', 'General loss/KLD_test', 'General loss/MSE',
                 'General loss/MSE_test', 'General loss/Total',
                 'General loss/Total_test']]
        list(df.columns)
        return df['step'].to_numpy(), df['General loss/KLD'].to_numpy(), df['General loss/MSE'].to_numpy(), df['General loss/MSE_test'].to_numpy(), df['General loss/KLD_test'].to_numpy()

    step, KL, MSE, MSE_t, KL_t = loadLog(logFile)

    fig, ax1 = plt.subplots(figsize=(5, 2))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax1.plot(step, KL, '-', label='KL', color='lightseagreen')
    ax2.plot(step, MSE, '-', label='MSE', color='saddlebrown')
    ax1.plot(step, KL_t, ':', label='KL test', color='lightseagreen')
    ax2.plot(step, MSE_t, ':', label='MSE test', color='saddlebrown')

    ax1.set_xlabel('Train step')
    #ax1.axis(xmin=0, xmax=1000)
    #ax1.axis(ymin=1)
    ax1.set_ylabel('KL loss', color='lightseagreen')
    ax1.tick_params(axis='y', labelcolor='lightseagreen')

    ax2.set_ylabel('Reconstruction loss', color='saddlebrown')
    ax2.tick_params(axis='y', labelcolor='saddlebrown')
    #ax2.axis(ymax=0.5)
    # plt.ylabel('Average error (L2)')
    # plt.legend()

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(color='whitesmoke', zorder=1)

    plt.legend()

    plt.savefig('training.png', format='png', bbox_inches="tight")
    plt.show()

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    #text= "f={:.3f}Hz, {:.1f}".format(xmax, ymax)
    text= "$f/f_c={:.3f}$".format(xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


def plotCompleteModes(modes, temporalModes, numberModes):
    from scipy import signal

    fs = 5

    fig, ax = plt.subplots(numberModes, 3, figsize=(16, numberModes*1.6), sharex='col')

    #for mode in range(numberModes):

    for i, mode in np.ndenumerate(order):
        i = i[0]
        print(i, mode)
        Uplot = modes[mode,0,:,:]
        Vplot = modes[mode,1,:,:]

        Ulim = max( abs(Uplot.flatten()) )
        Vlim = max( abs(Vplot.flatten()) )

        im = ax[i,0].imshow(Uplot, cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
        #ax[mode,0].set_title('Mode ' + str(mode) + ', u')
        ax[i,0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i,0], shrink=0.8, aspect=10)

        im = ax[i,1].imshow(Vplot, cmap="RdBu_r", vmin=-Vlim, vmax=Vlim,
                         extent=[-9, 87, -14, 14])
        ax[i,1].set_title('Mode ' + str(i+1))
        #ax[mode,1].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i,1], shrink=0.8, aspect=10)

        f, Pxx_den = signal.welch(temporalModes[:,mode], fs, nperseg=512*4)
        ax[i,2].plot(f, Pxx_den, color='lightseagreen')
        ax[i,2].axis(xmin=0,xmax=.15)
        annot_max(f,Pxx_den, ax=ax[i,2])
        ax[i,2].grid(color='whitesmoke', zorder=1)
        ax[i,2].spines['top'].set_visible(False)
        ax[i,2].spines['right'].set_visible(False)
        if mode == (numberModes-1):
            ax[i,0].set_xlabel('$x/c$')
            ax[i,1].set_xlabel('$x/c$')
            ax[i,2].set_xlabel('$f/f_c$')

    plt.savefig('modes.png', format='png', bbox_inches="tight")

def correlationMatrix(temporalModes):
    import matplotlib.pyplot as plt
    latent_dim = temporalModes.shape[1]
    corr_matrix_latent = abs(np.corrcoef(temporalModes.T))

    fig, ax = plt.subplots(figsize=(10 * latent_dim / 10, 10 * latent_dim / 10), dpi=80)
    im = ax.imshow(corr_matrix_latent, cmap='GnBu', vmin=0, vmax=1)
    ax.grid(False)
    for i in range(latent_dim):
        for j in range(latent_dim):
            ax.text(j, i, round(corr_matrix_latent[i, j], 2), ha='center', va='center',
                    color='k')
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f', shrink=0.8)
    plt.title(str(np.linalg.det(corr_matrix_latent)))

    plt.savefig('matrix.png', format='png', bbox_inches="tight")
    plt.show()

def plotTemporalSeries(modes):
    latent_dim = modes.shape[1]
    fig, ax = plt.subplots(latent_dim, 1, figsize=(16, latent_dim * 1.6), sharex='col')

    for i in range(modes.shape[1]):
        ax[i].plot(modes[:1000,i], label='Mode '+str(i))
        ax[i].axis(ymin=-2.5, ymax=2.5)

        ax[i].legend()
        ax[i].grid()


    plt.savefig('series.png', format='png', bbox_inches="tight")
    plt.show()

def plotSamples(rec_train, rec_test, true_train, true_test):
    print(rec_train.shape)

    Ulim = max(abs(rec_train[0].flatten()))
    Vlim = max(abs(rec_train[1].flatten()))

    fig, ax = plt.subplots(4, 2, figsize=(16, 4 * 1.6))

    axs = ax[0, 0]
    im = axs.imshow(true_train[0], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])

    axs.set_title('True train u')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[1, 0]
    im = axs.imshow(true_train[1], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('True train v')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[2, 0]
    im = axs.imshow(true_test[0], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('True test u')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[3, 0]
    im = axs.imshow(true_test[1], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('True test v')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)
    ###########################################################################################
    axs = ax[0, 1]
    im = axs.imshow(rec_train[0], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])

    axs.set_title('Rec train u')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[1, 1]
    im = axs.imshow(rec_train[1], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('Rec train v')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[2, 1]
    im = axs.imshow(rec_test[0], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('Rec test u')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)

    axs = ax[3, 1]
    im = axs.imshow(rec_test[1], cmap="RdBu_r", vmin=-Ulim, vmax=Ulim,
                         extent=[-9, 87, -14, 14])
    axs.set_title('Rec test v')
    fig.colorbar(im, ax=axs, shrink=0.8, aspect=10)
    fig.set_tight_layout(True)
    plt.show()

def plotEk_t(Ek):
    plt.plot(Ek)
    plt.xlabel('Test data index')
    plt.ylabel('Ek')
    plt.show()

def plotEcum(Ecum):
    plt.plot(np.arange(1, Ecum.shape[0]+1), Ecum)
    plt.xlabel('Number of modes')
    plt.ylabel('Cumulative Ek')
    plt.grid()
    plt.show()

def plotNLmodePoint(mode, values):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    print(mode.shape)

    ax[0].plot(values, mode[:,0,22,150])
    ax[0].plot(values, mode[:,0,33,150])
    ax[0].plot(values, mode[:,0,44,150])
    ax[0].plot(values, mode[:,0,55,150])
    ax[0].plot(values, mode[:,0,66,150])
    ax[0].set_xlabel('Latent input')
    ax[0].set_ylabel('Mode values at field points')
    ax[0].grid()

    ax[1].plot(values, mode[:,1,22,150])
    ax[1].plot(values, mode[:,1,33,150])
    ax[1].plot(values, mode[:,1,44,150])
    ax[1].plot(values, mode[:,1,55,150])
    ax[1].plot(values, mode[:,1,66,150])
    ax[1].set_xlabel('Latent input')
    ax[1].set_ylabel('Mode values at field points')
    ax[1].grid()

    ax[0].set_title('u')
    ax[1].set_title('v')
    fig.set_tight_layout(True)

    plt.show()

def plotNLmodeField(mode, values):

    valuesToPlot = np.array([-2., -1., 0., 1., 2.])

    # Find the indices of values_array elements in main_array
    indices = []
    for value in valuesToPlot:
        matching_indices = np.where(np.abs(values - value) < 1e-3)[0]
        indices.extend(matching_indices)

    indices_array = np.array(indices)
    values = values[indices_array]

    fig, ax = plt.subplots(len(indices_array), 2, figsize=(8, 8), sharex='col', sharey='row')

    for idx, value in enumerate(values):
        Uplot = modes[idx,0,:,:]
        Vplot = modes[idx,1,:,:]

        #Ulim = max( abs(modes[:,0,:,:].flatten()) )
        #Vlim = max( abs(modes[:,1,:,:].flatten()) )

        Ulim = np.std(modes[:,0,:,:].flatten()) * 4
        Vlim = np.std(modes[:,1,:,:].flatten()) * 4

        im = ax[idx,0].imshow(Uplot, cmap="RdBu_r", vmin=-Ulim, vmax=Ulim, extent=[-9, 87, -14, 14])
        ax[idx,0].set_title('Value {}'.format(round(value,1)))
        ax[idx,0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[idx,0], shrink=0.8, aspect=10)

        im = ax[idx,1].imshow(Vplot, cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
        #ax[idx,1].set_title('Mode {}, value {}, v'.format(mode, round(NLvalues[value],2)))
        fig.colorbar(im, ax=ax[idx,1], shrink=0.8, aspect=10)


    ax[idx,0].set_xlabel('x/c')
    ax[idx,1].set_xlabel('x/c')

    plt.show()

if __name__ == "__main__":
    import h5py
    import numpy as np

    file = '04_modes/20230711_14_19_beta0.05_dim50_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py'

    logfile = '02_logs/' + file.split('_modes.')[0].split('/')[-1].split('_epoch_')[0]
    print(logfile)
    plot_training(logfile)

    with h5py.File(file, 'r') as f:
        print("Keys: %s" % f.keys())
        temporalModes = f['vector'][:, :]
        temporalModes_test = f['vector_test'][:, :]
        modes = f['modes'][:, :]
        zero_output = f['zero_output'][:, :]
        mean_data = f['mean'][:,:,:,:]
        std_data = f['std'][:,:,:,:]
        order = f['order'][:]
        Ecum = f['Ecum'][:]
        NLvalues = f['NLvalues'][:]
        NLmodes = f['NLmodes'][:]

    plotNLmodePoint(NLmodes, NLvalues)
    plotNLmodeField(NLmodes, NLvalues)
    plotCompleteModes(modes, temporalModes, modes.shape[0])
    plt.show()

    plotTemporalSeries(temporalModes)
    correlationMatrix(temporalModes)

    plotEcum(Ecum)