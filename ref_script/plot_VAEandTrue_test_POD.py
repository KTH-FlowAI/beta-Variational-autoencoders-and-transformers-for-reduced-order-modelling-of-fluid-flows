import matplotlib.pyplot as plt
import h5py
from scipy import signal
import numpy as np

# file_VAE = '08_POD/POD_twoPlatesRe100_reconstructed_OnlyTest.h5py'
file_VAE = '08_POD/POD_twoPlatesRe100_predicted_OnlyTest.h5py'
file_true = '08_POD/POD_twoPlatesRe100_tc1_OnlyTest.h5py'

with h5py.File(file_VAE, 'r') as f:
    Psi_VAE = f['Psi_train'][:]
    phi_u_VAE, phi_v_VAE = np.split(f['phi_train'][:], 2)

with h5py.File(file_true, 'r') as f:
    Psi_true = f['Psi_train'][:]
    phi_u_true, phi_v_true = np.split(f['phi_train'][:], 2)

phi_u_VAE = np.reshape(phi_u_VAE, (88, 300, -1))
phi_v_VAE = np.reshape(phi_v_VAE, (88, 300, -1))
phi_u_VAE = np.moveaxis(phi_u_VAE, -1, 0)
phi_v_VAE = np.moveaxis(phi_v_VAE, -1, 0)

phi_u_true = np.reshape(phi_u_true, (88, 300, -1))
phi_v_true = np.reshape(phi_v_true, (88, 300, -1))
phi_u_true = np.moveaxis(phi_u_true, -1, 0)
phi_v_true = np.moveaxis(phi_v_true, -1, 0)

print(f'Psi_VAE shape: {Psi_VAE.shape}')
print(f'Psi_true shape: {Psi_true.shape}')
print(f'phi_u_VAE shape: {phi_u_VAE.shape}')
print(f'phi_u_true shape: {phi_u_true.shape}')
print(f'phi_v_VAE shape: {phi_v_VAE.shape}')
print(f'phi_v_true shape: {phi_v_true.shape}')

firstN = 4
fs = 1

def line(ax, row, mode_u, mode_v, Ulim,  Vlim):

    im = ax[row, 0].imshow(mode_u, cmap='RdBu_r', extent=[-9, 87, -14, 14], vmin=-Ulim, vmax=Ulim)
    # fig.colorbar(im, ax=ax[row, 0], shrink=0.8, aspect=10)
    ax[row, 0].text(0.9, 0.15, 'u', fontsize=10, transform=ax[row, 0].transAxes, bbox=dict(facecolor='white', alpha=0.5))
    im = ax[row, 1].imshow(mode_v, cmap='RdBu_r', extent=[-9, 87, -14, 14], vmin=-Vlim, vmax=Vlim)
    # fig.colorbar(im, ax=ax[row, 1], shrink=0.8, aspect=10)
    ax[row, 1].text(0.9, 0.15, 'v', fontsize=10, transform=ax[row, 1].transAxes, bbox=dict(facecolor='white', alpha=0.5))



fig, ax = plt.subplots(ncols=3, nrows=firstN*2, sharex='col', sharey='row', figsize=(9, 2 * firstN))

for m in range(firstN):

    Ulim = max(abs(phi_u_true[m].flatten()))
    Vlim = max(abs(phi_v_true[m].flatten()))

    gs = ax[1, 2].get_gridspec()
    # remove the underlying axes
    for axx in ax[2 * m:2 * m + 2, -1]:
        axx.remove()
    axbig = fig.add_subplot(gs[2 * m:2 * m + 2, -1])


    f, Pxx_den = signal.welch(Psi_true[:, m], fs, nperseg=512)
    axbig.plot(f / fs, Pxx_den / max(Pxx_den), color='k', label='True')
    f, Pxx_den = signal.welch(Psi_VAE[:, m], fs, nperseg=512)
    axbig.plot(f / fs, Pxx_den / max(Pxx_den), color='lightseagreen', label='ROM')
    axbig.axis(xmin=0, xmax=.15 / fs)
    axbig.grid(color='whitesmoke', zorder=1)
    axbig.spines['top'].set_visible(False)
    axbig.spines['right'].set_visible(False)
    axbig.legend()
    if m != firstN-1:
        axbig.set_xticklabels([])
    else:
        pass
        axbig.set_xticks([0, 0.05, 0.1, 0.15])

    axbig.set_yticks([0, 0.5, 1])
    axbig.set_title(f'Mode {m + 1}', y=0.96, fontsize=11)
    # axbig.set_title(f'Mode {m + 1}')

    line(ax, row=m*2, mode_u=phi_u_true[m, :], mode_v=phi_v_true[m, :], Ulim=Ulim, Vlim=Vlim)
    ax[m*2, 1].set_title(' ', fontsize=10)  # to make room
    fig.text(0.35, 0.965-m*.232, f'POD mode {m + 1}, true data', fontsize=11, ha="center")
    ax[m*2, 0].set_ylabel('y/c')
    ax[m*2+1, 0].set_ylabel('y/c')
    line(ax, row=m*2+1, mode_u=phi_u_VAE[m, :], mode_v=phi_v_VAE[m, :], Ulim=Ulim, Vlim=Vlim)
    ax[m*2+1, 1].set_title(' ', fontsize=10)  # to make room
    fig.text(0.35, 0.965-.232/2-m*.232, fr'POD mode {m + 1}, easyAttn+$\beta$-VAE', fontsize=11, ha="center")

    if m == (firstN - 1):
        ax[m*2+1, 0].set_xlabel('$x/c$')
        ax[m*2+1, 1].set_xlabel('$x/c$')
        axbig.set_xlabel('$f/f_c$')

#fig.set_tight_layout(True)
plt.subplots_adjust(left=0.05,
                    bottom=0.07,
                    right=0.95,
                    top=0.96,
                    wspace=0.1,
                    hspace=0.5)
plt.savefig(f"05_Figs/PODpreds.png", bbox_inches="tight")
plt.savefig(f"05_Figs/PODpreds.eps", bbox_inches="tight", format='eps')
plt.show()

