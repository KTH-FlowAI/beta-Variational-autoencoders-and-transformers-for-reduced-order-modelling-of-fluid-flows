import matplotlib.pyplot as plt
import h5py
from scipy import signal
import numpy as np

file = '08_POD/POD_twoPlatesRe40_Data.h5py'
case = file.split('/')[-1].split('.')[0]
print(case)

def annot_max(x,y, ax=None, xytext=(0.94,0.96)):
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
    ax.annotate(text, xy=(xmax, ymax), xytext=xytext, **kw)


def modesPlot(Phi_x, Phi_y, Psi, fs, firstN):
    fig, ax = plt.subplots(firstN, 3, figsize=(12, 1.5*firstN), sharex='col')

    for m in range(firstN):

        Ulim = max( abs(Phi_x[m, :].flatten()) )
        Vlim = max( abs(Phi_y[m, :].flatten()) )

        im = ax[m, 0].imshow(Phi_x[m, :], cmap='RdBu_r', extent=[-9, 87, -14, 14], vmin=-Ulim, vmax=Ulim)
        fig.colorbar(im, ax=ax[m, 0], shrink=0.8, aspect=10)
        im = ax[m, 1].imshow(Phi_y[m, :], cmap='RdBu_r', extent=[-9, 87, -14, 14], vmin=-Vlim, vmax=Vlim)
        fig.colorbar(im, ax=ax[m, 1], shrink=0.8, aspect=10)
        ax[m, 1].set_title('Mode ' + str(m + 1))
        ax[m,0].set_ylabel('y/c')

        f, Pxx_den = signal.welch(Psi[:, m], fs, nperseg=512)
        ax[m, 2].plot(f / fs, Pxx_den/max(Pxx_den), color='lightseagreen')
        ax[m, 2].axis(xmin=0, xmax=.15 / fs)
        annot_max(f / fs, Pxx_den/max(Pxx_den), ax=ax[m, 2], xytext=(0.54, 0.96))
        ax[m, 2].grid(color='whitesmoke', zorder=1)
        ax[m, 2].spines['top'].set_visible(False)
        ax[m, 2].spines['right'].set_visible(False)
        if m == (firstN - 1):
            ax[m, 0].set_xlabel('$x/c$')
            ax[m, 1].set_xlabel('$x/c$')
            ax[m, 2].set_xlabel('$f/f_c$')

    fig.set_tight_layout(True)
    return fig, ax

with h5py.File(file, 'r') as f:
    Psi_train = f['Psi_train'][:]
    phi_u, phi_v = np.split(f['phi_train'][:], 2)

phi_u = np.reshape(phi_u, (88, 300, -1))
phi_v = np.reshape(phi_v, (88, 300, -1))
phi_u = np.moveaxis(phi_u, -1, 0)
phi_v = np.moveaxis(phi_v, -1, 0)

print(f'Psi_train shape: {Psi_train.shape}')
print(f'phi_u shape: {phi_u.shape}')
print(f'phi_v shape: {phi_v.shape}')

firstN = 2
fs = 1

fig, ax = modesPlot(phi_u, phi_v, Psi_train, fs, firstN)
#plt.figtext(0.06, 0.95, 'a)', size=14)


fig.set_tight_layout(True)
plt.savefig(f"05_Figs/{case}.png", bbox_inches="tight")
plt.savefig(f"05_Figs/{case}.eps", format='eps', bbox_inches="tight")
plt.show()
