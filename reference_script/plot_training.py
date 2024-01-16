import matplotlib.pyplot as plt
import h5py
import numpy as np

"""Plot from tensorboard file"""
from tbparse import SummaryReader
import matplotlib.pyplot as plt


def loadLog(logFile):
    df = SummaryReader(logFile, pivot=True).scalars
    df = df[['step', 'General loss/KLD', 'General loss/KLD_test', 'General loss/MSE',
             'General loss/MSE_test', 'General loss/Total',
             'General loss/Total_test']]
    list(df.columns)
    return df['step'].to_numpy(), df['General loss/KLD'].to_numpy(), df['General loss/MSE'].to_numpy(), df[
        'General loss/MSE_test'].to_numpy(), df['General loss/KLD_test'].to_numpy()




files = ['04_modes/20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py',
         '04_modes/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final_modes.h5py']

fig, ax = plt.subplots(1, 2,figsize=(7, 2))

logfile = '02_logs/' + files[0].split('_modes.')[0].split('/')[-1].split('_epoch_')[0]
print(logfile)
step, KL, MSE, MSE_t, KL_t = loadLog(logfile)
ax1 = ax[0]
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax1.plot(step, KL, '-', label='KL', color='lightseagreen')
ax2.plot(step, MSE, '-', label='MSE', color='saddlebrown')
ax1.plot(step, KL_t, ':', label='KL test', color='lightseagreen')
ax2.plot(step, MSE_t, ':', label='MSE test', color='saddlebrown')

ax1.set_xlabel('Training epoch')
ax1.set_ylabel('KL loss', color='lightseagreen')
ax1.set_ylim(0.5, 1.5)
ax2.set_ylim(0, 1)
ax1.tick_params(axis='y', labelcolor='lightseagreen')

ax2.set_ylabel('Reconstruction loss', color='saddlebrown')
ax2.tick_params(axis='y', labelcolor='saddlebrown')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(color='whitesmoke', zorder=1)



logfile = '02_logs/' + files[1].split('_modes.')[0].split('/')[-1].split('_epoch_')[0]
print(logfile)
step, KL, MSE, MSE_t, KL_t = loadLog(logfile)
ax1 = ax[1]
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax1.plot(step, KL, '-', label='KL', color='lightseagreen')
ax2.plot(step, MSE, '-', label='MSE', color='saddlebrown')
ax1.plot(step, KL_t, ':', label='KL test', color='lightseagreen')
ax2.plot(step, MSE_t, ':', label='MSE test', color='saddlebrown')

ax1.set_xlabel('Training epoch')
ax1.set_ylabel('KL loss', color='lightseagreen')
ax1.set_ylim(0.5, 1.5)
ax2.set_ylim(0, 1)
ax1.tick_params(axis='y', labelcolor='lightseagreen')

ax2.set_ylabel('Reconstruction loss', color='saddlebrown')
ax2.tick_params(axis='y', labelcolor='saddlebrown')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(color='whitesmoke', zorder=1)

#plt.legend()
fig.tight_layout(pad=1.5)

plt.show()