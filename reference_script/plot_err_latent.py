
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tbparse import SummaryReader

def loadLog(logFile):
    df = SummaryReader(logFile, pivot=True).scalars
    df = df[['step', 'General loss/KLD', 'General loss/KLD_test', 'General loss/MSE',
             'General loss/MSE_test', 'General loss/Total',
             'General loss/Total_test']]
    list(df.columns)
    return df['step'].to_numpy(), df['General loss/KLD'].to_numpy(), df['General loss/MSE'].to_numpy(), df[
        'General loss/MSE_test'].to_numpy(), df['General loss/KLD_test'].to_numpy()

files = ['04_modes/20230716_13_38_beta0.05_dim10_lr0.0001_bs256_epochs200_Wdecay0.0005_epoch_bestTest_modes.h5py',
         '04_modes/20230712_12_36_beta0.05_dim20_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py',
         '04_modes/20230712_09_30_beta0.05_dim30_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py',
         '04_modes/20230712_20_34_beta0.05_dim40_lr0.0001_bs256_epochs1000_Wdecay0.0005_epoch_final_modes.h5py',
         '04_modes/20230711_14_19_beta0.05_dim50_lr0.0001_bs256_epochs1000_Wdecay0.0005_modes.h5py']

MSE_train = np.zeros(len(files))
MSE_test = np.zeros(len(files))
nmodes = np.zeros(len(files), dtype=np.int16)

for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[-1].split('_')[0])
    logfile = '02_logs/' + file.split('_modes.')[0].split('/')[-1].split('_epoch_')[0]
    print(logfile)
    step, KL, MSE, MSE_t, KL_t = loadLog(logfile)

    MSE_train[idx] = MSE[-1]
    MSE_test[idx] = MSE_t[-1]

plt.plot(nmodes, MSE_train, 'o-', label='MSE train')
plt.plot(nmodes, MSE_test, 's--', label='MSE test')
plt.legend()
plt.xlabel('Latent space size')
plt.xticks(nmodes)
plt.show()

