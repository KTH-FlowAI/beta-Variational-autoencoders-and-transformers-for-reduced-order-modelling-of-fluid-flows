import matplotlib.pyplot as plt
import torch
import numpy as np
import lib_model
import lib_evaluate
import lib_data


def getVAE(nmodes, device):
    if nmodes == 20:
        weights_file = '20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final.pth.tar'
    elif nmodes == 10:
        weights_file = '20230915_13_46_smallerCNN_beta0.05_wDecay0_dim10_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final.pth.tar'

    # Get model
    model = lib_model.VAE(latent_dim=nmodes).to(device)
    model.eval()

    # Load weights
    lib_model.load_checkpoint(model=model, path_name='03_checkpoints/' + weights_file)

    return model


def predFieldFigure(true, VAErec, pred, std_data, mean_data, model_name, stepPlot):
    fig, ax = plt.subplots(2, 5, figsize=(18, 3), sharex='col', sharey='row')  # , dpi=300)

    Umax = 1.5
    Umin = 0
    Vlim = 1

    # From dataset
    im = ax[0, 0].imshow(true[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :],
                         cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    ax[0, 0].set_title('True u, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 0], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 0].imshow(true[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :],
                         cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    ax[1, 0].set_title('True v, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7)

    # Encoded and decoded
    im = ax[0, 1].imshow(VAErec[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :],
                         cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    ax[0, 1].set_title(r'$\beta$VAE u, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 1].imshow(VAErec[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :],
                         cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    ax[1, 1].set_title(r'$\beta$VAE v, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 1], shrink=0.7)

    # Encoded, predicted and decoded
    im = ax[0, 2].imshow(pred[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :],
                         cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    # ax[0,2].set_title(r'$\beta$VAE + transformer u, ($t+$' + (str(stepPlot) if stepPlot > 1 else "")+'$t_c$)')
    ax[0, 2].set_title(r'$\beta$VAE + ' + model_name + ' u, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 2], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 2].imshow(pred[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :],
                         cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    # ax[1,2].set_title(r'$\beta$VAE + transformer v, ($t+$' + (str(stepPlot) if stepPlot > 1 else "")+'$t_c$)')
    ax[1, 2].set_title(r'$\beta$VAE + ' + model_name + ' v, ($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 2], shrink=0.7)

    # error encoded
    im = ax[0, 3].imshow(np.abs((VAErec[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :]) -
                                (true[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :])),
                         cmap="nipy_spectral", vmin=0, vmax=1, extent=[-9, 87, -14, 14])
    ax[0, 3].set_title(r'Error encoded')
    fig.colorbar(im, ax=ax[0, 3], shrink=0.7)

    im = ax[1, 3].imshow(np.abs((VAErec[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]) -
                                (true[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :])),
                         cmap="nipy_spectral", vmin=0, vmax=1, extent=[-9, 87, -14, 14])
    ax[1, 3].set_title(r'Error encoded')
    fig.colorbar(im, ax=ax[1, 3], shrink=0.7)

    # error predicted
    im = ax[0, 4].imshow(np.abs((pred[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :]) -
                                (true[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :])),
                         cmap="nipy_spectral", vmin=0, vmax=1, extent=[-9, 87, -14, 14])
    ax[0, 4].set_title(r'Error predicted')
    fig.colorbar(im, ax=ax[0, 4], shrink=0.7)

    im = ax[1, 4].imshow(np.abs((pred[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]) -
                                (true[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :])),
                         cmap="nipy_spectral", vmin=0, vmax=1, extent=[-9, 87, -14, 14])
    ax[1, 4].set_title(r'Error predicted')
    fig.colorbar(im, ax=ax[1, 4], shrink=0.7)

    ax[1, 0].set_xlabel('x/c')
    ax[1, 1].set_xlabel('x/c')
    ax[1, 2].set_xlabel('x/c')
    ax[1, 3].set_xlabel('x/c')
    ax[1, 4].set_xlabel('x/c')

    ax[0, 0].set_ylabel('y/c')
    ax[1, 0].set_ylabel('y/c')

    fig.set_tight_layout(True)
    fig.show()

    # plt.savefig('predFieldFigure_t'+str(stepPlot)+model_name+'.eps', format='eps', dpi=300, bbox_inches = "tight")

    return fig, ax


fieldPredStep_tc = 10

files = [
    # '06_Preds/dim10/LSTM_64in_64dmodel_1next_10dim_Noneemb_128hideen_4nlayer__Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    # '06_Preds/dim10/VAE_easyAttn_64in_64dmodel_1next_10dim_timeemb_4h_4nb_128ff_reluact_Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    '06_Preds/dim20/LSTM_64in_64dmodel_1next_20dim_Noneemb_128hideen_4nlayer__Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    '06_Preds/dim20/VAE_easyAttn_64in_64dmodel_1next_20dim_timeemb_4h_4nb_128ff_reluact_Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    # '06_Preds/dim10/KNF_10dim_5in_3_4_5Interval.npz',
    '06_Preds/dim20/KNF_20dim_5in_3_4_5Interval.npz'
]

device = torch.device('cuda')

# Scan model names
nmodes = np.zeros(len(files), dtype=np.int16)
modelNames = []
interval = []
for idx, file in enumerate(files):
    nmodes[idx] = int(file.split('dim')[1].split('_')[-1])
    if 'Interval' in file:
        interval.append(int(file.split('_')[-1].split('Interval')[0]))
    else:
        interval.append(1)

    if 'KNF' in file:
        modelNames.append('KNF')
    else:
        modelNames.append(file.split('/')[2].split('_64in_')[0].split('_')[-1])
    print(nmodes[idx], modelNames[idx], interval[idx])

datafile = '01_data/Re100alpha10_newData_150000.hdf5'
u_scaled, mean, std = lib_data.loadData(datafile, printer=True)
n_test = 15000
n_total = u_scaled.shape[0]
n_train = n_total - n_test
print(f"N train: {n_train:d}, N test: {n_test:d}, N total {n_total:d}")
u_test = u_scaled[n_train:]

diffNmodes = np.unique(nmodes)
# Load one VAE for all predictors with same nmodes
for uniqNmode in diffNmodes:
    print(f'Load VAE for {uniqNmode} modes')
    VAE = getVAE(uniqNmode, device)
    true = None

    # Load each predictor
    indexesNmode = np.where(nmodes == uniqNmode)[0]
    for idx in indexesNmode:
        file = files[[idx][0]]
        print(file)

        predLatent = np.load(file)['p'][:500].astype(np.float32)

        pred = lib_evaluate.decode(VAE, predLatent, device)
        if true is None:
            trueLatent = np.load(file)['g'][:len(predLatent)]
            trueDec = lib_evaluate.decode(VAE, trueLatent, device)
            true = u_test[:len(predLatent)]

        delta_t = 1 / 5 * interval[idx]
        fieldPredStep = int(fieldPredStep_tc / delta_t)
        print(f'Delta t = {delta_t}, steps = {fieldPredStep}')

        predFieldFigure(true[64:], trueDec[64:], pred[64:], std, mean,
                        f'{modelNames[idx]}, {nmodes[idx]}m', fieldPredStep_tc)
        plt.savefig(f"05_Figs/vis_pred/predField_{nmodes[idx]}_{fieldPredStep_tc}tc_{modelNames[idx]}.png",
                    bbox_inches="tight")

plt.show()
