import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

files = [
    '06_Preds/dim20/LSTM_64in_64dmodel_1next_20dim_Noneemb_128hideen_4nlayer__Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    '06_Preds/dim20/VAE_easyAttn_64in_64dmodel_1next_20dim_timeemb_4h_4nb_128ff_reluact_Noneoutact_1000Epoch_135000N_FalseES_0P_5Interval.npz',
    '06_Preds/dim20/KNF_20dim_10in_2_3_5Interval.npz',
]

color_idx = [2, 1, 0]

firstN = 6
seriesLen = 200

nmodes = np.zeros(len(files), dtype=np.int16)
interval = []
modelNames = []
colors = []
in_len = np.zeros(len(files), dtype=np.int16)

for model_idx, file in enumerate(files):
    nmodes[model_idx] = int(file.split('dim')[1].split('_')[-1])
    in_len[model_idx] = int(file.split('in_')[0].split('_')[-1])
    colors.append(sns.color_palette("tab10")[model_idx])
    print(sns.color_palette("tab10")[model_idx])
    if 'Interval' in file:
        interval.append(int(file.split('_')[-1].split('Interval')[0]))
    else:
        interval.append(1)

    if 'KNF' in file:
        modelNames.append('KNF')
    else:
        modelNames.append(file.split('/')[2].split('_64in_')[0].split('_')[-1])
    print(nmodes[model_idx], modelNames[model_idx], interval[model_idx], in_len[model_idx])

diffNmodes = np.unique(nmodes)
diffModels = list(set(modelNames))
print(diffModels)

# Load one VAE for all predictors with same nmodes
for i, uniqNmode in enumerate(diffNmodes):
    print(f'Series for {uniqNmode} modes ######################################################################')
    indexesNmode = np.where(nmodes == uniqNmode)[0]

    nplots = min(uniqNmode, firstN)
    fig, ax = plt.subplots(nplots, 1, figsize=(1.5, 6), sharex='col', sharey='row')
    truePlotted = False

    for model_idx in indexesNmode:
        print(modelNames[model_idx])
        file = files[[model_idx][0]]
        order = np.load(file)['o']
        true = np.load(file)['t'][:seriesLen, order]
        pred = np.load(file)['p'][:seriesLen, order]
        print(f'True: {true.shape}, pred: {pred.shape}')
        len = min(seriesLen, true.shape[0])

        for mode in range(nplots):
            if not truePlotted:
                sns.distplot(true[:, mode], hist=False, kde=True, kde_kws={'linewidth': 1.5, 'color': 'k'},
                             label='True', ax=ax[mode], axlabel=None)

            sns.distplot(pred[:, mode], hist=False, kde=True, kde_kws={'linewidth': 1.5},
                         color=colors[color_idx[model_idx]],
                         label=modelNames[model_idx], ax=ax[mode], axlabel=None)

            ax[mode].grid(color='gainsboro', zorder=1)
            ax[mode].spines['right'].set_visible(False)
            ax[mode].spines['left'].set_visible(False)
            ax[mode].spines['top'].set_visible(False)
            ax[mode].set_xlim([-3, 3])
            ax[mode].set_xticks([-2, 0, 2])
            ax[mode].set_yticks([])
            ax[mode].set_ylabel(None)
            ax[mode].set_title(f'Mode {mode + 1}', fontsize=11)
            # ax[mode].text(0.05, 0.75, 'Mode '+str(mode+1), fontsize=10, transform=ax[mode].transAxes, bbox=dict(facecolor='white', alpha=0.5))

        truePlotted = True

    fig.set_tight_layout(True)

    handles, labels = ax[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, ncol=1, bbox_to_anchor=(0, 1.00, 1, 0.2), loc="lower center")

    plt.savefig(f"05_Figs/vis_pred/PDFpreds_{uniqNmode}_vertical.png", bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=300)
    # plt.savefig(f"05_Figs/vis_pred/PDFpreds_{uniqNmode}_vertical.eps", bbox_extra_artists=(lgd,), format='eps',
    #             bbox_inches="tight")

# plt.show()
