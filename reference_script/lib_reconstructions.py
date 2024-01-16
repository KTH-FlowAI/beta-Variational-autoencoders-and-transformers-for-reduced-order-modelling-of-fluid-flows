import torch
import numpy as np
import h5py
import lib_data
import lib_model


def predict(model, data, device):
    from tqdm import tqdm
    reconstructions = []

    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=512,
                                                shuffle=False, pin_memory=True, num_workers=2)

    with torch.no_grad():
        for batch in tqdm(dataset):
            batch = batch.to(device, non_blocking=True)
            rec, _, _ = model(batch)
            reconstructions.append(rec.cpu().numpy())

    reconstructions = np.concatenate(reconstructions, axis=0)

    print(data.shape)
    print(reconstructions.shape)

    return reconstructions


if __name__ == "__main__":

    # weights file
    weights_file = '03_checkpoints/20230915_10_21_smallerCNN_beta0.05_wDecay0_dim20_lr0.0002OneCycleLR1e-05_bs256_epochs1000_nt27000_epoch_final.pth.tar'

    # Get system info
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))
    device = torch.device('cuda')

    # load data
    datafile = '01_data/Re100alpha10_newData_150000.hdf5'

    # parameters
    batch_size = 256

    u_scaled, mean, std = lib_data.loadData(datafile)

    out_name = weights_file.split('.pth')[0].split('/')[-1]
    out_name = '04_modes/' + out_name + '_reconstructions.h5py'

    latent_dim = int(weights_file.split('dim')[-1].split('_')[0])
    print(f'Latent dim: {latent_dim:2d}')

    # Get model
    model = lib_model.VAE(latent_dim=latent_dim).to(device)
    model.eval()

    # Load weights
    lib_model.load_checkpoint(model=model, path_name=weights_file)

    # Use VAE
    recs = predict(model, u_scaled, device)

    out_name = weights_file.split('.pth')[0].split('/')[-1]
    out_name = '04_modes/' + out_name + '_reconstruction.h5py'

    print(out_name)

    with h5py.File(out_name, 'w') as f:
        f.create_dataset('UV', data=recs, dtype='float32')
        f.create_dataset('mean', data=mean, dtype='float32')
        f.create_dataset('std', data=std, dtype='float32')