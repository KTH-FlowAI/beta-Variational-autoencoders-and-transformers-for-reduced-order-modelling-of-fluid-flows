import h5py
import numpy as np

def loadData(file, printer=False):
    with h5py.File(file, 'r') as f:
        u_scaled = f['UV'][:]
        mean = f['mean'][:]
        std = f['std'][()]

    u_scaled = np.moveaxis(u_scaled, -1, 1)
    mean = np.moveaxis(mean, -1, 1)
    std = np.moveaxis(std, -1, 1)

    if printer:
        print('u_scaled: ', u_scaled.shape)
        print('mean: ', mean.shape)
        print('std: ', std)

    return u_scaled, mean, std


'''Testing code'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    u_scaled, mean, std = loadData('01_data/Re100alpha5_newData_v1_slice.hdf5', printer=True)

    plt.imshow(u_scaled[10, 0, :, :], cmap="RdBu_r")
    plt.colorbar()
    plt.title('u´')
    plt.show()

    plt.imshow(mean[0, 0, :, :], cmap="RdBu_r")
    plt.colorbar()
    plt.title('mean')
    plt.show()

    plt.hist(u_scaled[0:100].flatten())
    plt.title('u´ distribution')
    plt.show()
