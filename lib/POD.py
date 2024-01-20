import h5py
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd


class POD:
    def __init__(self, datafile, n_test, re, path, n_modes=100, delta_t=1) -> None:
        """
        A runner for POD

        Args:

            datafile        :       (Str) Path of training data
            n_test          :       (int) Number of test timesteps (not used for POD)
            re              :       (int) Reynolds number
            path            :       (str) Path to save POD results
            n_modes         :       (int) Number of POD modes to calculate
            delta_t         :       (int) Steps between snapshots to use
        """

        self.datafile = datafile
        self.n_modes = n_modes
        self.delta_t = delta_t
        self.n_test = n_test
        self.re = re
        self.casename = f'POD_Re{self.re}_dt{self.delta_t}_ntst{self.n_test}_nm{self.n_modes}.npz'
        self.filename = path + self.casename

        print(f"POD file name:\n {self.filename}")

    def load_data(self):
        # load data
        with h5py.File(self.datafile, 'r') as f:
            u_scaled = f['UV'][:]
            mean = f['mean'][:]
            std = f['std'][()]

        u_scaled = np.moveaxis(u_scaled, -1, 1)
        self.mean = np.moveaxis(mean, -1, 1)
        self.std = np.moveaxis(std, -1, 1)

        u_scaled = u_scaled[::self.delta_t]

        n_total = u_scaled.shape[0]
        self.n_train = n_total - self.n_test
        print(f"N train: {self.n_train:d}, N test: {self.n_test:d}, N total {n_total:d}")
        print(f'u_scaled {u_scaled.shape}')

        self.u_train = u_scaled[:self.n_train]
        self.u_test = u_scaled[self.n_train:]

    def get_POD(self):

        try:
            self.load_POD()
            print('POD loaded from file')
        except:
            print('Calculating POD')
            self.calc_POD()

    def load_POD(self):
        d = np.load(self.filename)
        self.temporal_modes = d['tm']
        self.spatial_modes = d['sm']
        self.eigVal = d['eig']

    def calc_POD(self):

        u_train_flat = self.u_train.reshape(self.u_train.shape[0], -1)
        u_test_flat = self.u_test.reshape(self.u_test.shape[0], -1)

        print(f'POD u_train: {u_train_flat.shape}')
        print(f'POD u_test: {u_test_flat.shape}')

        print(f'U shape: {u_train_flat.shape}')
        # C matrix
        print('Calc C matrix', end="")
        start = time.time()
        C = u_train_flat.T.dot(u_train_flat)
        C = C / (self.n_train - 1)
        print(f': {(time.time() - start):.1f}s')
        print(f'C shape: {C.shape}')

        # SVD
        print('Calc SVD', end="")
        start = time.time()
        self.spatial_modes, self.eigVal, _ = randomized_svd(C, n_components=self.n_modes, random_state=0)
        print(f': {(time.time() - start):.1f}s')
        print(f'spatial_modes {self.spatial_modes.shape}')

        self.temporal_modes = u_train_flat.dot(self.spatial_modes)

        print(f'temporal_modes shape: {self.temporal_modes.shape}')
        print(f'spatial_modes shape: {self.spatial_modes.shape}')

        np.savez_compressed(
            file=self.filename,
            tm=self.temporal_modes,
            sm=self.spatial_modes,
            eig=self.eigVal
        )

    def eval_POD(self):
        from lib.pp_space import get_Ek

        self.Ek_nm = np.zeros(self.n_modes)

        for nm in range(1, self.n_modes + 1):
            u_train_rec = self.temporal_modes[:, :nm].dot(self.spatial_modes[:, :nm].T).reshape(self.u_train.shape)

            self.Ek_nm[nm - 1] = get_Ek(self.u_train, u_train_rec)
            # print(f'POD train E = {self.Ek_nm[nm-1]:.4f}, {nm} modes')

        print(f'E POD: {self.Ek_nm}')


# Test code
if __name__ == "__main__":
    # data
    datafile = "../data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5"

    re = 40
    delta_t = 1
    n_modes = 10
    n_test = 200 // delta_t

    POD = POD(datafile, n_test, re, '../res/', n_modes, delta_t)
    POD.load_data()
    POD.get_POD()
    POD.eval_POD()
