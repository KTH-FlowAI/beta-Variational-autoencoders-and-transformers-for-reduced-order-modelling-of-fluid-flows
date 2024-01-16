# -*- coding: utf-8 -*-
"""AE twoPlates_preprocess_newData_v1.ipynb"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import glob
import re
from pathlib import Path

sourceFolder = '00_rawData/Re100alpha10/'
destFolder = '01_data/'

Path("01_data").mkdir(exist_ok=True)

"""# Join data"""

def natural_sort(file_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(file_list, key=alphanum_key)


sortedFilenames = natural_sort(glob.glob(sourceFolder + '*.mat'))

for item in sortedFilenames:
    print(item)

with h5py.File(sortedFilenames[0], 'r') as f:
    print("Keys: %s" % f.keys())
    print(f['Re'][0])
    print(f['alpha'][0])
    print(f['dt'][0])
    print(f['U'][0].dtype)
    print(f['U'][:].shape)

startFile = 1
endFile = 15

join = None

for filename in sortedFilenames:
    print(filename)
    with h5py.File(filename, 'r') as f:
        if join is None:
            join = np.stack((f['U'][:].astype(np.float32), f['V'][:].astype(np.float32)), axis=3)
        else:
            join = np.concatenate(
                (join, np.stack((f['U'][:].astype(np.float32), f['V'][:].astype(np.float32)), axis=3)))

join = np.moveaxis(join, 1, 2)
print(join.shape)
print(join.dtype)

"""# Prepare data"""


def mean_fcn(data):
    nt = data.shape[0]
    field_sum = np.zeros_like(data[0])
    for idx in tqdm(range(nt)):
        # Accumulate field values
        field_sum += data[idx]

    return field_sum / nt


def std_fcn(data, mean):
    nt = data.shape[0]
    sz = data.size / 2

    squareTerm_sum = 0

    for t in tqdm(range(nt)):
        # Accumulate field values
        squareTerm_sum += np.sum(np.square(data[t] - mean), axis=(0, 1), keepdims=True)

    std = np.sqrt(squareTerm_sum / sz)

    return std


def normalize_meanstd(data):
    mean = mean_fcn(data)
    std = std_fcn(data, mean)

    return mean, std


UV_mean, UV_std = normalize_meanstd(join)

print()
print(UV_mean.shape)
print(UV_std.shape)
print(UV_std)

nt = join.shape[0]
print(nt)

for i in tqdm(range(nt)):
    join[i] = (join[i] - UV_mean) / UV_std

print(join.shape, join.dtype)
print(UV_mean.shape, UV_mean.dtype)
print(UV_std.shape, UV_std.dtype)

UV_mean = np.expand_dims(UV_mean, axis=0)
UV_std = np.expand_dims(UV_std, axis=0)

print(UV_mean.shape, UV_mean.dtype)
print(UV_std.shape, UV_std.dtype)

print(np.std((join[:5000, :, :, 0]).flatten()))

print(np.isnan(np.sum(join)))
print(np.isnan(np.sum(UV_std)))
print(np.isnan(np.sum(UV_mean)))

name = sourceFolder.split('/')[-2]

dest_filename = destFolder + name + '_newData_' + str(nt) + '.hdf5'

print(dest_filename)

with h5py.File(dest_filename, 'w') as f:
    f.create_dataset('UV', data=join, dtype='float32')
    f.create_dataset('mean', data=UV_mean, dtype='float32')
    f.create_dataset('std', data=UV_std, dtype='float32')

dest_filename = destFolder + name + '_newData_' + str(nt) + '_slice.hdf5'

print(dest_filename)

with h5py.File(dest_filename, 'w') as f:
    f.create_dataset('UV', data=join[:1000], dtype='float32')
    f.create_dataset('mean', data=UV_mean, dtype='float32')
    f.create_dataset('std', data=UV_std, dtype='float32')

"""Figures"""

plt.hist((join[:50, :, :, 0]).flatten())
plt.title('u´ distribution')
plt.show()

plt.hist((join[:50, :, :, 1]).flatten())
plt.title('v´ distribution')
plt.show()

plt.imshow(join[5, :, :, 0].astype('float32'), cmap="RdBu_r")
plt.colorbar()
plt.title('u´ sample')
plt.show()

plt.imshow(join[5, :, :, 1].astype('float32'), cmap="RdBu_r")
plt.colorbar()
plt.title('v´ sample')
plt.show()

plt.imshow(UV_mean[0, :, :, 0].astype('float32'), cmap="RdBu_r")
plt.colorbar()
plt.title('u mean')
plt.show()

plt.imshow(UV_mean[0, :, :, 1].astype('float32'), cmap="RdBu_r")
plt.colorbar()
plt.title('v mean')
plt.show()
