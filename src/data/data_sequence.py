from keras.utils import Sequence
import numpy as np
import os
import math
import pathlib
import random

class Loader(Sequence):
    def __init__(self, batch_size=25, data_dir, train=True):
        self.batch_size = batch_size
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        self.num_files = len(self.fnames)
        self.chm = np.zeros((self.batch_size, 40, 40, 1))
        self.rgb = np.zeros((self.batch_size, 40, 40, 3))
        self.hsi = np.zeros((self.batch_size, 40, 40, 3))
        self.las = np.zeros((self.batch_size, 8, 8, 70, 1))
        self.train = train
        if train:
            self.bounds = np.zeros((self.batch_size, 9, 4))
            self.labels = np.zeros((self.batch_size, 9), dtype=int)

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.fnames)

    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        batch_files = self.get_batch_fnames(index)
        for i in range(self.batch_size):
            self.chm[i, :, :, :] = np.load(self.data_dir / "chm" / batch_files[i])
            self.rgb[i, :, :, :] = np.load(self.data_dir / "rgb" / batch_files[i])
            self.hsi[i, :, :, :] = np.load(self.data_dir / "hsi" / batch_files[i])
            self.las[i, :, :, :, 0] = np.load(self.data_dir / "las" / batch_files[i])
            self.las[i, :, :, :4, 0] = 0

            if self.train:
                self.bounds[i, :, :] = np.load(self.data_dir / "bounds" / batch_files[i])
                self.labels[i, :] = np.load(self.data_dir / "labels" / batch_files[i])
                self.rotate_arrays(i)

        if self.train:
            return ([self.rgb, self.chm, self.hsi, self.las], [self.bounds, self.labels])
        else:
            return [self.rgb, self.chm, self.hsi, self.las]

    def rotate_arrays(self, ind):
        rotations = random.randint(0,3)
        self.chm[ind, :, :, :] = np.rot90(self.chm[ind, :, :, :], rotations)
        self.rgb[ind, :, :, :] = np.rot90(self.rgb[ind, :, :, :], rotations)
        self.las[ind, :, :, :, 0] = np.rot90(self.las[ind, :, :, :, 0], rotations)

        for i in range(4):
            new_ind = (i - rotations) % 4
            self.bounds[ind, :, i] = self.bounds[ind, :, new_ind]
            if rotations == 1 or rotations == 2:
                self.bounds[ind, :, :] = 1 - self.bounds[ind, :, :]
