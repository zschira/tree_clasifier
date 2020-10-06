from keras.utils import Sequence
import numpy as np
import os
import math
import pathlib

class Loader(Sequence):
    def __init__(self, batch_size, data_dir, num_files):
        self.num_files = num_files
        self.batch_size = batch_size
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        self.chm = np.zeros((self.batch_size, 20, 20, 1))
        self.rgb = np.zeros((self.batch_size, 200, 200, 3))
        self.hsi = np.zeros((self.batch_size, 20, 20, 3))
        self.las = np.zeros((self.batch_size, 40, 40, 70, 1))
        self.bounds = np.zeros((self.batch_size, 30, 4))
        self.labels = np.zeros((self.batch_size, 30), dtype=int)

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)

    def get_batch_fnames(self, index):
        start = (index * self.batch_size) % self.num_files
        indices = np.array(range(start, start + 10))
        indices[indices >= self.num_files] = indices[indices >= self.num_files] - self.num_files
        return self.fnames[indices]

    def __len__(self):
        return math.ceil(self.num_files / self.batch_size)
    
    def __getitem__(self, index):
        batch_files = self.get_batch_fnames(index)
        for i in range(self.batch_size):
            self.chm[i, :, :, :] = np.load(self.data_dir / "chm" / batch_files[i])
            self.rgb[i, :, :, :] = np.load(self.data_dir / "rgb" / batch_files[i])
            self.hsi[i, :, :, :] = np.load(self.data_dir / "hsi" / batch_files[i])
            self.las[i, :, :, :, 0] = np.load(self.data_dir / "las" / batch_files[i])
            self.bounds[i, :, :] = np.load(self.data_dir / "bounds" / batch_files[i])
            self.labels[i, :] = np.load(self.data_dir / "labels" / batch_files[i])

        return ([self.rgb, self.chm, self.hsi, self.las], [self.bounds, self.labels])

class LoaderTest(Sequence):
    def __init__(self, batch_size, data_dir, num_files):
        self.num_files = num_files
        self.batch_size = batch_size
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        self.chm = np.zeros((self.batch_size, 20, 20, 1))
        self.rgb = np.zeros((self.batch_size, 200, 200, 3))
        self.hsi = np.zeros((self.batch_size, 20, 20, 3))
        self.las = np.zeros((self.batch_size, 40, 40, 70, 1))

    def get_batch_fnames(self, index):
        start = (index * self.batch_size) % self.num_files
        indices = np.array(range(start, start + 10))
        indices[indices >= self.num_files] = indices[indices >= self.num_files] - self.num_files
        return self.fnames[indices]

    def __len__(self):
        return math.ceil(self.num_files / self.batch_size)
    
    def __getitem__(self, index):
        batch_files = self.get_batch_fnames(index)
        for i in range(self.batch_size):
            self.chm[i, :, :, :] = np.load(self.data_dir / "chm" / batch_files[i])
            self.rgb[i, :, :, :] = np.load(self.data_dir / "rgb" / batch_files[i])
            self.hsi[i, :, :, :] = np.load(self.data_dir / "hsi" / batch_files[i])
            self.las[i, :, :, :, 0] = np.load(self.data_dir / "las" / batch_files[i])
            self.bounds[i, :, :] = np.load(self.data_dir / "bounds" / batch_files[i])
            self.labels[i, :] = np.load(self.data_dir / "labels" / batch_files[i])

        return [self.rgb, self.chm, self.hsi, self.las]
