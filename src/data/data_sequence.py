from keras.utils import Sequence
import numpy as np
import os
import math
import pathlib
import random
import time

class Loader(Sequence):
    def __init__(self, data_dir, train=True, batch_size=1):
        self.batch_size = batch_size
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        self.num_files = len(self.fnames)
        self.train = train

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.fnames)

    def __len__(self):
        return self.num_files

    def get_batch_fnames(self, index):
        start = (index * self.batch_size) % self.num_files
        indices = np.array(range(start, start + self.batch_size))
        indices[indices >= self.num_files] = indices[indices >= self.num_files] - self.num_files
        return self.fnames[indices]
    
    def __getitem__(self, index):
        chm = np.zeros((self.batch_size, 200, 200, 1))
        rgb = np.zeros((self.batch_size, 200, 200, 3))
        hsi = np.zeros((self.batch_size, 200, 200, 3))
        las = np.zeros((self.batch_size, 40, 40, 70, 1))

        batch_files = self.get_batch_fnames(index)

        ratio = 24 / 168

        if self.train:
            bounds_batch = np.zeros((self.batch_size, 625, 9, 4))
            labels_batch = np.zeros((self.batch_size, 625, 9), dtype=int)

        for batch_ind in range(self.batch_size):
            chm[batch_ind, :, :, :] = np.load(self.data_dir / "chm" / batch_files[batch_ind]).repeat(10, axis=0).repeat(10, axis=1)
            rgb[batch_ind, :, :, :] = np.load(self.data_dir / "rgb" / batch_files[batch_ind])
            hsi[batch_ind, :, :, :] = np.load(self.data_dir / "hsi" / batch_files[batch_ind]).repeat(10, axis=0).repeat(10, axis=1)
            las[batch_ind, :, :, :, 0] = np.load(self.data_dir / "las" / batch_files[batch_ind])

            if self.train:
                bounds = np.load(self.data_dir / "bounds" / batch_files[batch_ind])
                labels = np.load(self.data_dir / "labels" / batch_files[batch_ind])
            else:
                bounds = []
                labels = []

            img_x = 0
            img_y = 0
            las_x = 0
            las_y = 0
            for i in range(25):
                img_x = 0
                for j in range(25):
                    min_batch_ind = i*25 + j

                    window_top = 1 - (img_y / 168)
                    window_left = img_x / 168
                    window_right = window_left + ratio
                    window_bot = window_top - ratio
                    window_centroid = [window_left + ratio/2, window_top - ratio/2]

                    img_x += 6

                    ind = 0
                    for (bound, label) in zip(bounds, labels):
                        if not label:
                            continue

                        centroid = [(bound[0]+bound[2])/2, (bound[1]+bound[3])/2]
                        if (centroid[0] > window_left) and (centroid[0] < window_right) and (centroid[1] > window_bot) and (centroid[1] < window_top):
                            labels_batch[batch_ind, min_batch_ind, ind] = 1
                            bounds_batch[batch_ind, min_batch_ind, ind, 0] = bound[0] - window_centroid[0]
                            bounds_batch[batch_ind, min_batch_ind, ind, 1] = bound[1] - window_centroid[1]
                            bounds_batch[batch_ind, min_batch_ind, ind, 2] = bound[2] - window_centroid[0]
                            bounds_batch[batch_ind, min_batch_ind, ind, 3] = bound[3] - window_centroid[1]
                            ind += 1

                img_y += 6

        if self.train:
            return ([rgb, chm, hsi, las], [np.reshape(bounds_batch, (self.batch_size, 625 * 9, 4)), np.reshape(labels_batch, (self.batch_size, 625 * 9))])
        else:
            return [rgb, chm, hsi, las]

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
