from keras.utils import Sequence
import numpy as np
import os
import math
import pathlib
import random

class Loader(Sequence):
    def __init__(self, data_dir, las_strides=1, train=True):
        self.strides = las_strides * 5
        self.batch_size = 33 ** 2
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        self.num_files = len(self.fnames)
        self.train = train

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.fnames)

    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        chm_batch = np.zeros((self.batch_size, 40, 40, 1))
        rgb_batch = np.zeros((self.batch_size, 40, 40, 3))
        hsi_batch = np.zeros((self.batch_size, 40, 40, 3))
        las_batch = np.zeros((self.batch_size, 8, 8, 70, 1))

        chm = np.load(self.data_dir / "chm" / self.fnames[index]).repeat(10, axis=0).repeat(10, axis=1)
        rgb = np.load(self.data_dir / "rgb" / self.fnames[index])
        hsi = np.load(self.data_dir / "hsi" / self.fnames[index]).repeat(10, axis=0).repeat(10, axis=1)
        las = np.load(self.data_dir / "las" / self.fnames[index])

        if self.train:
            bounds_batch = np.zeros((self.batch_size, 9, 4))
            labels_batch = np.zeros((self.batch_size, 9), dtype=int)

            bounds = np.load(self.data_dir / "bounds" / self.fnames[index])
            labels = np.load(self.data_dir / "labels" / self.fnames[index])
        else:
            bounds = []
            labels = []

        img_x = 0
        img_y = 0
        las_x = 0
        las_y = 0
        for i in range(33):
            img_x = 0
            las_x = 0
            for j in range(33):
                min_batch_ind = i*33 + j

                window_top = 1 - (img_y / 200)
                window_left = img_x / 200
                window_right = window_left + 0.2
                window_bot = window_top - 0.2
                window_centroid = [window_left + 0.1, window_top - 0.1]

                chm_batch[min_batch_ind, :, :, :] = chm[img_y:img_y+40, img_x:img_x+40, :]
                rgb_batch[min_batch_ind, :, :, :] = rgb[img_y:img_y+40, img_x:img_x+40, :]
                hsi_batch[min_batch_ind, :, :, :] = hsi[img_y:img_y+40, img_x:img_x+40, :]
                las_batch[min_batch_ind, :, :, :, 0] = las[las_y:las_y+8, las_x:las_x+8, :]
                img_x += 5
                las_x += 1

                ind = 0
                for (bound, label) in zip(bounds, labels):
                    if not label:
                        continue

                    centroid = [(bound[0]+bound[2])/2, (bound[1]+bound[3])/2]
                    if (centroid[0] > window_left) and (centroid[0] < window_right) and (centroid[1] > window_bot) and (centroid[1] < window_top):
                        labels_batch[min_batch_ind, ind] = 1
                        bounds_batch[min_batch_ind, ind, 0] = bound[0] - window_centroid[0]
                        bounds_batch[min_batch_ind, ind, 1] = bound[1] - window_centroid[1]
                        bounds_batch[min_batch_ind, ind, 2] = bound[2] - window_centroid[0]
                        bounds_batch[min_batch_ind, ind, 3] = bound[3] - window_centroid[1]
                        ind += 1

            img_y += 5
            las_y += 1

        if self.train:
            return ([rgb_batch, chm_batch, hsi_batch, las_batch], [bounds_batch, labels_batch])
        else:
            return [rgb_batch, chm_batch, hsi_batch, las_batch]

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
