from src.data.data_formats import Image, PointCloud, Shapefile
import numpy as np
from keras.utils import Sequence
import pathlib
import os
import math
from sklearn.decomposition import PCA
import pickle

class Preprocessor:
    def __init__(self):
        self.polys = {"MLBS": None, "OSBS": None}
        self.pca_name = "pca.pickle"
        self.pca = None

    def fit_pca(self, base_direc, num_dims=3):
        full_hsi = np.zeros((78, 20, 20, 369))
        for (i, site)  in enumerate(["MLBS", "OSBS"]):
            for j in range(1, 40):
                full_hsi[i*39 + j-1, :, :, :] = Image(base_direc, "HSI", site, j).as_normalized_array()

        full_hsi = full_hsi.reshape((78 * 20 * 20, 369))
        pca = PCA(n_components=3)
        pca.fit(full_hsi)
        print(pca.explained_variance_ratio_)
        self.pca = pca
        with open(self.pca_name, "wb") as f:
            pickle.dump(pca, f)

    def load_plot(self, base_direc, site, plot_id, apply_pca=True):
        self.base_direc = base_direc
        chm = Image(base_direc, "CHM", site, plot_id)
        rgb = Image(base_direc, "RGB", site, plot_id)
        hsi = Image(base_direc, "HSI", site, plot_id)
        las = PointCloud(base_direc, site, plot_id)

        if self.polys[site] == None:
            self.polys[site] = Shapefile(base_direc, site)

        polys = self.polys[site]
        bounds = rgb.get_bounds()
        plot_polys = polys.filter(bounds.left, bounds.right, bounds.bottom, bounds.top)
        bounding_vec = np.zeros((30, 5))

        for (i, poly) in enumerate(plot_polys.iterrows()):
            poly_bounds = poly[1]["geometry"].bounds
            bounding_vec[i, 0] = (poly_bounds[0] - bounds.left) / 20
            bounding_vec[i, 1] = (poly_bounds[1] - bounds.bottom) / 20
            bounding_vec[i, 2] = (poly_bounds[2] - bounds.left) / 20
            bounding_vec[i, 3] = (poly_bounds[3] - bounds.bottom) / 20
            bounding_vec[i, 4] = 1

        return (chm.as_normalized_array(), rgb.as_normalized_array(), self.apply_pca(hsi.as_normalized_array(), apply_pca), las.to_voxels(bounds.left, bounds.top, 0.5), bounding_vec)

    def apply_pca(self, hsi, apply_pca):
        if not apply_pca:
            return hsi

        if self.pca == None:
            if not os.path.isfile(self.pca_name):
                self.fit_pca(self.base_direc)

            with open(self.pca_name, "rb") as f:
                self.pca = pickle.load(f)

        hsi_transformed = self.pca.transform(hsi.reshape((20*20, 369)))
        return hsi_transformed.reshape((20, 20, 3))


class Loader(Sequence):
    def __init__(self, batch_size, data_dir):
        self.batch_size = batch_size
        self.data_dir = pathlib.Path(data_dir)
        self.fnames = np.array(os.listdir(self.data_dir / "chm"))
        np.random.shuffle(self.fnames)
        self.chm = np.zeros((self.batch_size, 20, 20, 1))
        self.rgb = np.zeros((self.batch_size, 200, 200, 3))
        self.hsi = np.zeros((self.batch_size, 20, 20, 3))
        self.las = np.zeros((self.batch_size, 40, 40, 70, 1))
        self.bounds = np.zeros((self.batch_size, 30, 4))
        self.labels = np.zeros((self.batch_size, 30), dtype=int)

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)

    def get_batch_fnames(self, index):
        start = (index * self.batch_size) % 78
        indices = np.array(range(start, start + 10))
        indices[indices >= 78] = indices[indices >= 78] - 78
        return self.fnames[indices]

    def __len__(self):
        return math.ceil(78 / self.batch_size)
    
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
