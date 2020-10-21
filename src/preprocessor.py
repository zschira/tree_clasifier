import numpy as np
from sklearn.decomposition import PCA
import pickle
from src.data.data_formats import Image
import os
import numpy as np

class PcaHandler:
    def __init__(self, name="pca.pickle"):
        self.pca = None
        self.pca_name = name

    def fit(self, base_direc):
        if os.path.isfile(self.pca_name):
            return

        self.pca = PCA(n_components=3)

        full_hsi = np.zeros((78, 20, 20, 369))
        for (i, site)  in enumerate(["MLBS", "OSBS"]):
            for j in range(1, 40):
                full_hsi[i*39 + j-1, :, :, :] = Image(base_direc, "HSI", site, j).as_array()

        full_hsi = full_hsi.reshape((78 * 20 * 20, 369))
        self.pca.fit(full_hsi)
        print(self.pca.explained_variance_ratio_)

        with open(self.pca_name, "wb") as f:
            pickle.dump(self.pca, f)

    def apply_pca(self, hsi):
        if self.pca == None:
            if not os.path.isfile(self.pca_name):
                raise Exception("PCA file does not exist")

            with open(self.pca_name, "rb") as f:
                self.pca = pickle.load(f)

        hsi_transformed = self.pca.transform(hsi.reshape((20*20, 369)))
        hsi_normalized = (hsi_transformed - np.mean(hsi_transformed)) / np.std(hsi_transformed)
        return hsi_normalized.reshape((20, 20, 3))
