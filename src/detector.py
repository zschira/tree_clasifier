from preprocessor import Preprocessor, Loader
from model import LeafNet
import pathlib
import os
import numpy as np

class Detector:
    def __init__(self, base_direc=None):
        self.base_direc = pathlib.Path(base_direc)
        self.preprocessor = Preprocessor(base_direc)
        self.leaf_net = LeafNet()

    def preprocess(self, save_direc=None):
        if save_direc == None:
            save_direc = self.base_direc

        save_direc = pathlib.Path(save_direc) / "processed"
        self.data_direc = save_direc

        self.create_paths(save_direc)

        for site in ["MLBS", "OSBS"]:
            for plot in range(1,40):
                (chm, rgb, hsi, las, y) = self.preprocessor.load_plot(site, plot)
                np.save(save_direc / "chm" / "{}_{}".format(site, plot), chm)
                np.save(save_direc / "rgb" / "{}_{}".format(site, plot), rgb)
                np.save(save_direc / "hsi" / "{}_{}".format(site, plot), hsi)
                np.save(save_direc / "las" / "{}_{}".format(site, plot), las)
                np.save(save_direc / "bounds" / "{}_{}".format(site, plot), y[:, :4])
                np.save(save_direc / "labels" / "{}_{}".format(site, plot), y[:, 4].astype(int))

    def fit_model(self, data_direc=None, weights_path=None):
        if data_direc == None:
            data_direc = self.data_direc
        else:
            data_direc = pathlib.Path(data_direc)

        data_loader = Loader(10, data_direc)

        # Load weights if they exist
        if weights_path != None:
            weights_path = pathlib.Path(weights_path).absolute()
            self.leaf_net.load_weights(weights_path)
        else:
            weights_path = data_direc

        self.leaf_net.compile()
        self.leaf_net.fit(data_loader, weights_path)

    def create_paths(self, save_direc):
        if not os.path.exists(save_direc):
            os.mkdir(save_direc)

        # Create data paths
        for feature in ["chm", "rgb", "hsi", "las", "bounds", "labels"]:
            path = save_direc / feature
            if not os.path.exists(path):
                os.mkdir(path)
