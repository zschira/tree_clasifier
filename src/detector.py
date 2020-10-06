from src.preprocessor import PcaHandler
from src.data.data_formats import Image, PointCloud, Shapefile
from src.data.data_sequence import Loader
from src.model import LeafNet
import pathlib
import os
import numpy as np
import re

class Detector:
    def __init__(self):
        self.data_direc = None
        self.leaf_net = LeafNet()
        self.fname_parser = re.compile("([A-Z]+)_(\d+).\w+")
        self.pca = PcaHandler()

    def preprocess(self, base_direc, save_direc=None, sites=["MLBS", "OSBS"]):
        base_direc = pathlib.Path(base_direc).absolute()
        if save_direc == None:
            save_direc = base_direc / "processed"

        save_direc = pathlib.Path(save_direc).absolute()
        self.data_direc = save_direc

        self.create_paths(save_direc)

        # Fit PCA
        self.pca.fit(base_direc)

        fnames = os.listdir(base_direc / "RemoteSensing/CHM")

        polys = {}
        if os.path.isdir(base_direc / 'ITC'):
            labels = True
            for site in sites:
                polys[site] = Shapefile(base_direc, site)
        else:
            labels = False

        for fname in fnames:
            match = self.fname_parser.match(fname)
            site = match.group(1)
            plot = match.group(2)
            chm = Image(base_direc, "CHM", site, plot)
            rgb = Image(base_direc, "RGB", site, plot)
            hsi = Image(base_direc, "HSI", site, plot)
            las = PointCloud(base_direc, site, plot)
            bounds = chm.get_bounds()

            np.save(save_direc / "chm" / "{}_{}".format(site, plot), chm.as_normalized_array())
            np.save(save_direc / "rgb" / "{}_{}".format(site, plot), rgb.as_normalized_array())
            np.save(save_direc / "hsi" / "{}_{}".format(site, plot), self.pca.apply_pca(hsi.as_normalized_array()))
            np.save(save_direc / "las" / "{}_{}".format(site, plot), las.to_voxels(bounds.left, bounds.top, 0.5))

            if labels:
                y = polys[site].get_train(bounds)
                np.save(save_direc / "bounds" / "{}_{}".format(site, plot), y[:, :4])
                np.save(save_direc / "labels" / "{}_{}".format(site, plot), y[:, 4].astype(int))

    def fit_model(self, data_direc=None, weights_path=None):
        if data_direc == None:
            data_direc = self.data_direc
        else:
            data_direc = pathlib.Path(data_direc)

        data_loader = Loader(10, data_direc, 78)

        # Load weights if they exist
        if weights_path != None:
            weights_path = pathlib.Path(weights_path).absolute()
            self.leaf_net.load_weights(weights_path)
        else:
            weights_path = data_direc

        self.leaf_net.compile()
        self.leaf_net.fit(data_loader, weights_path)

    def predict(self, test_dir, weights_path):
        data_loader = Loader(10, test_dir, 153)
        self.leaf_net.load_weights(weights_path)
        self.leaf_net.predict(data_loader)

    def create_paths(self, save_direc):
        if not os.path.exists(save_direc):
            os.mkdir(save_direc)

        # Create data paths
        for feature in ["chm", "rgb", "hsi", "las", "bounds", "labels"]:
            path = save_direc / feature
            if not os.path.exists(path):
                os.mkdir(path)
