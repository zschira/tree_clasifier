from src.preprocessor import PcaHandler
from src.data.data_formats import Image, PointCloud, Shapefile
from src.data.data_sequence import Loader
from src.utils import compute_iou, bb_2_polygons
from src.model import LeafNet
import pandas as pd
import geopandas as gpd
import pathlib
import os
import numpy as np
import re
import matplotlib.pyplot as plt

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

        self.create_paths(save_direc, labels)

        df = pd.DataFrame(columns=['site', 'left', 'bottom'])
        for fname in fnames:
            match = self.fname_parser.match(fname)
            site = match.group(1)
            plot = match.group(2)
            chm = Image(base_direc, "CHM", site, plot)
            rgb = Image(base_direc, "RGB", site, plot)
            hsi = Image(base_direc, "HSI", site, plot)
            las = PointCloud(base_direc, site, plot)
            bounds = chm.get_bounds()
            df = df.append({'site': '{}_{}'.format(site, plot), 'left': bounds.left, 'bottom': bounds.bottom}, ignore_index=True)

            np.save(save_direc / "chm" / "{}_{}".format(site, plot), chm.as_normalized_array())
            np.save(save_direc / "rgb" / "{}_{}".format(site, plot), rgb.as_normalized_array())
            np.save(save_direc / "hsi" / "{}_{}".format(site, plot), self.pca.apply_pca(hsi.as_array()))
            np.save(save_direc / "las" / "{}_{}".format(site, plot), las.to_voxels(bounds.left, bounds.top, 0.5))

            if labels:
                y = polys[site].get_train(bounds)
                np.save(save_direc / "bounds" / "{}_{}".format(site, plot), y[:, :4])
                np.save(save_direc / "labels" / "{}_{}".format(site, plot), y[:, 4].astype(int))

        df.to_csv(save_direc / 'bounds.csv')

    def fit_model(self, data_direc=None, weights_path=None):
        if data_direc == None:
            data_direc = self.data_direc
        else:
            data_direc = pathlib.Path(data_direc)

        data_loader = Loader(17, data_direc)

        # Load weights if they exist
        if weights_path != None:
            weights_path = pathlib.Path(weights_path).absolute()
            self.leaf_net.load_weights(weights_path)
        else:
            weights_path = data_direc

        self.leaf_net.compile()
        self.leaf_net.fit(data_loader, weights_path)

    def predict(self, test_dir, weights_path):
        test_dir = pathlib.Path(test_dir).absolute()
        data_loader = Loader(1, test_dir, False)
        self.leaf_net.load_weights(pathlib.Path(weights_path).absolute())

        polygons = []
        df = pd.read_csv(test_dir / 'bounds.csv').set_index('site', drop=True)

        for i in range(len(data_loader)):
            fname = data_loader.fnames[i].split('.')[0]
            print(fname)
            predictions = self.leaf_net.predict(data_loader.__getitem__(i))
            (bounds, labels) = self.convert_predictions(predictions)
            self.score_predictions(bounds, labels, test_dir, fname)
            left = df.loc[fname, 'left']
            bottom = df.loc[fname, 'bottom']
            polygons += bb_2_polygons(left, bottom, bounds[labels == 1, :])

        polys = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:32617')
        polys.to_file('delin_subm.shp')

    def convert_predictions(self, predictions):
        bounds_pred = predictions[0][0, :, :]
        labels_pred = predictions[1][0, :]

        labels_pred[labels_pred > 0.75] = 1
        labels_pred[labels_pred < 0.75] = 0
        labels_pred = labels_pred.astype(int)

        return (bounds_pred, labels_pred)

    def score_predictions(self, bounds_pred, labels_pred, test_dir, fname):
        if not os.path.isfile(test_dir / "labels" / (fname + ".npy")):
            return

        bounds_truth = np.load(test_dir / "bounds" / (fname + ".npy"))
        labels_truth = np.load(test_dir / "labels" / (fname + ".npy"))

        iou_avg = 0
        num_bb = 0

        print(labels_pred)

        for (b_t, l_t, b_p, l_p) in zip(bounds_truth, labels_truth, bounds_pred, labels_pred):
            if l_t != l_p:
                print("Mismatched label for file {}".format(fname))
                continue

            if l_p == 0:
                continue

            iou_avg += compute_iou(b_t, b_p)
            num_bb += 1

        print("Average iou for file, {}: {}".format(fname, iou_avg / num_bb))

    def create_paths(self, save_direc, labels):
        if not os.path.exists(save_direc):
            os.mkdir(save_direc)

        # Create data paths
        features = ["chm", "rgb", "hsi", "las"]
        if labels:
            features.append("bounds")
            features.append("labels")

        for feature in features:
            path = save_direc / feature
            if not os.path.exists(path):
                os.mkdir(path)
