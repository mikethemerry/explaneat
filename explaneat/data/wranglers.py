import os
import json
import numpy as np
import pandas as pd

import csv

import torch

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split

import explaneat.data.utils as utils

from torch.utils.data import Dataset, DataLoader

import pmlb

import logging


TYPE_DICT = {
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
    "DOUBLE": np.double,
    "INT32": np.int32
}

TRANSFORMERS = {
    "STANDARD_SCALAR": utils.linear_scale,
    "ONE_HOT_ENCODE": utils.one_hot_encode
}


class TabularDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return (x, y)


class WRANGLER(object):
    def __init__(self, folder):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    @property
    def Train_Dataset(self):
        return TabularDataset(self.X_train.to_numpy(), self.y_train.to_numpy())

    @property
    def Test_Dataset(self):
        return TabularDataset(self.X_test.to_numpy(), self.y_test.to_numpy())

    @property
    def X_train(self):
        if self._X_train is None:
            raise AttributeError(
                "Train test split must be created before getting X train")
        return self._X_train

    @property
    def X_test(self):
        if self._X_test is None:
            raise AttributeError(
                "Train test split must be created before getting X test")
        return self._X_test

    @property
    def y_train(self):
        if self._y_train is None:
            raise AttributeError(
                "Train test split must be created before getting y train")
        return self._y_train

    @property
    def y_test(self):
        if self._y_test is None:
            raise AttributeError(
                "Train test split must be created before getting y test")
        return self._y_test

    @property
    def data_lengths(self):
        """xtrain, xtest, ytrain, ytest lengths
        """
        return (
            len(self.X_train),
            len(self.X_test),
            len(self.y_train),
            len(self.y_test),
        )

    @property
    def data_shapes(self):
        """xtrain, xtest, ytrain, ytest lenghts
        """
        return (
            self.X_train.shape,
            self.X_test.shape,
            self.y_train.shape,
            self.y_test.shape,
        )

    @property
    def train_sets(self):
        return (
            self.X_train,
            self.y_train
        )

    @property
    def test_sets(self):
        return (
            self.X_test,
            self.y_test
        )

    @property
    def train_sets_as_np(self):
        return (
            self.X_train.to_numpy(),
            self.y_train.to_numpy()
        )

    @property
    def test_sets_as_np(self):
        return (
            self.X_test.to_numpy(),
            self.y_test.to_numpy()
        )

    @property
    def train_sets_as_tensor(self):
        return (
            self.X_train.to_numpy(),
            self.y_train.to_numpy()
        )

    @property
    def test_sets_as_tensor(self):
        return (
            self.X_test.to_numpy(),
            self.y_test.to_numpy()
        )

    @property
    def input_size(self):
        return self.X_train.shape[1]

    @property
    def output_size(self):
        return self.y_train.shape[1]


class GENERIC_WRANGLER(WRANGLER):
    def __init__(self, folder):
        """Looks at folder, grabs x& y train and test

        Args:
            folder (str): Folder that contains x/y train and test`
        """
        self.folder = folder

        self.load_data()

    def load_data(self):
        def load_from_file(file, folder):
            path = os.path.join(folder, file)
            return pd.read_csv(path)

        self._X_train = load_from_file("x_train.csv", self.folder)
        self._X_test = load_from_file("x_test.csv", self.folder)
        self._y_train = load_from_file("y_train.csv", self.folder)
        self._y_test = load_from_file("y_test.csv", self.folder)


class UCI_WRANGLER(WRANGLER):

    def __init__(self,
                 data_file,
                 meta_file,
                 config={},
                 logger=logging.getLogger("experimenter.uci_wrangler")):

        self.logger = logger

        if not os.path.exists(data_file):
            raise FileNotFoundError
        else:
            self.data_file = data_file
        if not os.path.exists(meta_file):
            raise FileNotFoundError
        else:
            self.meta_file = meta_file

        self.logger.info("Loading meta file")
        self.load_meta_file()
        self.logger.info("Loading raw data file")
        self.load_raw_data()

        self.logger.info("Preprocessing data")
        self.preprocess_x()
        self.preprocess_y()
        self.logger.info("Finished preprocessing data")

    def load_meta_file(self):
        if self.meta_file is None:
            raise AttributeError("Must have meta_file set")
        with open(self.meta_file, 'r') as fp:
            self.meta = json.load(fp)

    def load_raw_data(self):
        if self.data_file is None:
            raise AttributeError("Must have data_file set")
        self.data = pd.read_csv(self.data_file, header=None)
        self.data.columns = self.meta['columns']
        for category in self.meta['definitions']:
            self.data[category] = self.data[category].apply(
                lambda x: self.meta['definitions'][category].index(x))

        # Load raw data
        self.xs_raw = np.array(self.data[self.meta['x_columns']]).astype(
            TYPE_DICT[self.meta['x_type']])
        self.ys_raw = np.array(self.data[self.meta['y_column']]).astype(
            TYPE_DICT[self.meta['y_type']])

    def preprocess_x(self):
        self.xs = self.xs_raw.copy()
        for transform in self.meta['x_transforms']:
            self.xs = TRANSFORMERS[transform](self.xs)

    def preprocess_y(self):

        self.ys = self.ys_raw.copy()
        for transform in self.meta['y_transforms']:
            self.ys = TRANSFORMERS[transform](self.ys)

        self.logger.info("ys shape is {}".format(self.ys.shape))
        if(len(self.ys.shape)) == 1:
            self.logger.info("recasting ys to (n,1)")
            self.ys = self.ys[:, None]

    def create_train_test_split(self,
                                test_size,
                                random_state):
        self.logger.info("Creating train test split")
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self.xs, self.ys, test_size=test_size, random_state=random_state)
        self.logger.info("split created")

    def send_train_test_to_device(self, device):
        self.logger.info("sending train test to device {}".format(device))
        self._X_train = torch.from_numpy(self._X_train).to(device)
        self._X_test = torch.from_numpy(self._X_test).to(device)
        self._y_train = torch.from_numpy(self._y_train).to(device)
        self._y_test = torch.from_numpy(self._y_test).to(device)
        self.logger.info("train test are on device {}".format(device))

    def write_train_test_to_csv(self, folder):
        def write_to_file(data, file, folder, x_header=False, y_header=False):
            if x_header and y_header:
                raise Exception("Cannot have both X and Y headers")
            path = os.path.join(folder, file)
            self.logger.info("Writing to {}".format(path))
            with open(path, 'w') as fp:
                writer = csv.writer(fp)
                if x_header:
                    self.logger.info("Adding x header")
                    writer.writerow(self.meta['x_columns'])
                if y_header:
                    self.logger.info("Adding y header")
                    if len(data[0]) > 1:
                        raise NotImplementedError(
                            "Y Headers only coded for single category output")
                    writer.writerow(["y"])
                writer.writerows(data)
            self.logger.info("Completed to {}".format(path))

        self.logger.info(
            "sending train test to csv in folder {}".format(folder))
        write_to_file(self._X_train, "x_train.csv", folder, x_header=True)
        write_to_file(self._X_test, "x_test.csv", folder, x_header=True)
        write_to_file(self._y_train, "y_train.csv", folder, y_header=True)
        write_to_file(self._y_test, "y_test.csv", folder, y_header=True)
        self.logger.info("train test are in folder {}".format(folder))


class PMLB_WRANGLER(WRANGLER):

    def __init__(self,
                 dataset_name,
                 config={},
                 logger=logging.getLogger("experimenter.uci_wrangler")):

        self.logger = logger

        if dataset_name not in pmlb.dataset_names:
            raise AttributeError("No PMLB dataset with that name")
        self.dataset_name = dataset_name

        # self.logger.info("Loading meta file")
        # self.load_meta_file()
        self.logger.info("Loading raw data file")
        self.load_raw_data()

        self.logger.info("Preprocessing data")
        self.preprocess_x()
        self.preprocess_y()
        self.logger.info("Finished preprocessing data")

    # def load_meta_file(self):
    #     if self.meta_file is None:
    #         raise AttributeError("Must have meta_file set")
    #     with open(self.meta_file, 'r') as fp:
    #         self.meta = json.load(fp)

    def load_raw_data(self):
        if self.dataset_name is None:
            raise AttributeError("Must have a dataset nameset")

        xs, ys = pmlb.fetch_data(self.dataset_name, return_X_y=True)
        self.full_data = pmlb.fetch_data(self.dataset_name)

        self.xs_raw = xs
        self.ys_raw = ys

        # Get the list of column names (excluding the 'target' column)
        column_names = list(self.full_data.columns)
        column_names.remove('target')
        self.x_columns = column_names
        self.y_columns = "target"

    def preprocess_x(self):
        self.xs = self.xs_raw.copy()
        for transform in ["STANDARD_SCALAR"]:
            self.xs = TRANSFORMERS[transform](self.xs)

    def preprocess_y(self):

        self.ys = self.ys_raw.copy()
        for transform in []:
            self.ys = TRANSFORMERS[transform](self.ys)

        self.logger.info("ys shape is {}".format(self.ys.shape))
        if(len(self.ys.shape)) == 1:
            self.logger.info("recasting ys to (n,1)")
            self.ys = self.ys[:, None]

    def create_train_test_split(self,
                                test_size,
                                random_state):
        self.logger.info("Creating train test split")
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self.xs, self.ys, test_size=test_size, random_state=random_state)
        self.logger.info("split created")

    def send_train_test_to_device(self, device):
        self.logger.info("sending train test to device {}".format(device))
        self._X_train = torch.from_numpy(self._X_train).to(device)
        self._X_test = torch.from_numpy(self._X_test).to(device)
        self._y_train = torch.from_numpy(self._y_train).to(device)
        self._y_test = torch.from_numpy(self._y_test).to(device)
        self.logger.info("train test are on device {}".format(device))

    def write_train_test_to_csv(self, folder):
        def write_to_file(data, file, folder, x_header=False, y_header=False):
            if x_header and y_header:
                raise Exception("Cannot have both X and Y headers")
            path = os.path.join(folder, file)
            self.logger.info("Writing to {}".format(path))
            with open(path, 'w') as fp:
                writer = csv.writer(fp)
                if x_header:
                    self.logger.info("Adding x header")
                    writer.writerow(self.x_columns)
                if y_header:
                    self.logger.info("Adding y header")
                    if len(data[0]) > 1:
                        raise NotImplementedError(
                            "Y Headers only coded for single category output")
                    writer.writerow(["y"])
                writer.writerows(data)
            self.logger.info("Completed to {}".format(path))

        self.logger.info(
            "sending train test to csv in folder {}".format(folder))
        write_to_file(self._X_train, "x_train.csv", folder, x_header=True)
        write_to_file(self._X_test, "x_test.csv", folder, x_header=True)
        write_to_file(self._y_train, "y_train.csv", folder, y_header=True)
        write_to_file(self._y_test, "y_test.csv", folder, y_header=True)
        self.logger.info("train test are in folder {}".format(folder))
