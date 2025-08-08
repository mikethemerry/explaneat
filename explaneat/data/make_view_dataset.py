# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import os
import sys

import json

import pandas as pd

import random

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to take synthetic_view.csv and preprocess
    it, ready for modelling, and write it to output_filepath directory as
    synthetic_view_train.csv and synthetic_view_test.csv, with additional
    synthetic_view_train_XXXXXXX.csv for specific-sized subsets of the data
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    seed = 42
    logger.info('set random seed to {}'.format(seed))
    random.seed(seed)

    synth = pd.read_csv(input_filepath)

    # Verify num rows
    assert synth.shape == (2451278, 32)

    logger.info('creating train-test split')
    synth_train, synth_test = train_test_split(synth, test_size = 451278, random_state = seed)

    logger.info('writing train and test to file')
    synth_train.to_csv(os.path.join(output_filepath, 'synthetic_view_train.csv'))
    synth_test.to_csv(os.path.join(output_filepath, 'synthetic_view_test.csv'))

    logger.info('creating smaller training subsets')
    datasetSizes = [
        1000,
        2500,
        5000,
        10000,
        25000,
        50000,
        100000,
        250000,
        500000,
        1000000,
        1500000,
        2000000
    ]
    for dsSize in datasetSizes:
        logger.info('creating dataset of size {}'.format(dsSize))
        subset = synth_train.sample(dsSize)
        subset.to_csv(os.path.join(output_filepath, 'synthetic_view_test_{:07d}.csv'.format(dsSize)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
