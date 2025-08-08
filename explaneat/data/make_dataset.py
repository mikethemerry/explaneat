    # -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import os
import sys

import json

import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get list of experiments (excluding hidden files like .gitignore)
    experiments = [d for d in os.listdir(input_filepath) if not d.startswith('.')]

    # Define file names
    configurationFileName = 'config.conf'
    generationRecordFileName = 'generationRecords.json'
    resultsFileName = 'results.csv'

    generationRecords = None
    results = None
    configurations = None


    for experiment in experiments:
        experimentDir = os.path.join(input_filepath, experiment)
        
        experimentRuns = os.listdir(experimentDir)

        for run in experimentRuns:
            
            logger.info('parsing {}'.format(run))

            if(run[0] == '.'):
                logger.info('not a valid path, skipping')
                continue
            runDir = os.path.join(experimentDir, run)

            runDetails = parse_experiment_filename(run)
            runDetails['dataset'] = experiment

            # Get generation records
            try:
                generationRecordsDF = parse_generation_records(
                    os.path.join(runDir, generationRecordFileName),
                    columnsToAdd = runDetails
                    )
            except FileNotFoundError:
                logger.warning('File not found, skipping %s' % runDir)
                continue
            if 'fitnesses' in generationRecordsDF.columns:
                generationRecordsDF['fitnessMax'] = generationRecordsDF['fitnesses'].apply(
                    lambda x: max(x)
                )
                generationRecordsDF['fitnessMin'] = generationRecordsDF['fitnesses'].apply(
                    lambda x: min(x)
                )

            if generationRecords is not None:
                generationRecords = generationRecords.append(generationRecordsDF)
            else:
                generationRecords = generationRecordsDF
            # Get ersults
            resultsDF = parse_results(
                os.path.join(runDir, resultsFileName),
                columnsToAdd = runDetails
            )
            if results is not None:
                results = results.append(resultsDF)
            else:
                results = resultsDF

    ## Output to ../processed
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    generationRecords.to_csv(os.path.join(output_filepath, 'generationRecords.csv'))
    results.to_csv(os.path.join(output_filepath, 'results.csv'))


def parse_generation_records(grFilePath, columnsToAdd = None):
    with open(grFilePath, 'r') as fp:
        gr = json.load(fp)
    df = pd.DataFrame.from_dict(gr).transpose()
    if columnsToAdd is not None:
        # assert columnsToAdd in(dict, object)
        for col, val in columnsToAdd.items():
            df[col] = val
    return df

def parse_results(resultsFilePath, columnsToAdd = None):
    df = pd.DataFrame.from_csv(resultsFilePath)
    if columnsToAdd is not None:
        # assert columnsToAdd in(dict, object)
        for col, val in columnsToAdd.items():
            df[col] = val
    return df


def parse_experiment_filename(filename):
    runDetails = filename.split('-')
    details = {
        "experimentType": runDetails[1],
        "experimentValue": runDetails[2],
        "experimentIteration": runDetails[3]
    }
    return details

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
