from jsonschema import validate
from explaneat.experimenter.schemas.experiment import experiment as EXPERIMENT_SCHEMA
from explaneat.experimenter.results import Result, ResultsDatabase

from pathlib import Path
import json
import logging
import os
import datetime

import psutil
import socket

# from git import Repo
# from git import InvalidGitRepositoryError

import shutil

from hashlib import sha256


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


class obj:

    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


class GenericExperiment(object):

    folders = [
        'results',
        'results/interim',
        'results/final',
        'configurations',
        'logs'
    ]

    file_loggers = {
        "error.log": logging.ERROR,
        "info.log": logging.INFO
    }

    def __init__(self, config, confirm_path_creation=True, experiment_sha=None, logging_level=logging.INFO, ref_file=None):
        self.create_stream_logging(logging_level)

        self.config_files = {}

        if ref_file is not None:
            existing_refs = self.get_existing_run_from_file(ref_file)
            self.experiment_sha = existing_refs['sha']
            self.experiment_start_time = existing_refs['start_time']
            self.experiment_start_time_with_ms = existing_refs['start_time_ms']
        else:
            self.experiment_sha = experiment_sha
            self.experiment_start_time = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
            self.experiment_start_time_with_ms = datetime.datetime.now().strftime("%y%m%dT%H%M%S.%f")

        self._requested_config = config

        self.load_config()

        # self.add_device_and_repo_to_config()

        self.config_string = json.dumps(self.config)

        self.confirm_path_creation = confirm_path_creation

        self.generate_folder_structure()
        self.create_file_logging()

        self.save_configurations()

        self.logger.info(
            "Experiment init for project %s has completed" % self.experiment_sha)

    def load_config(self):

        if Path(self._requested_config).is_file():
            with open(self._requested_config, 'r') as fp:
                self.config = json.load(fp)
        else:
            raise FileNotFoundError("Config file not found")

    def run_experiment(self):
        pass

    def _pre_run_experiment(self):
        pass

    def _post_run_experiment(self):
        pass

    def generate_folder_structure(self):
        # This will create the standard folder structure for experiments
        # including results, experiment information, etc.
        self.logger.info("Starting to create folder structures")

        self.experiment_folder_name = "%s_%s_%s" % (
            self._config['experiment']['codename'],
            self.experiment_start_time,
            self.experiment_sha
        )
        self.logger.info("Experiment folder name is %s" %
                         self.experiment_folder_name)

        self.root_path = os.path.join(
            self.config['experiment']['base_location'],
            self.experiment_folder_name
        )
        self.logger.info("Experiment root path is %s" % self.root_path)

        if not os.path.exists(self.config['experiment']['base_location']):
            if self.confirm_path_creation:
                if not (input("The base location does not exist, continue? Y/N\nBase location is %s" % self.config['experiment']['base_location']).lower() == "y"):
                    raise FileNotFoundError(
                        "The base location does not exist!")
        # baselocation/[experiment_name]_[time]
        if not os.path.exists(self.root_path):
            self.logger.info("Creating the root path")
            os.makedirs(self.root_path)
        self.logger.info('Root path created')

        for folder in self.folders:
            folder_name = self.path(folder)
            self.logger.info('Creating %s' % folder)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        self.logger.info('All folders created')

    def path(self, *args):
        return os.path.join(self.root_path, *args)

    def prepend_sha(self, myString):
        return "%s-%s" % (self.experiment_sha, myString)

    def save_experimenter_config(self):
        self.logger.info('Saving experiment configuration')
        with open(self.path('configurations', self.prepend_sha('experiment.json')), 'w') as fp:
            json.dump(self.config, fp, indent=4, sort_keys=True)

    def save_configurations(self):
        self.save_experimenter_config()
        self.logger.info("Saving other config files")
        for config_name, config_fp in self.config_files.items():
            src = config_fp
            dst = self.path('configurations', self.prepend_sha(
                '{}.configuration'.format(config_name)))
            shutil.copyfile(src, dst)

    def dict2obj(self, dict1):
        # https://www.geeksforgeeks.org/convert-nested-python-dictionary-to-object/
        # using json.loads method and passing json.dumps
        # method and custom object hook as arguments
        return json.loads(json.dumps(dict1), object_hook=obj)

    def validate_configuration(self, configuration):
        # check json schema
        self.logger.info("Validating configuration schema")
        validate(configuration, EXPERIMENT_SCHEMA)
        self.logger.info("Schema validation passed")

        # Check base location exists
        if not Path(configuration['experiment']['base_location']).is_dir():
            self.logger.error("`%s` does not exist as base location" % (
                configuration['experiment']['base_location']
            )
            )

        # Check data file locations
        for location_name, location in configuration['data']['locations'].items():
            self.logger.info("Checking `%s` for existence" % (location_name))
            for data_set in ['xs', 'ys']:
                if not Path(location[data_set]).is_file():
                    self.logger.error("`{}` is not a file for `{}` - `{}`".format(
                        location[data_set], location, data_set
                    ))

    def create_stream_logging(self, logging_level=logging.INFO):

        self.logger = logging.getLogger("experimenter")
        self.logger.propagate = False
        self.logger.setLevel(logging_level)

        ch = logging.StreamHandler()
        # Do I need this or does it inheret from logger?
        # ch.setLevel(logging_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)

    def create_logging_header(self, text, width=25):

        self.logger.info('#'*width)
        self.logger.info('-'*width)
        self.logger.info(text.center(width))
        self.logger.info('-'*width)
        self.logger.info('#'*width)

    def create_file_logging(self):
        self.logger.info("Creating file logging")
        for file, level in self.file_loggers.items():
            file_handler = logging.FileHandler(
                self.path("logs", self.prepend_sha(file)))

            # Do I need this or does it inheret from logger?
            file_handler.setLevel(level)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # add formatter to ch
            file_handler.setFormatter(formatter)
            # add ch to logger
            self.logger.addHandler(file_handler)

        self.logger.info("Finished creating file logging")

    def register_config_file(self, file_path, name, save=True):
        self.config_files[name] = file_path
        if save:
            self.save_configurations()

    def add_device_and_repo_to_config(self):

        curr_path = os.getcwd()
        have_good_repo = False
        # for path_depth in range(len(curr_path.split(os.sep))):
        # try:
        # repo = Repo(curr_path)
        # except InvalidGitRepositoryError:
        # curr_path = curr_path + "/.."
        # continue
        # have_good_repo = True
        # break

        # self.config['repository'] = {
        # "branch": repo.head.reference.name,
        # "commit": repo.head.reference.commit.hexsha,
        # "changes": [
        # diff.a_path for diff in repo.head.commit.diff(None)
        # ]
        # }

        svmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.config['device'] = {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max,
                "min_frequency": psutil.cpu_freq().min,
            },
            "memory": {
                "total": get_size(svmem.total),
                "swap": get_size(swap.total)
            },
            "hostname": socket.gethostname()
        }

    def write_run_to_file(self, file):
        doc_to_write = {
            "sha": self.experiment_sha,
            "start_time": self.experiment_start_time,
            "start_time_ms": self.experiment_start_time_with_ms
        }
        with open(file, "w") as fp:
            json.dump(doc_to_write, fp)

    def get_existing_run_from_file(self, file):

        with open(file, "r") as fp:
            doc = json.load(fp)
        return doc

    @ property
    def experiment_sha(self):
        """Experiment SHA is used as random seed for all experiments - it is
        defined against the configuration + run time, and the SHA is both a
        canonical reference to this specific iteration of the run, but also can
        be used to validate results later
        """

        if self._experiment_sha is not None:
            return self._experiment_sha

        return sha256(("%s-%s" % (self.config_string, self.experiment_start_time_with_ms)).encode()).hexdigest()[:8]

    @ experiment_sha.setter
    def experiment_sha(self, sha):
        if sha is not None:
            self.logger.warning(
                "Setting experiment sha to %s - this will replicate results of previous experiments. Only do this if you know what you're doing" % sha)
        self._experiment_sha = sha

    @ experiment_sha.getter
    def experiment_sha(self):
        if self._experiment_sha is not None:
            return self._experiment_sha
        return sha256(("%s-%s" % (self.config_string, self.experiment_start_time_with_ms)).encode()).hexdigest()[:8]

    @ property
    def config(self):
        return self._config

    @ config.setter
    def config(self, configuration):
        self.validate_configuration(configuration)
        self._config = configuration

    @ config.getter
    def config(self):
        return self._config

    @ property
    def data_folder(self):
        my_path = os.path.join(self._config['data']['processed_location'],
                               self._config['experiment']['codename'],
                               self.experiment_sha)
        return my_path

    @ property
    def results_database(self):
        if not hasattr(self, "_results_database"):
            self._results_database = ResultsDatabase(
                self.config['results']['database'])
        return self._results_database

    @ property
    def random_seed(self):
        return self.config['random_seed']
