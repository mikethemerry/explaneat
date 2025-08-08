import unittest
from explaneat.experimenter import experiment
import os
import shutil
import logging


import json

test_path_location = '/Users/mike/dev-mtm/phd-neat-experiments/data/experiments/tests'
remove_paths = True

class TestExperimentMethods(unittest.TestCase):
    config_file = './test_config.json'
    def setUp(self):
        self.experiment = experiment.GenericExperiment(self.config_file, confirm_path_creation=False)
    
    def test_all_folders_exist(self):
        self.assertTrue(os.path.exists(test_path_location))
        for folder in self.experiment.folders:
            self.assertTrue(os.path.exists(
                os.path.join(self.experiment.root_path, folder)))

    def test_experiment_config_exists(self):
        with open(self.config_file, 'r') as fp:
            my_config = json.load(fp)
        with open(os.path.join(self.experiment.path('configurations', self.experiment.prepend_sha('experiment.json')))) as fp:
            saved_config = json.load(fp)
        self.assertEqual(my_config, saved_config)

    def tearDown(self):
        # Clear logging
        logger = logging.getLogger("experimenter")
        logger.handlers.clear() 

        if remove_paths:
            shutil.rmtree(test_path_location)

class TestExperimentSHAs(unittest.TestCase):
    config_file = './test_config.json'

    def test_sha_creation_changes(self):
        self.experiment1 = experiment.GenericExperiment(self.config_file, confirm_path_creation=False)
        self.experiment2 = experiment.GenericExperiment(self.config_file, confirm_path_creation=False)

        self.assertNotEqual(
            self.experiment1.experiment_sha,
            self.experiment2.experiment_sha
        )

    def test_sha_creation_stability(self):
        self.experiment1 = experiment.GenericExperiment(
            self.config_file, 
            confirm_path_creation=False,
            experiment_sha="asdf")
        self.experiment2 = experiment.GenericExperiment(
            self.config_file, 
            confirm_path_creation=False,
            experiment_sha="asdf")

        self.assertEqual(
            self.experiment1.experiment_sha,
            self.experiment2.experiment_sha
        )

    def test_sha_prepend(self):
        test_sha = "asdf"
        test_string = "foo"
        self.experiment = experiment.GenericExperiment(
            self.config_file, 
            confirm_path_creation=False,
            experiment_sha=test_sha)
        good_string = "%s-%s"%(test_sha, test_string)
        self.assertEqual(
            good_string, 
            self.experiment.prepend_sha(test_string)
        )

    def tearDown(self):
        # Clear logging
        logger = logging.getLogger("experimenter")
        logger.handlers.clear() 
        if remove_paths:
            shutil.rmtree(test_path_location)


if __name__ == '__main__':
    unittest.main()
