#!/usr/bin/env python3
"""
ExplaNEAT Experiment Management Example

This example demonstrates how to use the GenericExperiment framework
for comprehensive experiment tracking and management.
"""

import os
import json
import tempfile
import shutil
from datetime import datetime

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.experimenter.results import Result
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def create_experiment_config():
    """Create an example experiment configuration"""
    config = {
        "experiment": {
            "name": "explaneat_example_experiment",
            "description": "Example experiment demonstrating ExplaNEAT experiment management"
        },
        "random_seed": 42,
        "model": {
            "propneat": {
                "population_size": 50,
                "n_iterations": 2,
                "max_n_generations": 10,
                "epochs_per_generation": 5
            },
            "propneat_retrain": {
                "n_iterations": 2,
                "n_epochs": 20
            }
        },
        "data": {
            "format": "synthetic",
            "n_samples": 500,
            "n_features": 8
        }
    }
    
    # Create temporary config file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name

def create_reference_config():
    """Create a reference configuration file"""
    ref_config = {
        "hardware": {
            "gpu_available": True,
            "cpu_cores": 4
        },
        "versions": {
            "python": "3.8+",
            "pytorch": "1.9.0+",
            "neat": "0.92+"
        }
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(ref_config, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name

def main():
    """Main example function"""
    print("ExplaNEAT Experiment Management Example")
    print("=" * 45)
    
    # Create temporary directory for experiment
    temp_dir = tempfile.mkdtemp(prefix="explaneat_experiment_")
    print(f"Experiment directory: {temp_dir}")
    
    try:
        # Create configuration files
        config_file = create_experiment_config()
        ref_file = create_reference_config()
        
        # Initialize experiment
        print("\n1. Initializing experiment...")
        experiment = GenericExperiment(
            config_file, 
            confirm_path_creation=False, 
            ref_file=ref_file
        )
        
        # Change to experiment directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        # Create experiment header
        experiment.create_logging_header("Starting Example Experiment", 50)
        
        # Access experiment properties
        print(f"Experiment name: {experiment.config['experiment']['name']}")
        print(f"Experiment SHA: {experiment.experiment_sha}")
        print(f"Random seed: {experiment.config['random_seed']}")
        
        # Generate some example data
        print("\n2. Generating example data...")
        np.random.seed(experiment.config['random_seed'])
        
        X, y = make_classification(
            n_samples=experiment.config['data']['n_samples'],
            n_features=experiment.config['data']['n_features'],
            n_informative=6,
            n_redundant=2,
            random_state=experiment.config['random_seed']
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=experiment.config['random_seed']
        )
        
        experiment.logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Simulate experiment iterations
        print("\n3. Running experiment iterations...")
        
        for iteration in range(experiment.config['model']['propneat']['n_iterations']):
            print(f"\nIteration {iteration + 1}...")
            
            # Simulate training time
            import time
            start_time = datetime.now()
            time.sleep(1)  # Simulate computation
            end_time = datetime.now()
            
            # Generate some mock results
            mock_accuracy = 0.7 + np.random.random() * 0.2  # Random accuracy between 0.7-0.9
            mock_loss = 0.1 + np.random.random() * 0.3      # Random loss between 0.1-0.4
            mock_predictions = np.random.random(len(y_test)).tolist()
            
            # Store training time result
            train_time_result = Result(
                data=(end_time - start_time).seconds,
                result_type="training_time",
                experiment_name=experiment.config['experiment']['name'],
                dataset_name="synthetic_dataset",
                experiment_sha=experiment.experiment_sha,
                iteration=iteration,
                meta={"units": "seconds"}
            )
            experiment.results_database.add_result(train_time_result)
            
            # Store accuracy result
            accuracy_result = Result(
                data=mock_accuracy,
                result_type="test_accuracy",
                experiment_name=experiment.config['experiment']['name'],
                dataset_name="synthetic_dataset",
                experiment_sha=experiment.experiment_sha,
                iteration=iteration,
                meta={"metric": "binary_accuracy"}
            )
            experiment.results_database.add_result(accuracy_result)
            
            # Store loss result
            loss_result = Result(
                data=mock_loss,
                result_type="test_loss",
                experiment_name=experiment.config['experiment']['name'],
                dataset_name="synthetic_dataset",
                experiment_sha=experiment.experiment_sha,
                iteration=iteration,
                meta={"loss_function": "binary_cross_entropy"}
            )
            experiment.results_database.add_result(loss_result)
            
            # Store predictions
            predictions_result = Result(
                data=json.dumps(mock_predictions),
                result_type="predictions",
                experiment_name=experiment.config['experiment']['name'],
                dataset_name="synthetic_dataset",
                experiment_sha=experiment.experiment_sha,
                iteration=iteration,
                meta={"prediction_type": "probabilities"}
            )
            experiment.results_database.add_result(predictions_result)
            
            experiment.logger.info(f"Iteration {iteration}: Accuracy={mock_accuracy:.4f}, Loss={mock_loss:.4f}")
        
        # Save results
        print("\n4. Saving experiment results...")
        experiment.results_database.save()
        
        # Demonstrate results retrieval
        print("\n5. Demonstrating results retrieval...")
        
        # Get all results for this experiment
        all_results = experiment.results_database.results
        print(f"Total results stored: {len(all_results)}")
        
        # Filter results by type
        accuracy_results = [r for r in all_results if r.result_type == "test_accuracy"]
        print(f"Accuracy results: {len(accuracy_results)}")
        
        if accuracy_results:
            accuracies = [r.data for r in accuracy_results]
            print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        
        # Create final experiment header
        experiment.create_logging_header("Experiment Completed Successfully", 50)
        
        print("\n6. Experiment artifacts created:")
        print(f"   - Results database: {experiment.results_database.file_path}")
        print(f"   - Log files in: logs/")
        print(f"   - Configuration files in: configurations/")
        
        print("\n" + "=" * 45)
        print("Experiment management example completed!")
        print(f"All artifacts saved in: {temp_dir}")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        
        # Clean up temporary files (but keep experiment directory for inspection)
        os.unlink(config_file)
        os.unlink(ref_file)
        
        print(f"\nNote: Experiment directory preserved at: {temp_dir}")
        print("You can inspect the generated files there.")

if __name__ == "__main__":
    main()