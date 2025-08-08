#!/usr/bin/env python3
"""
Basic ExplaNEAT Usage Example

This example demonstrates the basic usage of ExplaNEAT for evolving and analyzing
neural networks on a simple binary classification task.
"""

import numpy as np
import torch
import neat
import tempfile
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import binary_cross_entropy
from explaneat.visualization import visualize

def create_neat_config():
    """Create a basic NEAT configuration file"""
    config_content = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.9
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
# Network parameters
num_inputs            = 10
num_outputs           = 1
num_hidden            = 0
initial_connection    = full_direct
connection_add_prob   = 0.5
connection_delete_prob = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2

# Connection parameters
enabled_default       = True
enabled_mutate_rate   = 0.01
weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_max_value      = 30
weight_min_value      = -30
weight_mutate_power   = 0.5
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    # Create temporary config file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    temp_file.write(config_content)
    temp_file.close()
    
    return temp_file.name

def main():
    """Main example function"""
    print("ExplaNEAT Basic Usage Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate synthetic binary classification data
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape y for binary classification
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create NEAT configuration
    print("\n2. Setting up NEAT configuration...")
    config_file = create_neat_config()
    
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )
        
        # Create population
        print("\n3. Creating population...")
        population = BackpropPopulation(config, X_train, y_train)
        
        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Evolve the network
        print("\n4. Starting evolution...")
        winner = population.run(
            binary_cross_entropy, 
            n_generations=10,  # Small number for demo
            nEpochs=5  # Few epochs per generation for speed
        )
        
        print(f"\nEvolution completed! Best fitness: {winner.fitness}")
        
        # Create ExplaNEAT analyzer
        print("\n5. Analyzing the evolved network...")
        explainer = ExplaNEAT(winner, config)
        
        # Print network statistics
        print(f"Network depth: {explainer.depth()}")
        print(f"Number of parameters: {explainer.n_genome_params()}")
        print(f"Parameter density: {explainer.density():.4f}")
        print(f"Network skippiness: {explainer.skippines():.4f}")
        
        # Make predictions
        print("\n6. Making predictions on test set...")
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions = explainer.net.forward(X_test_tensor)
            predictions_np = predictions.cpu().numpy()
        
        # Calculate accuracy
        binary_predictions = (predictions_np > 0.5).astype(int)
        accuracy = np.mean(binary_predictions == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Demonstrate retraining
        print("\n7. Demonstrating network retraining...")
        explainer.net.reinitialise_network_weights()
        explainer.net.retrain(
            X_train, 
            y_train,
            n_epochs=20,
            choose_best=True,
            validate_split=0.2,
            random_seed=42
        )
        
        # Make predictions with retrained network
        with torch.no_grad():
            retrained_predictions = explainer.net.forward(X_test_tensor)
            retrained_predictions_np = retrained_predictions.cpu().numpy()
        
        retrained_binary_predictions = (retrained_predictions_np > 0.5).astype(int)
        retrained_accuracy = np.mean(retrained_binary_predictions == y_test)
        print(f"Retrained network accuracy: {retrained_accuracy:.4f}")
        
        # Generate network visualization
        print("\n8. Generating network visualization...")
        try:
            graph = visualize.draw_net(config, winner, view=False)
            print("Network visualization created (GraphViz source):")
            print("(Note: Install GraphViz to render the visualization)")
            print(f"Nodes: {len(winner.nodes)}, Connections: {len(winner.connections)}")
        except Exception as e:
            print(f"Visualization failed (this is normal if GraphViz is not installed): {e}")
        
        print("\n" + "=" * 40)
        print("Example completed successfully!")
        
    finally:
        # Clean up temporary config file
        os.unlink(config_file)

if __name__ == "__main__":
    main()