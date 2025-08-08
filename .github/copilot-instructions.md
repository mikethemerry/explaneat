# ExplaNEAT Development Instructions

ExplaNEAT is a Python package that provides tools for creating explanations of neural networks trained using the PropNEAT algorithm. It combines NEAT (NeuroEvolution of Augmenting Topologies) with backpropagation and includes experiment management, visualization, and analysis tools.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup and Dependencies
- Set Python path: `export PYTHONPATH=/home/runner/work/explaneat/explaneat:$PYTHONPATH`
- Install core dependencies: `pip3 install pandas numpy torch psutil jsonschema scikit-learn matplotlib seaborn`
  - **TIMING**: Dependency installation takes 2-5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Install optional dependencies (may timeout due to network issues): `pip3 install neat-python graphviz opencv-python`
  - If installation fails with timeout, document as "Optional dependencies may fail due to network limitations"
  - The core functionality works without these optional dependencies

### Running and Testing
- **CRITICAL**: Import time is ~2 seconds, experiment creation is ~0.3 seconds
- Test basic imports: `python3 -c "import sys; sys.path.append('.'); from explaneat.experimenter import experiment; from explaneat.core import explaneat; print('Imports successful')"`
- Run basic functionality test: 
  ```bash
  cd /home/runner/work/explaneat/explaneat
  PYTHONPATH=/home/runner/work/explaneat/explaneat python3 -c "
  from explaneat.experimenter.experiment import GenericExperiment
  import json, tempfile
  config = {
    'experiment': {'name': 'Test', 'description': 'Test', 'codename': 'test', 'base_location': '/tmp/test'},
    'results': {'location': 'data'}, 
    'data': {'locations': {'train': {'xs': 'x', 'ys': 'y'}, 'test': {'xs': 'x', 'ys': 'y'}}},
    'model': {'config_file': 'config'}
  }
  with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(config, f); config_path = f.name
  exp = GenericExperiment(config_path, confirm_path_creation=False)
  print('✓ Basic functionality works')
  "
  ```
- Run unit tests: `cd explaneat/experimenter/tests && PYTHONPATH=/home/runner/work/explaneat/explaneat python3 -m unittest experiment_tests -v`
  - **TIMING**: Tests complete in ~0.5 seconds. Most tests will fail due to hardcoded paths (/Users/mike/...), this is expected.
  - Alternative: Test individual components as shown in validation examples

### Build Process
- **No formal build process**: ExplaNEAT is a pure Python package with no compilation step
- **No package installation needed**: Run directly from source with proper PYTHONPATH
- **No build artifacts**: All functionality is available immediately after dependency installation

## Validation

### Comprehensive Validation Test
Run this complete validation to ensure all functionality works:

```bash
cd /home/runner/work/explaneat/explaneat
python3 -c "
print('=== COPILOT INSTRUCTIONS VALIDATION ===')
import time, sys, os
sys.path.append('.')

# Test imports (2s)
start_time = time.time()
from explaneat.experimenter import experiment
from explaneat.core import explaneat
print(f'✓ Imports work ({time.time() - start_time:.1f}s)')

# Test experiment creation (0.3s)
from explaneat.experimenter.experiment import GenericExperiment
import json, tempfile
config = {
  'experiment': {'name': 'Test', 'description': 'Test', 'codename': 'test', 'base_location': '/tmp/test'},
  'results': {'location': 'data'}, 
  'data': {'locations': {'train': {'xs': 'x', 'ys': 'y'}, 'test': {'xs': 'x', 'ys': 'y'}}},
  'model': {'config_file': 'config'}
}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
  json.dump(config, f); config_path = f.name
exp = GenericExperiment(config_path, confirm_path_creation=False)
print(f'✓ Experiment created: {exp.experiment_sha}')
print(f'✓ Folders: {len(exp.folders)} created')
print(f'✓ Path generation: {exp.path(\"results\", \"file.txt\").endswith(\"file.txt\")}')

# Cleanup
import shutil; shutil.rmtree('/tmp/test', ignore_errors=True); os.unlink(config_path)
print('✓ All functionality validated')
"
```

### Key Validation Points
- **ALWAYS** test core imports after making changes: `from explaneat.experimenter import experiment; from explaneat.core import explaneat`
- **ALWAYS** create a test experiment after making changes to experiment functionality
- **NEVER CANCEL** dependency installations even if they take 5+ minutes
- Test that you can create an experiment configuration and instantiate GenericExperiment
- Verify experiment folder structure creation works correctly

## Common Tasks

### Repository Structure
```
explaneat/
├── core/           # Core ExplaNEAT algorithms (backprop, neuralneat)  
├── experimenter/   # Experiment management and configuration
├── visualization/  # Network visualization tools
├── data/          # Data handling and wranglers
├── evaluators/    # Model evaluation tools
├── features/      # Feature extraction utilities
├── models/        # Model definitions and utilities
├── reporters/     # Results reporting and logging
└── __init__.py
```

### Key Components
- **experimenter.experiment.GenericExperiment**: Main class for managing experiments
  - Handles configuration validation, folder creation, logging
  - Creates structured experiment directories with SHA-based naming
  - Required config keys: experiment.{name,description,codename,base_location}, results.location, data.locations, model.config_file
- **core.explaneat.ExplaNEAT**: Main neural network explanation class
- **core.neuralneat.NeuralNeat**: Neural network implementation using NEAT topology

### Working with Experiments
- Create experiment config JSON with required fields (see schemas/experiment.py)
- Use `confirm_path_creation=False` for testing to avoid folder creation issues  
- Experiment folders follow pattern: `{codename}_{timestamp}_{sha}`
- Default folders created: results, results/interim, results/final, configurations, logs

### Dependencies Status
- **Core working**: pandas, numpy, torch, psutil, jsonschema, scikit-learn, matplotlib, seaborn
- **Optional (may fail)**: neat-python, graphviz, opencv-python
- **Package-lock.json in core/**: Only contains lockfileVersion, no actual dependencies

### Testing Strategy  
- Use PYTHONPATH=/home/runner/work/explaneat/explaneat for all Python commands
- Original unit tests may fail due to hardcoded paths (/Users/mike/...)
- Create new tests with /tmp/ paths for validation
- Focus on testing experiment creation, configuration validation, and basic imports

### Common Issues
- **Path Issues**: Always use absolute paths or proper PYTHONPATH
- **Network Timeouts**: pip install may timeout, continue without optional packages
- **Missing Files**: Experiments expect data files that may not exist, use confirm_path_creation=False
- **Cache Files**: .gitignore excludes __pycache__ directories

## Example Working Session
```bash
cd /home/runner/work/explaneat/explaneat
export PYTHONPATH=/home/runner/work/explaneat/explaneat:$PYTHONPATH

# Install dependencies (5+ minutes, NEVER CANCEL)
pip3 install pandas numpy torch psutil jsonschema scikit-learn matplotlib seaborn

# Test basic functionality (2 seconds)
python3 -c "
from explaneat.experimenter import experiment
from explaneat.core import explaneat
print('✓ Core imports work')
"

# Create and test an experiment (0.3 seconds) 
python3 -c "
from explaneat.experimenter.experiment import GenericExperiment
import json, tempfile, os
config = {'experiment': {'name': 'Test', 'description': 'Test', 'codename': 'test', 'base_location': '/tmp/test'}, 'results': {'location': 'data'}, 'data': {'locations': {'train': {'xs': 'x', 'ys': 'y'}, 'test': {'xs': 'x', 'ys': 'y'}}}, 'model': {'config_file': 'config'}}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f: json.dump(config, f); config_path = f.name
exp = GenericExperiment(config_path, confirm_path_creation=False)
print(f'✓ Experiment created: {exp.experiment_sha}')
print(f'✓ Folders: {exp.folders}')
import shutil; shutil.rmtree('/tmp/test', ignore_errors=True); os.unlink(config_path)
"
```