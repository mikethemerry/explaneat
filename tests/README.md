# ExplaNEAT Test Suite

Comprehensive test suite for the ExplaNEAT project covering database operations, CLI functionality, and analysis modules.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── utils.py                    # Test utility functions
├── test_db/                    # Database tests
│   ├── test_base.py            # Database base operations
│   ├── test_models.py          # Model CRUD tests
│   ├── test_serialization.py   # Serialization tests
│   └── test_relationships.py   # Relationship tests
├── test_cli/                   # CLI tests
│   ├── test_genome_explorer_cli.py  # CLI method tests
│   └── test_cli_main.py        # Main entry point tests
└── test_analysis/              # Analysis module tests
    ├── test_annotation_manager.py
    ├── test_explanation_manager.py
    ├── test_node_splitting.py
    ├── test_genome_explorer.py
    ├── test_coverage.py
    └── test_subgraph_validator.py
```

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_db/test_models.py

# Run specific test class or method
uv run pytest tests/test_db/test_models.py::TestDatasetModel::test_create_dataset

# Run with coverage report
uv run pytest --cov=explaneat --cov-report=html

# Run tests in parallel (requires pytest-xdist)
uv run pytest -n auto
```

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.db` - Database tests
- `@pytest.mark.cli` - CLI tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (can be skipped with `-m "not slow"`)

Examples:

```bash
# Run only database tests
uv run pytest -m db

# Run only CLI tests
uv run pytest -m cli

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Infrastructure

### Database Fixtures

Tests use an in-memory SQLite database for fast, isolated testing:

- `test_db` - Creates fresh database for each test
- `db_session` - Provides database session scoped to each test
- `neat_config` - Provides NEAT configuration for genome creation

### Test Data Fixtures

Pre-configured fixtures for common test scenarios:

- `test_experiment` - Test experiment with minimal config
- `test_population` - Test population for an experiment
- `test_genome` - Simple test genome
- `test_dataset` - Test dataset
- `test_explanation` - Test explanation for a genome
- `test_annotation` - Test annotation
- `test_node_split` - Test node split

### Utility Functions

The `tests/utils.py` module provides helper functions for creating test data:

- `create_test_experiment()` - Create experiment with config
- `create_test_genome()` - Create genome with specified structure
- `create_test_population()` - Create population with genomes
- `create_test_annotation()` - Create annotation for testing
- `create_test_explanation()` - Create explanation for testing
- `create_test_node_split()` - Create node split for testing

## Writing New Tests

### Test Organization

1. **Unit Tests**: Test individual methods/functions in isolation
2. **Integration Tests**: Test interactions between components
3. **Database Tests**: Test database operations and relationships
4. **CLI Tests**: Test command-line interface functionality

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
@pytest.mark.unit
class TestMyFeature:
    """Test my feature functionality."""
    
    def test_basic_functionality(self, test_db, db_session):
        """Test basic functionality."""
        # Arrange
        data = create_test_data(db_session)
        
        # Act
        result = my_function(data)
        
        # Assert
        assert result is not None
        assert result.value == expected_value
```

## Test Coverage Goals

- **Database layer**: 100% of CRUD operations
- **CLI methods**: 100% of public methods
- **Manager classes**: 100% of public methods
- **Overall target**: >90% code coverage

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Speed**: Use in-memory database and mock slow operations
3. **Clarity**: Use descriptive test names and clear assertions
4. **Fixtures**: Reuse fixtures for common test data
5. **Mocking**: Mock external dependencies (file I/O, network, etc.)

## Troubleshooting

### Database Connection Issues

If tests fail with database connection errors:

1. Ensure PostgreSQL is running (for integration tests)
2. Check `DATABASE_URL` environment variable
3. Verify database fixtures are properly configured

### Import Errors

If tests fail with import errors:

1. Ensure you're running tests from project root
2. Check that `explaneat` package is installed: `uv pip install -e .`
3. Verify Python path includes project directory

### Fixture Errors

If fixtures fail:

1. Check that `conftest.py` is in the correct location
2. Verify fixture dependencies are met
3. Ensure database is properly initialized

## Continuous Integration

Tests are designed to run in CI environments:

- Use in-memory database (no external dependencies)
- Fast execution (< 1 minute for full suite)
- Clear error messages for debugging
- Coverage reporting for quality metrics

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if adding new test patterns
