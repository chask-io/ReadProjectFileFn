# Function Tests

This directory contains test files for the `ReadProjectFileFn` Lambda function.

## Test File Structure

Test files are JSON files with the following structure:

```json
{
  "function_name": "ReadProjectFileFn",
  "args": {
    "param1": "value1",
    "param2": "value2"
  },
  "widget_params": {
    "api_key": "your_test_api_key"
  },
  "prompt": "Optional custom prompt for the orchestration event",
  "event_type": "function_call",
  "extra_params": {
    "openai_api_key": "your_openai_key",
    "model": "gpt-4o"
  },
  "files": [],
  "metadata": {
    "description": "Description of this test case",
    "expected_status": "success"
  }
}
```

### Fields

- **function_name**: The name of the function being tested (must match manifest.yml)
- **args**: Function arguments passed to the Lambda
- **widget_params**: Widget/secret parameters (API keys, tokens, etc.)
- **prompt**: Optional custom prompt for the orchestration event (for lambdas that depend on the prompt parameter)
- **event_type**: The orchestration event type (default: `function_call`). Some lambdas expect specific event types (e.g., `pipeline_designer_request`)
- **extra_params**: Additional parameters passed to the Lambda (e.g., `openai_api_key`, `model`, custom configs)
- **files**: Array of file paths from `test_files/` directory to include
- **metadata**: Test metadata
  - **description**: Human-readable description of the test case
  - **expected_status**: Expected outcome (`success` or `failure`)

## Running Tests

```bash
# Generate a test file from manifest.yml
chask function test:init

# Run a specific test
chask function test tests/test_basic.json

# Run with verbose output
chask function test tests/test_basic.json --verbose
```

## Test Files Directory

Place any input files needed for testing in the `test_files/` directory.
Reference them in your test JSON using relative paths:

```json
{
  "files": ["test_files/sample_input.csv"]
}
```
