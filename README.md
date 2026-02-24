# ReadProjectFileFn

Lambda function created with Chask CLI

## Overview

This is a Chask organization-specific Lambda function deployed from GitHub.

## Configuration

- **Runtime**: python3.12
- **Handler**: handler.lambda_handler
- **Timeout**: 300s
- **Memory**: 256MB
- **Layers**: {{ cookiecutter.layers }}

## Parameters

### Required

- **action** (string): Action to perform

### Optional

- **verbose** (boolean): Enable verbose logging (default: false)

## Project Structure

```
ReadProjectFileFn/
├── manifest.yml              # Function configuration
├── src/
│   ├── handler.py            # ⚠️ Infrastructure code (DO NOT MODIFY)
│   └── backend/
│       ├── __init__.py
│       └── function_logic.py # ✏️ Your business logic (MODIFY THIS)
├── tests/
│   └── test_basic.json       # Test file template
├── README.md                 # This file
├── INSTRUCTIONS.md           # AI agent documentation
├── .pre-commit-config.yaml   # Pre-commit hook configuration
├── setup.sh                  # One-time setup script
└── .gitignore
```

## Architecture

### Handler Pattern

This function uses a **resilient handler pattern** that separates infrastructure code from business logic:

**handler.py** (Infrastructure - DO NOT MODIFY):
- Parses Lambda events
- Handles errors and edge cases
- **Guarantees agent liberation** via finally block
- Sends error responses automatically

**backend/function_logic.py** (Your Code - MODIFY THIS):
- Contains `FunctionBackend` class
- Implement `process_request()` method
- Extract parameters and call APIs
- Return results as strings

### Example Implementation

```python
# src/backend/function_logic.py

class FunctionBackend:
    def process_request(self) -> str:
        # 1. Extract parameters
        tool_args = self._extract_tool_args()
        action = tool_args.get("action")

        # 2. Your business logic
        result = my_custom_function(action)

        # 3. Send response
        self._send_response(result, is_error=False)

        return result
```

**Benefits:**
- ✅ Clean separation of concerns
- ✅ Agent never gets stuck (guaranteed liberation)
- ✅ Automatic error handling
- ✅ Simple developer interface

### Event Evolution

Lambda functions use the **Event Evolution** pattern to maintain proper parent-child relationships in the Event Tracking System:

```python
# Create a child event linked to the parent via evolved_from_uuid
evolve_response = orchestrator_api_manager.call(
    "evolve_event",
    parent_event_uuid=str(orchestration_event.event_id),
    event_type="function_call_response",
    source="agent",
    target="orchestrator",
    prompt=message,
    extra_params=extra_params,
    access_token=orchestration_event.access_token,
    organization_id=orchestration_event.organization.organization_id,
)

# Reconstruct local event object for Kafka forwarding
evolved_uuid = evolve_response.get("uuid")
response_event = orchestration_event.model_copy(deep=True)
response_event.event_id = evolved_uuid
```

**Why Event Evolution?**
- ✅ Creates proper event genealogy trees for debugging
- ✅ Enables event replay and virtual branching
- ✅ Maintains `evolved_from_uuid` linkage in database
- ✅ Better visibility into event chains

**Note:** The `_send_response()` helper in `function_logic.py` already implements this pattern correctly. You typically don't need to modify it.

## Local Development

### Prerequisites

- Python python3.12 or higher
- AWS CLI configured
- SAM CLI (for local testing)

### Quick Setup

Run the setup script after cloning:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Initialize git (if needed)
- Install pre-commit hooks
- Install project dependencies (if requirements.txt exists)

**Pre-commit hooks validate:**
- Python syntax errors
- Linting issues (via ruff)
- manifest.yml syntax
- Test JSON file validity
- Accidental private key commits

### Testing Locally

```bash
# Install dependencies (if any)
pip install -r requirements.txt

# Test with SAM CLI
sam local invoke -e event.json
```

Example `event.json`:
```json
{
  "organization_id": "test-org-uuid",
  "params": {
    "action": "analyze",
    "verbose": true
  },
  "access_token": "test-token"
}
```

## Deployment

### Via Chask CLI

```bash
# First time setup
chask setup

# Deploy function
chask function publish
```

This will:
1. Validate manifest.yml
2. Build Lambda function with SAM CLI
3. Deploy to AWS with CloudFormation
4. Register function in Chask database
5. Assign to your organization

### Manual Deployment

For advanced use cases, you can deploy manually using SAM CLI:

```bash
sam build
sam deploy --guided
```

## Usage

Once deployed, this function can be invoked by the Chask orchestrator when:
- An agent needs to call an organization-specific function
- The function name matches "ReadProjectFileFn"
- The request comes from your organization

## Built-in Layers

This function includes Chask's built-in layers:

- **foundationMinimal**: Core utilities, database access, S3 helpers
- **foundationApi**: Chask API client, request helpers

## Environment Variables

### Automatic Environment Variables

The following environment variables are automatically injected by Chask during deployment:

- `BASE_DOMAIN`: API domain for orchestrator communication (auto-detected based on environment)
- `MODE`: Environment mode (LOCAL, DEVELOPMENT, PRODUCTION)
- `AWS_REGION`: AWS region (default: us-east-1)

### Custom Environment Variables

You can set custom environment variables in `manifest.yml`:

```yaml
function:
  name: MyFunctionFn

  # Environment variables
  environment:
    CUSTOM_VAR: some_value
    API_ENDPOINT: https://api.example.com
```

## Security

- Function is scoped to your organization via IAM policies
- Has access to organization secrets in AWS Secrets Manager:
  `arn:aws:secretsmanager:*:*:secret:chask/org/<your-org-id>/*`
- Uses Chask API tokens for authentication

## Monitoring

View Lambda execution logs in:
- AWS CloudWatch Logs: `/aws/lambda/ReadProjectFileFn`
- Chask dashboard: Organization → Functions → ReadProjectFileFn

## Troubleshooting

### Function not found

Ensure the function is deployed and assigned to your organization:
```bash
chask function publish
```

### Timeout errors

Increase timeout in `manifest.yml`:
```yaml
function:
  timeout: 300  # Up to 900 seconds (15 minutes)
```

### Memory errors

Increase memory in `manifest.yml`:
```yaml
function:
  memory_size: 1024  # Up to 10240 MB
```

## Development

### Adding Dependencies

1. Create `requirements.txt` in project root
2. List your dependencies:
   ```
   requests>=2.31.0
   pandas>=2.0.0
   ```
3. Rebuild and deploy

### Adding Custom Layers

To include custom Lambda layers:

1. Create layer directory:
   ```bash
   mkdir -p layers/custom_layer/python
   pip install -t layers/custom_layer/python <package>
   ```

2. Update `manifest.yml`:
   ```yaml
   function:
     layers:
       - foundationMinimal
       - foundationApi
       - custom/custom_layer  # Your custom layer
   ```

## Lambda Layers Best Practices

### Avoid Heavy Dependencies

⚠️ **Avoid heavy scientific packages** (pandas, numpy, scipy) as they cause build issues in Lambda layers.

**Use lightweight alternatives instead**:
- ✅ `openpyxl` for Excel files (instead of pandas)
- ✅ `requests` for HTTP requests
- ✅ `python-dateutil` for date handling (instead of pandas)

### Creating Layers Correctly

```bash
# 1. Create layer structure
mkdir -p layers/your_layer_name/python

# 2. Install with correct platform targeting
pip install --platform manylinux2014_x86_64 \
  --target=layers/your_layer_name/python \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  package1 package2 package3

# 3. Clean up unnecessary files
find layers/ -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find layers/ -name "*.pyc" -delete
find layers/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find layers/ -name "meson.build" -delete
find layers/ -name "setup.py" -delete
find layers/ -name "pyproject.toml" -delete

# 4. Update manifest.yml
# Add: - custom/your_layer_name
```

### Layer Size Guidelines

- ✅ Keep layers < 10 MB for fast cold starts
- ⚠️ Layers 10-50 MB are acceptable but slower
- ❌ Avoid layers > 50 MB (causes deployment issues)

### Verify Layer Before Publishing

```bash
# Check layer size
du -sh layers/your_layer_name/

# Check dependencies are included
ls layers/your_layer_name/python/

# Test imports locally (Python 3.12)
python3.12 -c "import sys; sys.path.insert(0, 'layers/your_layer_name/python'); import your_package"
```

## Support

For issues or questions:
1. Check Chask documentation: https://docs.chask.io
2. Contact Chask support
3. File an issue in this repository

## License

Proprietary - Organization-specific Chask function
