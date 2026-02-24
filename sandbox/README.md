# Sandbox

This folder is a **development playground** for testing and prototyping before committing to the actual function logic.

## Purpose

Use this sandbox to:

- **Test API connections** - Verify authentication, endpoints, and response formats
- **Prototype implementations** - Experiment with different approaches before finalizing
- **Debug integrations** - Isolate and troubleshoot third-party service interactions
- **Validate data transformations** - Test parsing, mapping, and formatting logic

## Usage

1. Create test scripts here (e.g., `test_api.py`, `prototype.py`)
2. Run and iterate until the implementation is solid
3. Move the validated logic to `src/backend/function_logic.py`
4. Delete or keep sandbox files as reference (they won't be deployed)

## Important Notes

- **Tracked for reference** - Sandbox files are committed to serve as future reference
- **Not deployed** - Sandbox files are excluded from the Lambda package
- **No credentials** - Never hardcode secrets; use the chask-sdk to retrieve them

## Getting Secrets with chask-sdk

Use the `chask-sdk` to securely retrieve secrets from your organization:

```bash
pip install chask-sdk
```

```python
from chask_sdk import ChaskClient

# Initialize with token (or set CHASK_TOKEN env var)
client = ChaskClient(token="sdk_...")

# List available secrets in your organization
secrets = client.list_secrets()
for s in secrets:
    print(f"{s['key']}: {s['uuid']}")

# Get a specific secret by UUID
secret = client.get_secret("550e8400-e29b-41d4-a716-446655440000")
api_key = secret.reveal()  # Access the actual value

# Bulk retrieve multiple secrets (max 10)
secrets = client.get_secrets_bulk(["uuid1", "uuid2"])
for uuid, secret in secrets.items():
    print(f"{secret.name}: {secret.reveal()}")
```

The `SecretValue` class protects against accidental exposure - values are masked by default and require explicit `.reveal()` to access.

## Example Workflow

```bash
# 1. Create a test script
touch sandbox/test_connection.py

# 2. Test your API integration
python sandbox/test_connection.py

# 3. Once working, move logic to function_logic.py
# 4. Clean up sandbox if desired
```
