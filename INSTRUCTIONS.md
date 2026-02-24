# ReadProjectFileFn - AI Agent Instructions

This document provides comprehensive instructions for AI agents to understand and work with this Chask Lambda function.

## Table of Contents

**⚠️ [CRITICAL CHECKLIST - READ FIRST](#️-critical-checklist---read-before-implementing-️)**

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Sandbox: Development Playground](#sandbox-development-playground)
4. [Understanding OrchestrationEvent](#understanding-orchestrationevent)
5. [Implementing Your Business Logic](#implementing-your-business-logic)
6. [Understanding Parameters: Two Types](#understanding-parameters-two-types)
7. [Accessing Event Data](#accessing-event-data)
8. [Widget Parameters](#widget-parameters)
9. [Emitting Results](#emitting-results)
10. [CRITICAL: Preserving Test Execution Flags](#️-critical-preserving-test-execution-flags-️)
11. [Testing Your Lambda Function](#testing-your-lambda-function)
12. [Complete Examples](#complete-examples)
13. [Working with Files](#working-with-files)
14. [Best Practices](#best-practices)
15. [Common Issues and Solutions](#common-issues-and-solutions)
16. [Platform Integrations](#platform-integrations)
17. [Additional Resources](#additional-resources)

---

## ⚠️ CRITICAL CHECKLIST - READ BEFORE IMPLEMENTING ⚠️

**STOP! Before writing any code, verify these requirements:**

### If `widget.enabled: true` in manifest.yml:

- [ ] **ADD `node_id` to `parameters.required`** - Without this, validation fails:
  ```yaml
  parameters:
    required:
      - name: node_id
        type: string
        description: Node ID for widget configuration injection
  ```

### If ANY widget param has `type: secret`:

- [ ] **USE `WidgetParamResolver`** - NOT `_extract_widget_params()`. Secrets arrive as UUIDs that MUST be resolved:
  ```python
  from api.widget_resolver import WidgetParamResolver

  resolver = WidgetParamResolver(self.orchestration_event)
  widget_data = self.orchestration_event.extra_params.get("widget_data", {})
  param1, param2 = resolver.resolve_positional(widget_data, count=2)
  ```

- [ ] **ACCESS BY POSITION, NOT NAME** - Widget data arrives as `widget_param_1`, `widget_param_2`, etc. (matching manifest order)

**Validation errors you'll see if you ignore this:**
> "Widget is enabled but node_id parameter is missing."
> "Secret widget parameters found but WidgetParamResolver is not used in any code file."

---

## Overview

This is a **Chask organization-specific Lambda function** that integrates with the Chask orchestration system. Lambda functions in Chask are invoked by AI agents through the orchestrator and receive structured event data.

**Key Concepts:**
- **OrchestrationEvent**: The primary data structure containing all context about the agent request
- **Widget Parameters**: User-configured secrets and settings passed to the function
- **Agent Lifecycle**: Functions must notify the orchestrator when they complete
- **Tool Calls**: Functions receive structured tool call arguments from LLM agents

---

## Project Structure

```
ReadProjectFileFn/
├── manifest.yml          # Function configuration (required)
│                         # - Defines function name, runtime, handler
│                         # - Lists required/optional parameters
│                         # - Configures widget settings
│
├── src/
│   ├── handler.py        # ⚠️ INFRASTRUCTURE CODE - DO NOT MODIFY
│   │                     # - Lambda entry point with resilient wrapper
│   │                     # - Handles event parsing and error handling
│   │                     # - Guarantees agent liberation via finally block
│   │
│   └── backend/
│       ├── __init__.py
│       └── function_logic.py  # ✏️ IMPLEMENT YOUR BUSINESS LOGIC HERE
│                              # - Contains FunctionBackend class
│                              # - Implement process_request() method
│                              # - Extract parameters and call APIs
│
├── sandbox/              # 🧪 DEVELOPMENT PLAYGROUND
│   └── README.md         # - Test API connections before implementation
│                         # - Prototype and validate integrations
│                         # - Tracked for future reference
│
├── INSTRUCTIONS.md       # This file - AI agent guidance
├── README.md             # User documentation
└── .gitignore
```

**File Responsibilities:**

1. **⚠️ src/handler.py** - INFRASTRUCTURE CODE (DO NOT MODIFY)
   - Entry point for Lambda invocations
   - Parses events and extracts OrchestrationEvent
   - Instantiates FunctionBackend and calls process_request()
   - Catches exceptions and sends error responses
   - **Guarantees agent liberation** via finally block

2. **✏️ src/backend/function_logic.py** - YOUR BUSINESS LOGIC (MODIFY THIS)
   - Contains FunctionBackend class with your custom code
   - Implement the `process_request()` method
   - Extract parameters from OrchestrationEvent
   - Call external APIs, process data, return results
   - Use provided helper methods (_extract_tool_args, _send_response, _extract_widget_params)

**Key Architecture Benefits:**
- **Developer Safety:** Clear separation prevents accidental handler modification
- **Guaranteed Resilience:** Agent is ALWAYS freed, even on crash/timeout
- **Clean Code:** Business logic isolated in one file (function_logic.py)
- **Better Error Handling:** Structured error categories with proper responses

---

## Sandbox: Development Playground

The `sandbox/` folder is your **development playground** for testing and prototyping before committing to the actual function logic.

### When to Use the Sandbox

- **Test API connections** - Verify authentication, endpoints, and response formats
- **Prototype implementations** - Experiment with different approaches
- **Debug integrations** - Isolate and troubleshoot third-party service interactions
- **Validate data transformations** - Test parsing, mapping, and formatting logic

### Sandbox Workflow

```bash
# 1. Create a test script in sandbox
touch sandbox/test_api_connection.py

# 2. Use chask-sdk to get secrets (see below)
# 3. Test your integration
python sandbox/test_api_connection.py

# 4. Once working, move validated logic to src/backend/function_logic.py
```

### Getting Secrets for Sandbox Testing (chask-sdk)

**Important:** The `chask-sdk` is for **sandbox/local development only**. In Lambda functions, secrets are passed via widget parameters (see [Widget Parameters](#widget-parameters) section).

Use the `chask-sdk` to securely retrieve secrets from your organization during sandbox testing:

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

### Secrets: Sandbox vs Lambda

| Context | How to Get Secrets |
|---------|-------------------|
| **Sandbox testing** | Use `chask-sdk` with your SDK token |
| **Lambda function** | Use `self._extract_widget_params()` - secrets are injected by orchestrator |

### Sandbox Best Practices

1. **Tracked for reference** - Sandbox files are committed to serve as future reference
2. **Not deployed** - Sandbox files are excluded from the Lambda package
3. **No credentials in code** - Use `chask-sdk` for testing, widget params in Lambda

---

## Understanding OrchestrationEvent

Every Lambda invocation receives an **OrchestrationEvent** which is a Pydantic model from `chask_foundation.models.events`.

### OrchestrationEvent Structure

```python
from chask_foundation.models.events import OrchestrationEvent

# When your Lambda is invoked, the event will contain:
event = {
    "body": json.dumps({
        "orchestration_event": {
            "event_id": str,                    # Unique event UUID
            "event_type": str,                  # Usually "function_call"
            "orchestration_session_id": str,    # Session UUID
            "organization_id": str,             # Organization UUID
            "user_id": str,                     # User who triggered
            "orchestrator_agent_id": str,       # Agent identifier
            "extra_params": {                   # Additional context
                "tool_calls": [                 # LLM tool call data
                    {
                        "name": "function_name",
                        "args": {
                            "param1": "value1",
                            "param2": "value2"
                        }
                    }
                ],
                "widget_data": {                # Widget parameters (if any)
                    "widgets": [
                        {
                            "name": "param_name",
                            "value": "secret_value"  # Resolved secrets
                        }
                    ]
                }
            }
        },
        "access_token": str,                    # API access token
        "organization_id": str                  # Organization UUID
    })
}
```

---

## Implementing Your Business Logic

### Quick Start: The FunctionBackend Class

Your business logic goes in `src/backend/function_logic.py`. The template provides a `FunctionBackend` class with helper methods - you just implement the `process_request()` method.

**Example Implementation:**

```python
# src/backend/function_logic.py

class FunctionBackend:
    """Your custom function logic."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
        self.response_event_sent = False

    def process_request(self) -> str:
        """
        ✏️ IMPLEMENT YOUR LOGIC HERE

        The handler automatically:
        - Catches any exceptions you raise
        - Sends error responses to the orchestrator
        - Frees the agent even if your code crashes
        """
        # 1. Extract parameters from tool call
        tool_args = self._extract_tool_args()

        # 2. Get required parameters
        action = tool_args.get("action")
        if not action:
            raise ValueError("Missing required parameter: action")

        # 3. Do your business logic
        result = self.my_custom_logic(action)

        # 4. Send response to orchestrator
        self._send_response(result, is_error=False)

        return result

    def my_custom_logic(self, action: str) -> str:
        """Your custom logic here."""
        # TODO: Call APIs, process data, etc.
        return f"Successfully executed: {action}"
```

**Key Points:**

1. **Handler calls your code:** The handler instantiates `FunctionBackend` and calls `process_request()`
2. **Don't worry about errors:** The handler catches all exceptions and handles them properly
3. **Agent liberation is automatic:** The handler's finally block ensures the agent is always freed
4. **Use helper methods:**
   - `self._extract_tool_args()` - Get tool call arguments (from LLM)
   - `self._send_response(message, is_error)` - Send result to orchestrator
   - **For widget params with `type: secret`:** Use `WidgetParamResolver` (see [Widget Parameters](#widget-parameters))
   - **For widget params with `type: string` only:** Use `self._extract_widget_params(param_names)`

### What the Handler Does For You

The `handler.py` file provides a resilient wrapper that:

```python
# This happens automatically - you don't write this code

def lambda_handler(event, context):
    backend = None
    orchestration_event = None

    try:
        # 1. Parse event automatically
        orchestration_event = parse_event(event)

        # 2. Instantiate your backend class
        backend = FunctionBackend(orchestration_event)

        # 3. Call your process_request() method
        result = backend.process_request()  # YOUR CODE RUNS HERE

        # 4. Return success response
        return success_response(result, backend.response_event_sent)

    except Exception as e:
        # 5. Handle errors automatically
        backend._send_response(f"Error: {e}", is_error=True)
        return error_response(str(e))

    finally:
        # 6. ALWAYS free the agent (even on crash/timeout)
        if orchestration_event:
            notify_agent_available(orchestration_event)
```

---

## Understanding Parameters: Two Types

Your Lambda function receives parameters from **two different sources**. Understanding this distinction is critical for proper implementation.

### 1. Function Parameters (from LLM Tool Calls)

**What are they?**
- Required and optional parameters defined in `manifest.yml`
- Passed by the LLM agent when calling your function
- Extracted from `extra_params.tool_calls[0].args`

**Example in manifest.yml:**
```yaml
function:
  name: MyFunctionFn

  # Function parameters (LLM passes these)
  parameters:
    required:
      - name: action
        type: string
        description: Action to perform

    optional:
      - name: verbose
        type: boolean
        description: Enable verbose logging
        default: false
```

**How to access in your code:**
```python
# In FunctionBackend.process_request()
tool_args = self._extract_tool_args()

action = tool_args.get("action")      # Required parameter
verbose = tool_args.get("verbose", False)  # Optional parameter
```

**When the LLM calls your function:**
```python
# LLM generates this tool call
{
    "name": "MyFunctionFn",
    "args": {
        "action": "analyze",
        "verbose": true
    }
}
```

### 2. Widget Parameters (from UI Configuration)

**What are they?**
- Secrets (API keys, tokens) and configuration settings
- Set by users via the Chask UI widget interface
- Stored securely (secrets in AWS Secrets Manager)
- Extracted from `extra_params.widget_data`

**Example in manifest.yml:**
```yaml
function:
  name: MyFunctionFn

  # Widget configuration (user sets these in UI)
  widget:
    enabled: true
    type: dynamic_widget
    params:
      - name: api_token
        type: secret      # Stored in Secrets Manager
        required: true

      - name: api_url
        type: string      # Plain string
        required: false
```

**How to access in your code:**
```python
# In FunctionBackend.process_request()
# ⚠️ IMPORTANT: Use WidgetParamResolver for type: secret params!
from api.widget_resolver import WidgetParamResolver

resolver = WidgetParamResolver(self.orchestration_event)
widget_data = self.orchestration_event.extra_params.get("widget_data", {})

# Resolve secrets by position (order matches manifest.yml widget.params order)
api_token, api_url = resolver.resolve_positional(widget_data, count=2)
# api_token is now the actual secret value (not a UUID)
```

**When users configure the widget:**
- User opens function widget in Chask UI
- Sets `api_token` = their secret API key
- Sets `api_url` = custom endpoint (optional)
- Values stored and passed to every function invocation

### Key Differences

| Aspect | Function Parameters | Widget Parameters |
|--------|-------------------|------------------|
| **Source** | LLM agent (dynamic) | User UI (static) |
| **When set** | Every function call | Once in UI settings |
| **Typical use** | Request-specific data | Configuration & secrets |
| **Access via** | `self._extract_tool_args()` | `WidgetParamResolver` (for secrets) |
| **Examples** | action, query, file_id | api_key, endpoint_url |
| **In manifest** | `parameters.required/optional` | `widget.params` |

### Complete Example

```python
from api.widget_resolver import WidgetParamResolver

class FunctionBackend:
    def process_request(self) -> str:
        # 1. Get function parameters (from LLM)
        tool_args = self._extract_tool_args()
        action = tool_args.get("action")
        query = tool_args.get("query")

        # 2. Get widget parameters (from UI) - USE WidgetParamResolver for secrets!
        resolver = WidgetParamResolver(self.orchestration_event)
        widget_data = self.orchestration_event.extra_params.get("widget_data", {})
        api_token, endpoint_url = resolver.resolve_positional(widget_data, count=2)

        # 3. Use both together in your business logic
        result = call_external_api(
            action=action,
            query=query,
            token=api_token,
            url=endpoint_url
        )

        self._send_response(result)
        return result
```

### When to Use Each Type

**Use Function Parameters for:**
- ✅ Request-specific data that changes every call
- ✅ Data that the LLM determines dynamically
- ✅ User queries, file IDs, actions, filters
- ✅ Data that varies based on conversation context

**Use Widget Parameters for:**
- ✅ API keys, tokens, credentials
- ✅ Configuration that rarely changes
- ✅ Organization-specific settings
- ✅ Endpoint URLs, default values
- ✅ Feature flags

> **REMINDER:** If widget params include `type: secret`, you MUST use `WidgetParamResolver` to access them. See [Widget Parameters](#widget-parameters).

---

## Accessing Event Data

### Step 1: Parse the Event

Lambda events come wrapped in an AWS API Gateway format. You need to extract the body:

```python
import json
from chask_foundation.models.events import OrchestrationEvent

def lambda_handler(event, context):
    # Get AWS request ID for logging
    request_id = context.aws_request_id if context else "unknown"

    # Parse event body
    if isinstance(event, str):
        event = json.loads(event)
    elif "body" in event:
        event = json.loads(event["body"])

    # Validate and parse orchestration event
    orchestration_event = OrchestrationEvent.model_validate(
        event.get("orchestration_event")
    )

    # Extract access token
    access_token = event.get("access_token")
    organization_id = event.get("organization_id")
```

### Step 2: Extract Tool Call Arguments

Agent requests include tool call data in `extra_params.tool_calls`:

```python
# Check event type
if orchestration_event.event_type != "function_call":
    return {"statusCode": 200, "body": "Event type not supported"}

# Extract tool calls
tool_calls = orchestration_event.extra_params.get("tool_calls", [])
if not tool_calls:
    return error_response("No tool calls found", 400)

# Get first tool call arguments
tool_call = tool_calls[0]
tool_name = tool_call.get("name")
tool_args = tool_call.get("args", {})

# Merge args into extra_params for easier access
orchestration_event.extra_params.update(tool_args)

# Now access parameters directly
param1 = orchestration_event.extra_params.get("param1")
param2 = orchestration_event.extra_params.get("param2")
```

### Step 3: Access Organization Context

```python
# Organization information
org_id = orchestration_event.organization_id
user_id = orchestration_event.user_id
session_id = orchestration_event.orchestration_session_id

# Use these for:
# - Querying organization-specific data
# - Logging and tracking
# - Scoping API calls
```

---

## Widget Parameters

Widget parameters allow users to configure secrets (API keys, tokens) and settings through the Chask UI.

### Defining Widget Parameters in manifest.yml

```yaml
function:
  name: MyFunctionFn

  # Function parameters - node_id is REQUIRED when widget is enabled
  parameters:
    required:
      - name: node_id
        type: string
        description: Node ID for widget configuration injection (REQUIRED when widget.enabled is true)

  # Widget configuration
  widget:
    enabled: true
    type: dynamic_widget  # Or custom widget type
    params:
      - name: api_token
        type: secret       # Secret stored in AWS Secrets Manager
        required: true
        description: API token for external service

      - name: endpoint_url
        type: string       # Plain string parameter
        required: false
        description: Custom endpoint URL
```

> **IMPORTANT:** When `widget.enabled: true`, you MUST include `node_id` in `parameters.required`. Without it, the validator will fail with: *"Widget is enabled but node_id parameter is missing."*

### Accessing Widget Parameters

Widget parameters are passed in `extra_params.widget_data`.

**IMPORTANT: Two types of widget parameters require different handling:**

| Parameter Type | How it arrives | How to access |
|---------------|----------------|---------------|
| `type: string` | Plain text value | Direct access or `extract_widget_params()` |
| `type: secret` | **UUID reference** | **MUST use `WidgetParamResolver`** |

#### Secret Parameters (type: secret) - REQUIRES WidgetParamResolver

**CRITICAL:** Secret parameters are NOT passed as plain values. They arrive as **UUIDs** that reference secrets stored in AWS Secrets Manager. You **MUST** use `WidgetParamResolver` to resolve these UUIDs to actual values.

**Validation Error:**
If you have secret parameters but don't use WidgetParamResolver, you will see:
> "Secret widget parameters found but WidgetParamResolver is not used in any code file. Secrets must be resolved using WidgetParamResolver.resolve_positional() to evaluate from AWS Secrets Manager."

**Widget data format with secrets:**
```python
# What you receive (secret is a UUID, not the actual value!)
{
  "widget_data": {
    "widget_param_1": "550e8400-e29b-41d4-a716-446655440000",  # UUID for api_token
    "widget_param_2": "https://api.example.com"                 # Direct string value
  }
}
```

**REQUIRED approach for secrets - Using WidgetParamResolver:**

```python
from api.widget_resolver import WidgetParamResolver

def process_request(self) -> str:
    # Initialize resolver (provides access_token for API calls)
    resolver = WidgetParamResolver(self.orchestration_event)

    # Get widget data
    widget_data = self.orchestration_event.extra_params.get("widget_data", {})

    # Resolve ALL widget parameters (automatically detects and resolves UUIDs)
    # Order matches manifest.yml widget.params order!
    api_token, endpoint_url = resolver.resolve_positional(widget_data, count=2)

    # api_token is now the actual secret value (resolved from AWS Secrets Manager)
    # endpoint_url is the direct string value (was not a UUID)

    # Use in your business logic
    response = requests.get(endpoint_url, headers={"Authorization": f"Bearer {api_token}"})
```

**How WidgetParamResolver works:**
1. Receives widget_data with `widget_param_1`, `widget_param_2`, etc.
2. For each value, checks if it's a UUID (36-char format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
3. If UUID: Calls `organizations/secrets/retrieve` API to get actual secret value
4. If not UUID: Uses the value directly (string parameters)
5. Returns resolved values in positional order

**Alternative resolve() method (returns dictionary):**
```python
resolver = WidgetParamResolver(self.orchestration_event)
widget_data = self.orchestration_event.extra_params.get("widget_data", {})

# Returns dict with all params resolved
resolved = resolver.resolve(widget_data)
# {"widget_param_1": "actual_secret_value", "widget_param_2": "https://api.example.com"}

api_token = resolved.get("widget_param_1")
endpoint_url = resolved.get("widget_param_2")
```

#### String Parameters Only (No Secrets)

If your function ONLY uses `type: string` parameters (no secrets), you can use a simpler approach:

```python
def extract_widget_params(orchestration_event: OrchestrationEvent, param_names: list) -> dict:
    """
    Extract widget parameters for STRING-ONLY parameters.

    WARNING: This does NOT resolve secrets! If you have type: secret parameters,
    you MUST use WidgetParamResolver instead.

    Args:
        orchestration_event: The orchestration event
        param_names: List of parameter names to extract

    Returns:
        Dictionary with extracted parameter values
    """
    widget_data = orchestration_event.extra_params.get("widget_data", {})

    # Try nested widgets array first (production format)
    widgets = widget_data.get("widgets", [])
    widget_values = {w.get("name"): w.get("value") for w in widgets}

    # Extract params with fallback to direct access (test format)
    result = {}
    for param_name in param_names:
        result[param_name] = widget_values.get(param_name) or widget_data.get(param_name)

    return result

# Usage (ONLY for string parameters!)
params = extract_widget_params(orchestration_event, ["endpoint_url", "timeout"])
endpoint_url = params["endpoint_url"]
timeout = params["timeout"]
```

#### Quick Reference: Which Method to Use

| Your manifest has | Use this method |
|-------------------|-----------------|
| Any `type: secret` params | `WidgetParamResolver.resolve_positional()` or `.resolve()` |
| Only `type: string` params | `extract_widget_params()` or `WidgetParamResolver` |
| No widget params | N/A - no widget data to extract |

### Widget Parameter Types

**Secret Parameters:**
- Stored in AWS Secrets Manager
- Path: `chask/org/<organization_id>/<param_name>`
- **Arrive as UUIDs, MUST be resolved via WidgetParamResolver**
- Never logged or exposed

**String Parameters:**
- Plain text configuration values
- Can be optional with defaults
- Suitable for non-sensitive data
- Can be accessed directly without WidgetParamResolver

### Node ID for Widget Configuration

**CRITICAL REQUIREMENT:** When `widget.enabled: true` in your manifest.yml, you **MUST** add `node_id` as a parameter in the `parameters.required` section.

**Validation Error:**
If you try to publish a function with `widget.enabled: true` but without `node_id` in parameters, you will see:
> "Widget is enabled but node_id parameter is missing. node_id is required to identify which widget configuration to inject."

**How it works:**
1. The Chask operator receives the function call with a `node_id`
2. The operator looks up the widget configuration associated with that node
3. The operator injects the resolved `widget_data` into the Lambda event's `extra_params`
4. Your function receives the widget parameters through `extra_params.widget_data`

**How to fix - Add node_id to manifest.yml:**

```yaml
function:
  name: MyFunctionFn

  # REQUIRED: Add node_id to parameters when widget is enabled
  parameters:
    required:
      - name: node_id
        type: string
        description: Node ID for widget configuration injection
      # ... your other parameters

  widget:
    enabled: true  # <-- This triggers the node_id requirement
    params:
      - name: api_token
        type: secret
        required: true
```

**When node_id is required vs not required:**

| Scenario | node_id Required? |
|----------|------------------|
| `widget.enabled: true` | **YES** - Must be in parameters.required |
| `widget.enabled: false` | No |
| No widget section | No |
| Widget section with `params: []` but `enabled: true` | **YES** |

**Example test file with `node_id`:**
```json
{
  "function_name": "MyIntegrationFn",
  "args": {
    "node_id": "550e8400-e29b-41d4-a716-446655440000",
    "action": "sync_data"
  },
  "widget_params": {
    "api_token": "test_token_123",
    "endpoint_url": "https://api.example.com"
  },
  "metadata": {
    "description": "Test with widget configuration"
  }
}
```

**Without `node_id`:** The Chask operator won't know which widget configuration to inject, and your function will receive empty or missing `widget_data`, causing failures when trying to access secrets or settings.

---

## Emitting Results

### Sending Results to Orchestrator via Kafka

**CRITICAL:** All Lambda functions MUST send their results back to the orchestrator via Kafka. This is how the agent receives the response.

```python
from api.orchestrator_requests import orchestrator_api_manager

def send_response_to_orchestrator(
    orchestration_event: OrchestrationEvent, message: str, is_error: bool = False
) -> bool:
    """
    Send the function result back to the orchestrator via Kafka.

    Uses evolve_event API to create proper parent-child event linkage
    for event traceability in the Event Tracking System.

    Returns:
        True if event was sent successfully, False otherwise
    """
    try:
        # Extract tool call information from original event
        original_extra_params = orchestration_event.extra_params or {}
        tool_call_id = None
        tool_name = None
        if ("tool_calls" in original_extra_params and
            original_extra_params["tool_calls"]):
            tool_call = original_extra_params["tool_calls"][0]
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name")

        # Build extra_params for the response event
        extra_params = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "is_error": is_error
        }

        # ====================================================================
        # CRITICAL: Preserve test execution flags for 'chask function test'
        # ====================================================================
        # These flags MUST be preserved to ensure test executions are properly
        # tracked and results are returned to the CLI instead of being routed
        # to the operator. DO NOT REMOVE OR MODIFY THIS CODE.
        if original_extra_params.get("is_test"):
            extra_params["is_test"] = True
            if original_extra_params.get("test_execution_uuid"):
                extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]
        # ====================================================================

        # Evolve the event to create a linked function_call_response child event
        # This maintains parent-child relationships via evolved_from_uuid
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

        if evolve_response.get("status_code") not in (200, 201):
            error_msg = evolve_response.get("error", "Unknown error")
            raise Exception(f"Failed to evolve event: {error_msg}")

        # Reconstruct the evolved event for Kafka forwarding
        evolved_uuid = evolve_response.get("uuid")
        if not evolved_uuid:
            raise Exception("API response missing uuid for evolved event")

        response_event = orchestration_event.model_copy(deep=True)
        response_event.event_id = evolved_uuid
        response_event.event_type = "function_call_response"
        response_event.source = "agent"
        response_event.target = "orchestrator"
        response_event.prompt = message
        response_event.extra_params = evolve_response.get("extra_params", extra_params)

        # Send the evolved orchestration event to orchestrator
        orchestrator_api_manager.call(
            "forward_oe_to_kafka",
            orchestration_event=response_event.model_dump(),
            topic="orchestrator",
            access_token=response_event.access_token,
            organization_id=response_event.organization.organization_id,
        )

        logger.info(
            f"Response sent to orchestrator via Kafka "
            f"[evolved from {orchestration_event.event_id} -> {evolved_uuid}]"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to send response to orchestrator: {e}")
        return False
```

### Lambda Response Format

Lambda functions MUST include a `response_event_sent` flag in their response to indicate whether the Kafka event was successfully sent.

**⚠️ CRITICAL: `response_event_sent` Location**

The `response_event_sent` flag MUST be at the `body` level, NOT inside the `result` object!

**❌ WRONG - Flag inside result:**
```python
return {
    "statusCode": 200,
    "body": {
        "status": "ok",
        "result": {
            "message": "Success",
            "response_event_sent": True  # ❌ WRONG! Agent manager won't find it here
        }
    }
}
```

**✅ CORRECT - Flag at body level:**
```python
return {
    "statusCode": 200,
    "body": {
        "status": "ok",
        "result": {
            "message": "Success"
        },
        "response_event_sent": True  # ✅ CORRECT! Agent manager checks here
    }
}
```

**Complete Example:**

```python
def lambda_handler(event, context):
    try:
        # Parse event and process...
        orchestration_event = OrchestrationEvent.model_validate(
            event.get("orchestration_event")
        )

        result_message = "Successfully processed request"

        # Send to orchestrator via Kafka (REQUIRED!)
        response_sent = send_response_to_orchestrator(orchestration_event, result_message, is_error=False)

        # Use helper functions to ensure correct structure
        return success_response(
            result={"message": result_message},
            response_event_sent=response_sent
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

        # Send error to orchestrator
        response_sent = False
        if 'orchestration_event' in locals():
            response_sent = send_response_to_orchestrator(
                orchestration_event,
                f"Lambda error: {str(e)}",
                is_error=True
            )

        # Use helper function to ensure correct structure
        return error_response(
            error_message=str(e),
            response_event_sent=response_sent
        )
```

**Why this matters:**
- The orchestrator's agent manager checks `body.response_event_sent` to determine if the Lambda successfully sent the Kafka event
- If `response_event_sent: false` (or missing/wrong location), the agent manager sends a fallback event
- This ensures the agent ALWAYS receives a response, even if Kafka fails
- Without this flag at the correct location, the fallback mechanism won't work properly

**Response Event Sent Flag Values:**
- `true`: Lambda successfully sent response via Kafka
- `false`: Lambda failed to send (network error, API unavailable, etc.)
- The orchestrator checks this flag at `body.response_event_sent` and sends a fallback if needed
- **Always use the provided `success_response()` and `error_response()` helper functions** to ensure correct structure

---

## Complete API Reference

The examples above show the two most critical endpoints for Lambda functions. However, chask-foundation provides **~98 additional API endpoints** across 16 domains.

📚 **See `API_ENDPOINTS.md` for the complete reference.**

### Available API Domains

- **Orchestrator API (34 endpoints)** - Event evolution, session management, Kafka forwarding, event tracking
- **Pipeline API (18 endpoints)** - Pipeline/node operations, widget data, status updates, testing
- **LLM API (3 endpoints)** - Call logging, batch logging, context retrieval
- **Files API (5 endpoints)** - File uploads, PDF reading, email attachments, session files
- **Organizations API (12 endpoints)** - Customer groups, function management, secrets, tokens
- **Functions API (2 endpoints)** - Function search and metadata
- **WhatsApp API (10 endpoints)** - Messaging, templates, conversations, media
- **Email API (2 endpoints)** - Email sending, thread retrieval
- **Outbound API (4 endpoints)** - Batch messaging, job management
- **Internal Channels API (3 endpoints)** - Internal conversations and messages
- **Channels API (2 endpoints)** - Channel management, template retrieval
- **CRM API (1 endpoint)** - Customer validation
- **Agents API (1 endpoint)** - Context by channel
- **Analysts API (1 endpoint)** - Organization analysts
- **Fintoc API (1 endpoint)** - Payment link tokens
- **Firebase API (1 endpoint)** - User token retrieval

### Quick Examples

**Create a pipeline:**
```python
from chask_foundation.api.pipeline_requests import pipeline_api_manager

response = pipeline_api_manager.call(
    "create_pipeline",
    is_template=False,
    organization_customer_uuid=customer_uuid,
    access_token=token,
    organization_id=org_id
)
pipeline_id = response.get("pipeline_id")
```

**Search for functions:**
```python
from chask_foundation.api.function_requests import function_api_manager

response = function_api_manager.call(
    "search_functions",
    search="payment",
    assigned_only=True,
    fuzzy_threshold=70,
    access_token=token,
    organization_id=org_id
)
functions = response.get("functions", [])
```

**Get function metadata:**
```python
from chask_foundation.api.function_requests import function_api_manager

response = function_api_manager.call(
    "get_function_by_uuid",
    function_uuid="func-uuid-123",
    access_token=token,
    organization_id=org_id
)
function_name = response.get("name")
parameters = response.get("parameters", {})
```

**Upload a file:**
```python
from chask_foundation.api.files_requests import files_api_manager

with open("report.pdf", "rb") as f:
    response = files_api_manager.call(
        "upload_file",
        file=f,
        orchestration_session_uuids=[session_uuid],
        access_token=token,
        organization_id=org_id
    )
file_uuid = response.get("file_uuid")
```

**Log LLM call:**
```python
from chask_foundation.api.llm_requests import llm_api_manager

llm_api_manager.call(
    "log_call",
    provider="anthropic",
    model="claude-sonnet-4.5",
    call_type="chat",
    latency_ms=1500,
    input_tokens=1000,
    output_tokens=500,
    access_token=token,
    organization_id=org_id
)
```

For complete documentation of all endpoints, parameters, and examples, see **`API_ENDPOINTS.md`**.

---

## ⚠️ CRITICAL: Preserving Test Execution Flags ⚠️

**THIS IS EXTREMELY IMPORTANT - READ CAREFULLY**

When creating response events in `send_response_to_orchestrator()`, you **MUST** preserve the `is_test` and `test_execution_uuid` flags if they exist in the original event.

### Why This Matters

Without these flags:
- ✗ Test results will be routed to the operator instead of the test handler
- ✗ The CLI will hang indefinitely waiting for results
- ✗ The `chask function test` command will NOT work
- ✗ Additional unwanted function calls will be triggered

### Required Code (Already in Template)

The template `send_response_to_orchestrator()` function includes this code:

```python
# ====================================================================
# CRITICAL: Preserve test execution flags for 'chask function test'
# ====================================================================
# These flags MUST be preserved to ensure test executions are properly
# tracked and results are returned to the CLI instead of being routed
# to the operator. DO NOT REMOVE OR MODIFY THIS CODE.
original_extra_params = orchestration_event.extra_params or {}
if original_extra_params.get("is_test"):
    response_event.extra_params["is_test"] = True
    if original_extra_params.get("test_execution_uuid"):
        response_event.extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]
# ====================================================================
```

### ⛔ DO NOT

- ❌ Remove this code block
- ❌ Modify the flag preservation logic
- ❌ Create response events without preserving these flags
- ❌ Override `extra_params` with a new empty dict after copying

### ✅ ALWAYS

- ✓ Keep the test flag preservation code intact
- ✓ Use `evolve_event` API to create linked child events with proper `evolved_from_uuid`
- ✓ Use `orchestration_event.model_copy(deep=True)` AFTER evolve_event to reconstruct local event
- ✓ Preserve existing `extra_params` before updating with new values

---

## Testing Your Lambda Function

### Using the CLI Test Command

Test your Lambda function locally before deploying to production:

```bash
# Test with a test file
chask function test test_1.json

# Skip fetching CloudWatch logs
chask function test test_1.json --no-logs
```

### Test File Format

Create test files in JSON format:

```json
{
  "function_name": "YourFunctionFn",
  "args": {
    "param1": "value1",
    "param2": "value2"
  },
  "widget_params": {
    "api_token": "test_token_123",
    "api_secret": "test_secret_456"
  },
  "node_id": "node_uuid_here",
  "metadata": {
    "description": "Basic test case"
  }
}
```

**Test File Structure:**
- `function_name` (required): Name of the Lambda function to test
- `args` (required): Arguments to pass to the function (will be in tool_calls)
- `widget_params` (optional): Widget parameters if your function uses widgets
- `node_id` (required for organization functions with widgets): Node UUID that identifies which widget configuration to use. The Chask operator uses this to inject the correct widget_data into your function's event
- `metadata` (optional): Additional metadata like description

### Viewing Test Results

The CLI will display:
- ✅ Test status (success/failure)
- 📋 Result message from your Lambda
- ⏱️ Execution time
- 🔗 Lambda Request ID (for CloudWatch logs)
- 🆔 Session UUID (for debugging)

### Accessing CloudWatch Logs

After running a test, you can view the Lambda execution logs:

```bash
# Option 1: Use AWS CLI with the Request ID from test output
aws logs tail /aws/lambda/your-function-name \
  --profile development \
  --filter-pattern "<request-id>" \
  --format short

# Option 2: View all recent logs
aws logs tail /aws/lambda/your-function-name \
  --profile development \
  --follow

# Option 3: Use the API endpoint (if Lambda Request ID is available)
curl -H "Authorization: Bearer $ANALYST_TOKEN" \
  "https://your-domain/api/v2/organizations/$ORG_ID/functions/test/$TEST_UUID/logs"
```

### Test Execution Flow

1. **CLI** sends test request to chask_api
2. **chask_api** creates test execution record and publishes to Kafka
3. **Orchestrator** receives event and invokes your Lambda
4. **Lambda** processes request and sends response back via Kafka
5. **Orchestrator** updates test execution record with results
6. **CLI** polls for results and displays them

**Important:** The test execution uses the same code path as production, but with `is_test: true` flag to track results separately.

---

## Complete Examples

### Example 1: Simple Function with Parameters

```python
import json
import logging
from chask_foundation.models.events import OrchestrationEvent
from api.notify import notify_agent_available

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "unknown"
    orchestration_event = None

    try:
        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        elif "body" in event:
            event = json.loads(event["body"])

        orchestration_event = OrchestrationEvent.model_validate(
            event.get("orchestration_event")
        )

        # Check event type
        if orchestration_event.event_type != "function_call":
            return {"statusCode": 200, "body": "Not a function call"}

        # Extract tool call arguments
        tool_calls = orchestration_event.extra_params.get("tool_calls", [])
        tool_args = tool_calls[0].get("args", {})

        # Get parameters
        action = tool_args.get("action")
        data = tool_args.get("data")

        logger.info(f"Processing action: {action}")

        # Process request
        result = process_action(action, data)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "result": result
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "error": str(e)
            })
        }

    finally:
        if orchestration_event:
            notify_agent_available(orchestration_event, logger, request_id)

def process_action(action: str, data: dict) -> dict:
    # Your business logic here
    return {"processed": True, "action": action}
```

### Example 2: Function with Widget Parameters

```python
import json
import logging
from chask_foundation.models.events import OrchestrationEvent
from api.widget_resolver import WidgetParamResolver
from api.notify import notify_agent_available

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "unknown"
    orchestration_event = None

    try:
        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        elif "body" in event:
            event = json.loads(event["body"])

        orchestration_event = OrchestrationEvent.model_validate(
            event.get("orchestration_event")
        )

        # Resolve widget parameters
        resolver = WidgetParamResolver(orchestration_event)
        widget_data = orchestration_event.extra_params.get("widget_data", {})

        # Get secrets (order must match manifest.yml)
        api_token, api_secret = resolver.resolve_positional(widget_data, count=2)

        logger.info(f"Resolved widget parameters for org: {orchestration_event.organization_id}")

        # Extract tool call arguments
        tool_calls = orchestration_event.extra_params.get("tool_calls", [])
        tool_args = tool_calls[0].get("args", {})
        orchestration_event.extra_params.update(tool_args)

        # Use parameters to call external API
        result = call_external_api(
            api_token=api_token,
            api_secret=api_secret,
            params=tool_args
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "result": result
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "error": str(e)
            })
        }

    finally:
        if orchestration_event:
            notify_agent_available(orchestration_event, logger, request_id)

def call_external_api(api_token: str, api_secret: str, params: dict) -> dict:
    # Call external service using secrets
    import requests

    headers = {
        "Authorization": f"Bearer {api_token}",
        "X-API-Secret": api_secret
    }

    response = requests.post(
        "https://api.example.com/endpoint",
        headers=headers,
        json=params
    )

    return response.json()
```

### Example 3: Function with Agent Processing

```python
import json
import logging
from chask_foundation.models.events import OrchestrationEvent
from api.widget_resolver import WidgetParamResolver
from api.notify import notify_agent_available

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MyAgent:
    """Custom agent for processing requests."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_request(self, orchestration_event: OrchestrationEvent) -> str:
        """Process the orchestration event and return result."""

        # Extract parameters
        tool_calls = orchestration_event.extra_params.get("tool_calls", [])
        tool_args = tool_calls[0].get("args", {})

        query = tool_args.get("query")
        options = tool_args.get("options", {})

        logger.info(f"Processing query: {query}")

        # Implement your agent logic here
        result = self.analyze_query(query, options)

        return result

    def analyze_query(self, query: str, options: dict) -> str:
        # Your analysis logic
        return f"Analysis complete for: {query}"

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "unknown"
    orchestration_event = None

    try:
        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        elif "body" in event:
            event = json.loads(event["body"])

        orchestration_event = OrchestrationEvent.model_validate(
            event.get("orchestration_event")
        )

        # Check event type
        if orchestration_event.event_type != "function_call":
            return {"statusCode": 200, "body": "Not a function call"}

        # Resolve widget parameters
        resolver = WidgetParamResolver(orchestration_event)
        widget_data = orchestration_event.extra_params.get("widget_data", {})
        api_key = resolver.resolve_positional(widget_data, count=1)[0]

        # Initialize agent
        agent = MyAgent(api_key=api_key)

        # Process request
        result_message = agent.process_request(orchestration_event)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "message": result_message
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "error": str(e)
            })
        }

    finally:
        if orchestration_event:
            notify_agent_available(orchestration_event, logger, request_id)
```

---

## Working with Files

Lambda functions often need to handle files - either processing files uploaded by users or generating files to return. The Chask system provides a clean API layer for file operations without requiring direct S3/boto3 access.

### When to Use File Handling

- **Generating reports**: PDF, Excel, CSV files
- **Processing user uploads**: Images, documents, data files
- **Creating artifacts**: Charts, graphs, processed data
- **Downloading external resources**: API responses, web scraping results

---

### Uploading Files to Return to User

When your Lambda function generates a file that needs to be returned to the user, use the `files_api_manager` to upload it:

```python
from api.files_requests import files_api_manager
from chask_foundation.backend.models import OrchestrationEvent
import io

def upload_generated_file(
    orchestration_event: OrchestrationEvent,
    file_content: bytes,
    filename: str
) -> dict:
    """
    Upload a generated file and return its metadata.

    Args:
        orchestration_event: The orchestration event context
        file_content: File content as bytes
        filename: Desired filename (e.g., "report.pdf")

    Returns:
        dict: File metadata with file_uuid, file_url, etc.
    """
    # Wrap content in file-like object
    file_obj = io.BytesIO(file_content)
    file_obj.name = filename

    # Upload file via API
    result = files_api_manager.call(
        "upload_file",
        file=file_obj,
        orchestration_session_uuids=[
            orchestration_event.orchestration_session_uuid
        ] if orchestration_event.orchestration_session_uuid else None,
        internal_orchestration_session_uuid=orchestration_event.internal_orchestration_session_uuid,
        shared=False,  # Set to True if file should be shared across sessions
        access_token=orchestration_event.access_token,
        organization_id=orchestration_event.organization.organization_id,
    )

    # Handle response (could be Response object or dict)
    if hasattr(result, "status_code"):
        if result.status_code != 200:
            raise ValueError(f"File upload failed: {result.status_code}")
        file_data = result.json()
    else:
        file_data = result

    # Returns: {"file_uuid": "...", "file_url": "...", "file_name": "...", ...}
    return file_data
```

**Response Structure:**
```python
{
    "file_uuid": "abc-123-def",           # Unique file identifier
    "file_url": "https://...",            # Presigned URL for download
    "file_name": "report.pdf",            # Original filename
    "mime_type": "application/pdf",       # MIME type
    "content_type": "application/pdf"     # Content type
}
```

**Including File URL in Response:**

After uploading a file, include the file URL in your response message to the orchestrator:

```python
# Upload file
file_data = upload_generated_file(orchestration_event, pdf_bytes, "report.pdf")

# Create response message with file URL
response_message = f"""Report generated successfully!

📄 Download your report: {file_data['file_url']}
Filename: {file_data['file_name']}
"""

# Send to orchestrator
response_sent = send_response_to_orchestrator(
    orchestration_event,
    response_message,
    is_error=False
)
```

---

### Accessing User-Uploaded Files

When users upload files, they're available through the orchestration event's `extra_params`:

```python
def get_user_uploaded_files(orchestration_event: OrchestrationEvent) -> list:
    """
    Extract file URLs from user-uploaded attachments.

    Returns:
        list: List of dicts with file_uuid, file_url, file_name, etc.
    """
    # Files are in extra_params.attachments
    attachments = orchestration_event.extra_params.get("attachments", [])

    # Each attachment has file_uuid, file_url (presigned), file_name, etc.
    files = []
    for attachment in attachments:
        if attachment.get("file_url"):
            files.append({
                "uuid": attachment.get("file_uuid"),
                "url": attachment.get("file_url"),  # Presigned URL
                "name": attachment.get("file_name"),
                "type": attachment.get("mime_type")
            })

    return files
```

**Downloading Files for Processing:**

Use the presigned URLs to download files:

```python
import requests
import pathlib
from urllib.parse import urlparse, unquote

def download_file(file_url: str, save_path: str = None) -> str:
    """
    Download a file from presigned URL.

    Args:
        file_url: Presigned URL from file_url field
        save_path: Optional local path to save file

    Returns:
        str: Path to downloaded file
    """
    # Download file
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()

    # Extract filename from URL if not provided
    if not save_path:
        filename = unquote(pathlib.Path(urlparse(file_url).path).name)
        save_path = f"/tmp/{filename}"

    # Save to disk
    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path
```

**Alternative: Get All Session Files via API**

You can also retrieve all files for a session:

```python
def get_all_session_files(orchestration_event: OrchestrationEvent) -> list:
    """Get all files associated with the current session."""
    result = files_api_manager.call(
        "get_all_files_for_session",
        orchestration_session_uuid=orchestration_event.orchestration_session_uuid,
        internal_orchestration_session_uuid=orchestration_event.internal_orchestration_session_uuid,
        access_token=orchestration_event.access_token,
        organization_id=orchestration_event.organization.organization_id,
    )

    if hasattr(result, "status_code"):
        if result.status_code != 200:
            raise ValueError(f"Failed to get files: {result.status_code}")
        return result.json().get("files", [])
    return result.get("files", [])
```

**Get a Specific File by UUID:**

When your function receives a `file_uuid` parameter, retrieve the file URL like this:

```python
from api.files_requests import files_api_manager

def get_file_by_uuid(orchestration_event: OrchestrationEvent, file_uuid: str) -> dict:
    """
    Get a specific file's download URL by its UUID.

    Args:
        orchestration_event: The orchestration event
        file_uuid: UUID of the file to retrieve

    Returns:
        dict with file_uuid, file_url, file_name, mime_type

    Raises:
        ValueError: If file not found in session
    """
    # Get all files for the session
    files_response = files_api_manager.call(
        "get_all_files_for_session",
        orchestration_session_uuid=orchestration_event.orchestration_session_uuid,
        access_token=orchestration_event.access_token,
        organization_id=orchestration_event.organization.organization_id,
    )

    if not files_response or "files" not in files_response:
        raise ValueError("Failed to get files for session")

    # Find the file with matching UUID
    for file_data in files_response["files"]:
        if file_data.get("file_uuid") == file_uuid:
            return file_data

    raise ValueError(f"File {file_uuid} not found in session")


# Usage in process_request():
def process_request(self) -> str:
    tool_args = self._extract_tool_args()
    file_uuid = tool_args.get("file_uuid")

    # Get file download URL
    file_data = get_file_by_uuid(self.orchestration_event, file_uuid)
    file_url = file_data["file_url"]  # Presigned URL

    # Download and process the file
    # ...
```

---

### Complete File Handling Example

Here's a complete example that processes user-uploaded images and returns a generated report:

```python
import json
import logging
import io
from typing import List
from PIL import Image
import requests
from chask_foundation.backend.models import OrchestrationEvent
from api.files_requests import files_api_manager
from api.orchestrator_requests import orchestrator_api_manager

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "unknown"

    try:
        # Parse event
        if isinstance(event, str):
            event = json.loads(event)
        elif "body" in event:
            body = event["body"]
            event = json.loads(body) if isinstance(body, str) else body

        orchestration_event = OrchestrationEvent.model_validate(
            event.get("orchestration_event")
        )

        logger.info(f"[{request_id}] Processing image analysis request")

        # 1. Get user-uploaded files
        attachments = orchestration_event.extra_params.get("attachments", [])
        image_files = [
            att for att in attachments
            if att.get("mime_type", "").startswith("image/")
        ]

        if not image_files:
            error_msg = "No image files found in request"
            send_response_to_orchestrator(orchestration_event, error_msg, is_error=True)
            return error_response(error_msg, 400)

        logger.info(f"Found {len(image_files)} images to process")

        # 2. Download and process images
        results = []
        for img_file in image_files:
            file_url = img_file.get("file_url")
            file_name = img_file.get("file_name")

            # Download image
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            # Process with PIL
            image = Image.open(io.BytesIO(response.content))
            width, height = image.size

            results.append({
                "name": file_name,
                "dimensions": f"{width}x{height}",
                "format": image.format
            })

        # 3. Generate report file
        report_content = "Image Analysis Report\\n\\n"
        for result in results:
            report_content += f"File: {result['name']}\\n"
            report_content += f"  Dimensions: {result['dimensions']}\\n"
            report_content += f"  Format: {result['format']}\\n\\n"

        # 4. Upload report file
        report_bytes = report_content.encode("utf-8")
        file_obj = io.BytesIO(report_bytes)
        file_obj.name = "image_analysis_report.txt"

        upload_result = files_api_manager.call(
            "upload_file",
            file=file_obj,
            orchestration_session_uuids=[
                orchestration_event.orchestration_session_uuid
            ] if orchestration_event.orchestration_session_uuid else None,
            internal_orchestration_session_uuid=orchestration_event.internal_orchestration_session_uuid,
            shared=False,
            access_token=orchestration_event.access_token,
            organization_id=orchestration_event.organization.organization_id,
        )

        # Handle upload response
        if hasattr(upload_result, "status_code"):
            if upload_result.status_code != 200:
                raise ValueError(f"Upload failed: {upload_result.status_code}")
            file_data = upload_result.json()
        else:
            file_data = upload_result

        # 5. Send response with file URL
        response_message = f"""Image analysis complete!

Processed {len(image_files)} images.

📄 Download full report: {file_data['file_url']}

Summary:
{report_content}
"""

        response_sent = send_response_to_orchestrator(
            orchestration_event,
            response_message,
            is_error=False
        )

        return success_response(
            result={
                "message": "Analysis complete",
                "images_processed": len(image_files),
                "report_url": file_data["file_url"],
                "report_uuid": file_data["file_uuid"]
            },
            response_event_sent=response_sent
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

        response_sent = False
        try:
            if 'orchestration_event' in locals():
                response_sent = send_response_to_orchestrator(
                    orchestration_event,
                    f"Error processing images: {str(e)}",
                    is_error=True
                )
        except:
            pass

        return error_response(str(e), response_event_sent=response_sent)

def send_response_to_orchestrator(
    orchestration_event: OrchestrationEvent, message: str, is_error: bool = False
) -> bool:
    """Send response back to orchestrator via Kafka using evolve_event."""
    try:
        # Extract tool call info
        original_extra_params = orchestration_event.extra_params or {}
        tool_call_id = None
        tool_name = None
        if ("tool_calls" in original_extra_params and
            original_extra_params["tool_calls"]):
            tool_call = original_extra_params["tool_calls"][0]
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name")

        extra_params = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "is_error": is_error
        }

        # Preserve test flags
        if original_extra_params.get("is_test"):
            extra_params["is_test"] = True
            if original_extra_params.get("test_execution_uuid"):
                extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]

        # Evolve the event to maintain parent-child linkage
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

        if evolve_response.get("status_code") not in (200, 201):
            raise Exception(f"Failed to evolve event: {evolve_response.get('error')}")

        evolved_uuid = evolve_response.get("uuid")
        response_event = orchestration_event.model_copy(deep=True)
        response_event.event_id = evolved_uuid
        response_event.event_type = "function_call_response"
        response_event.source = "agent"
        response_event.target = "orchestrator"
        response_event.prompt = message
        response_event.extra_params = evolve_response.get("extra_params", extra_params)

        orchestrator_api_manager.call(
            "forward_oe_to_kafka",
            orchestration_event=response_event.model_dump(),
            topic="orchestrator",
            access_token=response_event.access_token,
            organization_id=response_event.organization.organization_id,
        )

        return True
    except Exception as e:
        logger.error(f"Failed to send response: {e}")
        return False

def success_response(result: dict, response_event_sent: bool = False, status_code: int = 200) -> dict:
    return {
        "statusCode": status_code,
        "body": {
            "status": "ok",
            "result": result,
            "response_event_sent": response_event_sent
        }
    }

def error_response(error_message: str, response_event_sent: bool = False, status_code: int = 500) -> dict:
    return {
        "statusCode": status_code,
        "body": {
            "status": "error",
            "error": error_message,
            "response_event_sent": response_event_sent
        }
    }
```

---

### Best Practices for File Operations

1. **Always Link Files to Sessions**
   ```python
   # Always provide orchestration session UUIDs when uploading
   orchestration_session_uuids=[orchestration_event.orchestration_session_uuid]
   ```

2. **Clean Up Temporary Files**
   ```python
   import os
   try:
       local_path = download_file(file_url)
       # Process file...
   finally:
       if os.path.exists(local_path):
           os.remove(local_path)
   ```

3. **Handle Response Types**
   ```python
   # API calls may return Response objects or dicts
   if hasattr(result, "status_code"):
       data = result.json()
   else:
       data = result
   ```

4. **Validate File Types**
   ```python
   allowed_types = ["image/jpeg", "image/png", "application/pdf"]
   for attachment in attachments:
       mime_type = attachment.get("mime_type")
       if mime_type not in allowed_types:
           raise ValueError(f"Unsupported file type: {mime_type}")
   ```

5. **Use Descriptive Filenames**
   ```python
   # Include timestamp or identifiers in filenames
   from datetime import datetime
   timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
   filename = f"report_{timestamp}.pdf"
   ```

6. **Set Appropriate Timeouts**
   ```python
   # Use reasonable timeouts for downloads
   response = requests.get(file_url, timeout=30)
   ```

7. **Process Files in Parallel** (for multiple files)
   ```python
   from concurrent.futures import ThreadPoolExecutor

   def process_file(file_url):
       # Download and process
       pass

   with ThreadPoolExecutor(max_workers=5) as executor:
       results = list(executor.map(process_file, file_urls))
   ```

8. **Include File Metadata in Responses**
   ```python
   response_message = f"""
   Processing complete!

   📄 Report: {file_data['file_url']}
   📊 File: {file_data['file_name']} ({file_data['mime_type']})
   🔑 UUID: {file_data['file_uuid']}
   """
   ```

---

### Advanced File Management Patterns for AI Agents

This section provides production-ready patterns for AI agents implementing file handling in Lambda functions, based on successful implementations like ProximaAgendaFn.

#### Pattern 1: UUID-Based File Retrieval with Validation

When users upload files to the orchestration session, retrieve them by UUID with comprehensive validation:

```python
import requests
from typing import List
from api.files_requests import files_api_manager

def get_file_from_session(
    self,
    file_uuids: List[str],
    accepted_mime_types: List[str] = None,
    accepted_extensions: List[str] = None,
    verbose: bool = False
) -> str:
    """
    Download a file from the orchestration session by UUID.

    This pattern handles:
    - UUID-based file lookup
    - Dual validation (MIME type + extension)
    - Helpful error messages
    - Presigned URL downloads

    Args:
        file_uuids: List of file UUIDs to look for
        accepted_mime_types: List of acceptable MIME types (e.g., ['application/pdf'])
        accepted_extensions: List of acceptable extensions (e.g., ['.pdf', '.txt'])
        verbose: Enable detailed logging

    Returns:
        Local file path in /tmp/

    Raises:
        ValueError: If file_uuids is empty or no matching files found
        Exception: If download fails
    """
    # Step 1: Validate input
    if not file_uuids:
        error_msg = (
            "Missing required parameter: file_uuids. "
            "You must upload a file to the session and provide its UUID. "
            "Example: file_uuids=['abc-123-def-456']"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if verbose:
        logger.info(f"Fetching files from session with UUIDs: {file_uuids}")

    # Step 2: Get all files from the session
    response = files_api_manager.call(
        "get_all_files_for_session",
        orchestration_session_uuid=self.orchestration_event.orchestration_session_uuid,
        internal_orchestration_session_uuid=self.orchestration_event.internal_orchestration_session_uuid,
        access_token=self.orchestration_event.access_token,
        organization_id=self.orchestration_event.organization.organization_id,
    )

    attachments = response.get("files", [])
    if verbose:
        logger.info(f"Found {len(attachments)} total attachments in session")

    # Step 3: Filter by UUID
    selected_files = [f for f in attachments if f.get("file_uuid") in file_uuids]

    # Step 4: Filter by MIME type or extension (if specified)
    if accepted_mime_types or accepted_extensions:
        filtered_files = []
        for f in selected_files:
            mime_type = f.get("mime_type", "")
            file_name = f.get("file_name", "")

            # Check MIME type
            if accepted_mime_types and mime_type in accepted_mime_types:
                filtered_files.append(f)
                continue

            # Check extension as fallback (handles generic application/octet-stream)
            if accepted_extensions:
                for ext in accepted_extensions:
                    if file_name.lower().endswith(ext.lower()):
                        filtered_files.append(f)
                        break

        selected_files = filtered_files

    # Step 5: Validate we found a file
    if not selected_files:
        available_files = [
            f"{f.get('file_name')} ({f.get('file_uuid')}, {f.get('mime_type')})"
            for f in attachments
        ]
        error_msg = (
            f"No matching files found for UUIDs: {file_uuids}. "
            f"Available files in session: {available_files}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Use the first matching file
    selected_file = selected_files[0]
    file_name = selected_file.get("file_name")
    file_uuid = selected_file.get("file_uuid")
    file_url = selected_file.get("file_url")

    if verbose:
        logger.info(f"Selected file: {file_name} (UUID: {file_uuid})")
        logger.info(f"Downloading from: {file_url[:50]}...")

    # Step 6: Download the file from presigned URL
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to download file: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Step 7: Save to /tmp/ (only writable location in Lambda)
    file_path = f"/tmp/{file_name}"
    with open(file_path, "wb") as f:
        f.write(response.content)

    if verbose:
        file_size_mb = len(response.content) / (1024 * 1024)
        logger.info(f"Downloaded file to {file_path} ({file_size_mb:.2f} MB)")

    return file_path
```

**Usage in manifest.yml:**
```yaml
parameters:
  required:
    - name: file_uuids
      type: array
      description: |
        Array of file UUIDs to process. Upload files to the orchestration
        session first, then pass their UUIDs here.
        Example: ["abc-123-def", "xyz-456-ghi"]
```

**Usage in function logic:**
```python
def process_request(self, orchestration_event: OrchestrationEvent) -> Dict:
    # Extract parameters
    tool_args = orchestration_event.last_message.tool_args
    file_uuids = tool_args.get("file_uuids", [])
    verbose = tool_args.get("verbose", False)

    # Download Excel file with validation
    excel_path = self.get_file_from_session(
        file_uuids=file_uuids,
        accepted_mime_types=[
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
            "application/vnd.ms-excel",  # .xls
            "application/octet-stream"   # Generic binary (common for uploads)
        ],
        accepted_extensions=[".xlsx", ".xls"],
        verbose=verbose
    )

    # Process the file
    # ... your processing logic ...

    # Cleanup
    if os.path.exists(excel_path):
        os.unlink(excel_path)

    return {"status": "success"}
```

#### Pattern 2: Generating and Uploading Result Files

When generating files to return to users, use this pattern with proper cleanup:

```python
import io
import os
import tempfile
from typing import Optional

def generate_and_upload_file(
    self,
    content: str,
    filename: str,
    verbose: bool = False
) -> Optional[str]:
    """
    Generate a text file and upload it to the orchestration session.

    This pattern handles:
    - Temporary file creation in /tmp/
    - BytesIO wrapping for upload
    - Proper cleanup
    - Non-fatal error handling

    Args:
        content: File content as string
        filename: Desired filename (e.g., "report.txt")
        verbose: Enable detailed logging

    Returns:
        File URL if successful, None if upload fails
    """
    try:
        # Step 1: Create temporary file in /tmp/
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=os.path.splitext(filename)[1],
            delete=False
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        if verbose:
            logger.info(f"Generated temporary file: {tmp_path}")

        # Step 2: Read file content into memory
        with open(tmp_path, 'rb') as f:
            file_content = f.read()

        # Step 3: Wrap in BytesIO with filename attribute
        file_obj = io.BytesIO(file_content)
        file_obj.name = filename

        if verbose:
            logger.info(f"Uploading file: {filename} ({len(file_content)} bytes)")

        # Step 4: Upload via files_api_manager
        result = files_api_manager.call(
            "upload_file",
            file=file_obj,
            orchestration_session_uuids=[
                self.orchestration_event.orchestration_session_uuid
            ],
            internal_orchestration_session_uuid=self.orchestration_event.internal_orchestration_session_uuid,
            shared=False,  # Set to True if file should be shared across sessions
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )

        # Step 5: Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        file_url = result.get("file_url")
        if verbose:
            logger.info(f"File uploaded successfully: {file_url}")

        return file_url

    except Exception as e:
        logger.error(f"Error generating/uploading file: {str(e)}")
        # Don't fail the whole request if file upload fails
        return None
```

**For binary files (images, PDFs, etc.):**
```python
def upload_binary_file(self, binary_content: bytes, filename: str) -> Optional[str]:
    """Upload binary content directly without temporary file."""
    try:
        # Wrap binary content directly in BytesIO
        file_obj = io.BytesIO(binary_content)
        file_obj.name = filename

        result = files_api_manager.call(
            "upload_file",
            file=file_obj,
            orchestration_session_uuids=[
                self.orchestration_event.orchestration_session_uuid
            ],
            internal_orchestration_session_uuid=self.orchestration_event.internal_orchestration_session_uuid,
            shared=False,
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )

        return result.get("file_url")

    except Exception as e:
        logger.error(f"Error uploading binary file: {str(e)}")
        return None
```

#### Pattern 3: Excel File Processing with Custom Layers

For specialized file types like Excel, you'll need custom Lambda layers:

**Create custom layer:**
```bash
# Create layer directory structure
mkdir -p layers/excel_layer/python

# Install packages for Lambda environment
pip install --platform manylinux2014_x86_64 \
  --target=layers/excel_layer/python \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  openpyxl python-dateutil
```

**Add to manifest.yml:**
```yaml
layers:
  - foundationMinimal
  - foundationApi
  - custom/excel_layer  # Your custom layer
```

**Usage in function:**
```python
import openpyxl
from datetime import datetime

def process_excel_file(self, excel_path: str) -> dict:
    """Extract data from Excel file using openpyxl."""
    try:
        # Load workbook (data_only=True to get calculated values)
        workbook = openpyxl.load_workbook(excel_path, data_only=True)
        sheet = workbook.active

        # Get headers from first row
        headers = [cell.value for cell in sheet[1]]

        # Extract data rows
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            row_dict = dict(zip(headers, row))
            data.append(row_dict)

        logger.info(f"Extracted {len(data)} rows from Excel")
        return {"rows": data, "count": len(data)}

    except Exception as e:
        logger.error(f"Error processing Excel: {str(e)}")
        raise
```

#### Common File Types and MIME Type Handling

**Important:** Many file uploads use `application/octet-stream` as a generic MIME type. Always include extension-based validation as a fallback.

```python
# Common MIME types and their extensions
FILE_TYPE_MAPPING = {
    "excel": {
        "mime_types": [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
            "application/vnd.ms-excel",  # .xls
            "application/octet-stream"   # Generic fallback
        ],
        "extensions": [".xlsx", ".xls"]
    },
    "pdf": {
        "mime_types": [
            "application/pdf",
            "application/octet-stream"
        ],
        "extensions": [".pdf"]
    },
    "image": {
        "mime_types": [
            "image/jpeg",
            "image/png",
            "image/gif",
            "application/octet-stream"
        ],
        "extensions": [".jpg", ".jpeg", ".png", ".gif"]
    },
    "csv": {
        "mime_types": [
            "text/csv",
            "application/csv",
            "application/octet-stream"
        ],
        "extensions": [".csv"]
    }
}

def validate_file_type(file_info: dict, file_type: str) -> bool:
    """Validate file using both MIME type and extension."""
    mapping = FILE_TYPE_MAPPING.get(file_type)
    if not mapping:
        return False

    mime_type = file_info.get("mime_type", "")
    file_name = file_info.get("file_name", "")

    # Check MIME type
    if mime_type in mapping["mime_types"]:
        return True

    # Fallback to extension check
    for ext in mapping["extensions"]:
        if file_name.lower().endswith(ext.lower()):
            return True

    return False
```

#### Lambda Filesystem Constraints for AI Agents

**Critical constraints to remember:**

1. **/tmp/ is the ONLY writable location** in Lambda
   ```python
   # CORRECT
   file_path = "/tmp/myfile.txt"

   # WRONG - Will fail
   file_path = "./myfile.txt"
   file_path = "/home/user/myfile.txt"
   ```

2. **/tmp/ has 512 MB limit** - Monitor file sizes
   ```python
   file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
   if file_size_mb > 400:  # Leave headroom
       logger.warning(f"Large file: {file_size_mb:.2f} MB")
   ```

3. **/tmp/ persists between warm invocations** - Always cleanup
   ```python
   try:
       file_path = download_file(url)
       # Process file...
   finally:
       if os.path.exists(file_path):
           os.unlink(file_path)
   ```

4. **BytesIO holds entire file in memory** - Consider Lambda memory limits
   ```python
   # For 256 MB Lambda, keep files under 100 MB
   file_obj = io.BytesIO(file_content)  # file_content loaded in RAM
   ```

#### Error Handling Patterns for File Operations

**Input file errors - FAIL FAST:**
```python
if not file_uuids:
    raise ValueError(
        "file_uuids parameter is required. "
        "Upload files to the session and provide their UUIDs."
    )

if not matching_files:
    available = [f"{f['file_name']} ({f['file_uuid']})" for f in all_files]
    raise ValueError(
        f"No files found matching UUIDs: {file_uuids}. "
        f"Available: {available}"
    )
```

**Download errors - FAIL FAST:**
```python
try:
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    raise Exception(f"Failed to download file: {str(e)}")
```

**Upload errors - CAN BE NON-FATAL:**
```python
try:
    file_url = self.generate_and_upload_file(content, filename)
except Exception as e:
    logger.error(f"File upload failed: {str(e)}")
    file_url = None  # Don't crash entire function

return {
    "status": "success",
    "output_file": file_url or "Upload failed"
}
```

#### Performance Optimization for File Operations

**1. Adjust timeout and memory based on file sizes:**
```yaml
# manifest.yml
timeout: 120  # Increase for large files (max 900 seconds)
memorySize: 512  # Increase for file processing (256-10240 MB)

# Guidelines:
# - Small files (< 10 MB): 256 MB memory, 60s timeout
# - Medium files (10-100 MB): 512 MB memory, 120s timeout
# - Large files (100-500 MB): 1024 MB memory, 300s timeout
```

**2. Stream large file downloads:**
```python
def download_large_file(url: str, output_path: str):
    """Stream download to avoid loading entire file in memory."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
```

**3. Process multiple files in parallel:**
```python
from concurrent.futures import ThreadPoolExecutor

def process_multiple_files(self, file_uuids: List[str]) -> List[Dict]:
    """Process multiple files concurrently."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(self._process_single_file, uuid)
            for uuid in file_uuids
        ]
        results = [f.result() for f in futures]
    return results
```

#### Testing Considerations

**Support both production (file_uuids) and local testing (file paths):**
```python
def process_request(self, orchestration_event: OrchestrationEvent) -> Dict:
    tool_args = orchestration_event.last_message.tool_args
    file_uuids = tool_args.get("file_uuids", [])

    # Production: Use file_uuids
    if file_uuids:
        file_path = self._download_from_session(file_uuids)
    # Local testing: Use direct file path
    else:
        file_path = tool_args.get("local_file_path", "/tmp/test.xlsx")
        if not os.path.exists(file_path):
            raise ValueError("Either file_uuids or local_file_path required")

    # ... process file_path ...
```

#### Presigned URL Considerations

File URLs from `get_all_files_for_session` are **time-limited presigned S3 URLs**:

```python
# CORRECT - Download immediately after retrieval
file_url = file_info.get("file_url")
response = requests.get(file_url, timeout=30)

# WRONG - URL may expire
file_urls = [f.get("file_url") for f in files]
time.sleep(300)  # 5 minutes later
response = requests.get(file_urls[0])  # May fail - URL expired
```

---

## Best Practices

### 1. Always Use try/finally for Agent Notification

```python
def lambda_handler(event, context):
    orchestration_event = None
    try:
        # Your code
        pass
    finally:
        if orchestration_event:
            notify_agent_available(orchestration_event, logger, request_id)
```

### 2. Validate Event Type

```python
if orchestration_event.event_type != "function_call":
    return {"statusCode": 200, "body": "Event type not supported"}
```

### 3. Log Request IDs

```python
request_id = context.aws_request_id if context else "unknown"
logger.info(f"Request ID: {request_id}")
```

### 4. Handle Missing Parameters Gracefully

```python
tool_calls = orchestration_event.extra_params.get("tool_calls", [])
if not tool_calls:
    return error_response("No tool calls found", 400)

param = tool_args.get("param")
if not param:
    return error_response("Missing required parameter: param", 400)
```

### 5. Use Structured Logging

```python
logger.info(f"Processing request", extra={
    "organization_id": orchestration_event.organization_id,
    "session_id": orchestration_event.orchestration_session_id,
    "event_type": orchestration_event.event_type
})
```

### 6. Return Consistent Response Format

```python
# Success
return {
    "statusCode": 200,
    "body": json.dumps({"status": "success", "result": data})
}

# Error
return {
    "statusCode": 500,
    "body": json.dumps({"status": "error", "error": error_message})
}
```

### 7. Widget Parameter Order Matters

Widget parameters are resolved by position, so the order in `manifest.yml` must match the order in `resolve_positional()`:

```yaml
# manifest.yml
widget:
  params:
    - name: api_token    # Position 0
    - name: api_secret   # Position 1
```

```python
# handler.py - must match order above
api_token, api_secret = resolver.resolve_positional(widget_data, count=2)
```

### 8. Test Locally with SAM CLI

```bash
# Create test event
cat > test_event.json <<EOF
{
  "body": {
    "orchestration_event": {
      "event_id": "test-123",
      "event_type": "function_call",
      "orchestration_session_id": "session-123",
      "organization_id": "org-123",
      "user_id": "user-123",
      "orchestrator_agent_id": "agent-123",
      "extra_params": {
        "tool_calls": [{
          "name": "test_function",
          "args": {"param1": "value1"}
        }]
      }
    },
    "access_token": "test-token",
    "organization_id": "org-123"
  }
}
EOF

# Invoke locally
sam local invoke -e test_event.json
```

---

## Common Issues and Solutions

### Issue 1: Lambda Layer Dependency Conflicts

**Problem**: Import errors like `Unable to import numpy: you should not try to import numpy from its source directory` or build configuration conflicts.

**Solution**: Avoid heavy scientific packages (pandas, numpy, scipy) in Lambda layers due to platform-specific build issues. Use lightweight alternatives:

```python
# ❌ Avoid: pandas for Excel
import pandas as pd
df.to_excel('output.xlsx')

# ✅ Use: openpyxl for Excel (lightweight)
from openpyxl import Workbook
wb = Workbook()
ws = wb.active
ws.append(['Name', 'Value'])
wb.save('output.xlsx')

# ❌ Avoid: pandas for date handling
import pandas as pd
pd.to_datetime('2024-01-01')

# ✅ Use: python-dateutil (lightweight)
from dateutil.parser import parse
parse('2024-01-01')
```

**Creating Layers Correctly**:

```bash
# 1. Create layer structure
mkdir -p layers/your_layer/python

# 2. Install with correct platform targeting
pip install --platform manylinux2014_x86_64 \
  --target=layers/your_layer/python \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  package_name

# 3. Clean up build files that cause conflicts
find layers/ -name "meson.build" -delete
find layers/ -name "setup.py" -delete
find layers/ -name "pyproject.toml" -delete
find layers/ -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find layers/ -name "*.pyc" -delete
find layers/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 4. Update manifest.yml
# Add: - custom/your_layer
```

**Layer Size Guidelines**:
- ✅ Keep layers < 10 MB for fast cold starts
- ⚠️ Layers 10-50 MB are acceptable but slower
- ❌ Avoid layers > 50 MB (causes deployment issues)

### Issue 2: Widget Parameter Access Patterns

**Problem**: Widget parameters may come in different formats depending on context (production vs test), causing `KeyError` or `AttributeError`.

**Solution**: Always use a fallback pattern to handle both formats:

```python
def get_widget_params(orchestration_event: OrchestrationEvent) -> dict:
    """
    Extract widget parameters supporting both production and test formats.

    Production format (from Secrets Manager):
    {"widget_data": {"widgets": [{"name": "api_token", "value": "secret"}]}}

    Test format (from test files):
    {"widget_data": {"api_token": "test_token"}}
    """
    widget_data = orchestration_event.extra_params.get("widget_data", {})

    # Try nested widgets array first (production)
    widgets = widget_data.get("widgets", [])
    widget_values = {w.get("name"): w.get("value") for w in widgets}

    # Fallback to direct access (test format)
    return {
        "api_token": widget_values.get("api_token") or widget_data.get("api_token"),
        "api_url": widget_values.get("api_url") or widget_data.get("api_url"),
    }

# Usage
params = get_widget_params(orchestration_event)
api_token = params["api_token"]
```

### Issue 2.5: Secret Widget Parameters Not Resolved (UUID Error)

**Problem**: Your function receives a UUID like `550e8400-e29b-41d4-a716-446655440000` instead of the actual secret value.

**Cause**: Secret parameters (`type: secret` in manifest.yml) arrive as UUIDs that reference AWS Secrets Manager. If you access them directly without using `WidgetParamResolver`, you get the UUID instead of the resolved secret.

**Validation Error** (when publishing):
> "Secret widget parameters found but WidgetParamResolver is not used in any code file. Secrets must be resolved using WidgetParamResolver.resolve_positional() to evaluate from AWS Secrets Manager."

**Solution**: Use `WidgetParamResolver` to resolve secret UUIDs:

```python
from api.widget_resolver import WidgetParamResolver

# WRONG - gets UUID, not secret value
widget_data = orchestration_event.extra_params.get("widget_data", {})
api_token = widget_data.get("widget_param_1")  # Returns UUID!

# CORRECT - resolves UUID to actual secret
resolver = WidgetParamResolver(orchestration_event)
widget_data = orchestration_event.extra_params.get("widget_data", {})
api_token, = resolver.resolve_positional(widget_data, count=1)  # Returns actual secret!
```

**Why this happens**: Secrets are stored in AWS Secrets Manager with paths like `chask/org/<org_id>/<param_name>`. The widget system passes the secret's UUID, and `WidgetParamResolver` calls the organizations API to retrieve the actual value securely.

### Issue 3: Missing Layer Dependencies

**Problem**: `ModuleNotFoundError: No module named 'et_xmlfile'` or other sub-dependencies missing.

**Solution**: Always install packages WITH their dependencies (don't use `--no-deps`):

```bash
# ❌ Wrong - missing dependencies
pip install --no-deps openpyxl

# ✅ Correct - includes all dependencies
pip install --platform manylinux2014_x86_64 \
  --target=layers/your_layer/python \
  --python-version 3.12 \
  --only-binary=:all: \
  openpyxl
```

### Issue 4: Large File Generation Timeouts

**Problem**: Lambda timeout when generating large Excel/PDF files or processing many records.

**Solutions**:

```python
# 1. Process in chunks
chunk_size = 1000
for i in range(0, len(records), chunk_size):
    chunk = records[i:i+chunk_size]
    process_chunk(chunk)

# 2. Increase timeout in manifest.yml
# function:
#   timeout: 300  # Up to 900 seconds

# 3. Increase memory (more memory = more CPU)
# function:
#   memory_size: 1024  # Up to 10240 MB
```

### Issue 5: SSL/Certificate Errors with External APIs

**Problem**: `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]` when connecting to external services.

**Solutions**:

```python
import urllib3
import requests

# Option 1: Disable SSL warnings (for testing only)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
response = requests.get(url, verify=False)

# Option 2: Provide certificate bundle
response = requests.get(url, verify='/path/to/ca-bundle.crt')

# Option 3: Use certifi for updated certificates
import certifi
response = requests.get(url, verify=certifi.where())
```

---

## Platform Integrations

These are platform-level services available to Lambda functions. Unlike widget secrets (user-configured), these are shared infrastructure credentials stored in AWS Secrets Manager.

### Browserbase (Browser Automation)

If your function needs to automate browser interactions (web scraping, form filling, testing, etc.), use Browserbase:

**Secret Location:** `chask/browserbase`

**Credentials Available:**
- `BROWSERBASE_API_KEY` - API key for authentication
- `BROWSERBASE_PROJECT_ID` - Project identifier

**Usage:**

```python
from chask_foundation.configs.utils import get_secret
import json

def get_browserbase_credentials():
    """Retrieve Browserbase credentials from AWS Secrets Manager."""
    secret_value = get_secret('chask/browserbase', MODE='PRODUCTION')
    secrets = json.loads(secret_value)

    return {
        'api_key': secrets['BROWSERBASE_API_KEY'],
        'project_id': secrets['BROWSERBASE_PROJECT_ID']
    }

# In your function:
creds = get_browserbase_credentials()
# Use creds['api_key'] and creds['project_id'] with Browserbase SDK
```

**When to Use:**
- Web scraping that requires JavaScript rendering
- Automating form submissions
- Taking screenshots of web pages
- Testing web applications
- Any task requiring a real browser environment

**Note:** The Lambda's IAM role must have `secretsmanager:GetSecretValue` permission for `chask/browserbase`.

---

## Additional Resources

- **Chask Documentation**: https://docs.chask.io
- **chask-foundation Models**: `chask_foundation.models.events`
- **Widget Resolver**: `api.widget_resolver.WidgetParamResolver`
- **Notification Helper**: `api.notify.notify_agent_available`

---

**Last Updated**: 2026-01-17
**Function**: ReadProjectFileFn
**Runtime**: python3.12
