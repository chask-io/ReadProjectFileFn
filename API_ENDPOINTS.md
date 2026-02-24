# Lambda Function API Endpoints Reference

> Comprehensive quick reference for all chask-foundation API endpoints available to Lambda functions.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Authentication & Headers](#authentication--headers)
3. [Orchestrator API (34 endpoints)](#orchestrator-api)
4. [Pipeline API (24 endpoints)](#pipeline-api)
5. [LLM API (3 endpoints)](#llm-api)
6. [Files API (5 endpoints)](#files-api)
7. [Organizations API (12 endpoints)](#organizations-api)
8. [Functions API (5 endpoints)](#functions-api)
9. [WhatsApp API (10 endpoints)](#whatsapp-api)
10. [Email API (2 endpoints)](#email-api)
11. [Outbound API (4 endpoints)](#outbound-api)
12. [Internal Channels API (3 endpoints)](#internal-channels-api)
13. [Channels API (2 endpoints)](#channels-api)
14. [Canvas API (5 endpoints)](#canvas-api)
15. [CRM API (1 endpoint)](#crm-api)
16. [Agents API (1 endpoint)](#agents-api)
17. [Analysts API (1 endpoint)](#analysts-api)
18. [Fintoc API (1 endpoint)](#fintoc-api)
19. [Firebase API (1 endpoint)](#firebase-api)
20. [Error Handling](#error-handling)
21. [Common Patterns](#common-patterns)

## Getting Started

All API endpoints are accessed through domain-specific API managers from chask-foundation.

### Basic Usage Pattern

```python
from chask_foundation.api.orchestrator_requests import orchestrator_api_manager
from chask_foundation.models.events import OrchestrationEvent

def example_function(orchestration_event: OrchestrationEvent):
    response = orchestrator_api_manager.call(
        "endpoint_name",
        param1=value1,
        param2=value2,
        access_token=orchestration_event.access_token,
        organization_id=orchestration_event.organization.organization_id
    )

    if response.get("status_code") not in (200, 201):
        error = response.get("error", "Unknown error")
        raise Exception(f"API call failed: {error}")

    return response
```

### Required for All Calls
- `access_token` - From `orchestration_event.access_token`
- `organization_id` - From `orchestration_event.organization.organization_id`

## Authentication & Headers

All API calls automatically include:
- `Authorization: Bearer {access_token}`
- `Organization-ID: {organization_id}`

These are injected by the ApiManager from kwargs.

---

## Orchestrator API

**Import:** `from chask_foundation.api.orchestrator_requests import orchestrator_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/orchestrator`

### Essential Endpoints (Priority 1)

#### evolve_event
Create a child event linked to parent for proper event tracking.

**Parameters:**
- `parent_event_uuid` (str) - Parent event UUID
- `event_type` (str) - Event type (e.g., "function_call_response")
- `source` (str, optional) - Event source (default: "orchestrator")
- `target` (str, optional) - Event target (default: "agent_manager")
- `prompt` (str, optional) - Event message/content
- `extra_params` (dict, optional) - Additional event data
- `short_summary` (str, optional) - Brief summary for UI
- `color` (str, optional) - UI color indicator
- `caller_info` (str, optional) - Auto-detected if not provided

**Returns:** `{uuid, extra_params, status_code}`

**Example:**
```python
response = orchestrator_api_manager.call(
    "evolve_event",
    parent_event_uuid=str(orchestration_event.event_id),
    event_type="function_call_response",
    source="agent",
    target="orchestrator",
    prompt="Function completed successfully",
    extra_params={"tool_call_id": "call_123", "is_error": False},
    access_token=orchestration_event.access_token,
    organization_id=orchestration_event.organization.organization_id
)
evolved_uuid = response.get("uuid")
```

#### forward_oe_to_kafka
Send orchestration event to Kafka topic.

**Parameters:**
- `orchestration_event` (dict) - Event as dict (from `model_dump()`)
- `topic` (str) - Kafka topic ("orchestrator", "agent_manager", etc.)
- `caller_info` (str, optional) - Auto-detected if not provided

**Returns:** `{status_code}`

**Example:**
```python
orchestrator_api_manager.call(
    "forward_oe_to_kafka",
    orchestration_event=response_event.model_dump(),
    topic="orchestrator",
    access_token=response_event.access_token,
    organization_id=response_event.organization.organization_id
)
```

#### record_event_fingerprint
Record a fingerprint step in the event's path registry for observability.

**Parameters:**
- `event_uuid` (str) - Event UUID
- `step_type` (str) - Type: "creation", "kafka_routing", "handler_processing", "evolution", "celery_task", "lambda_invocation", "internal_tool"
- `step_name` (str) - Step identifier
- `service_name` (str) - Service name
- `handler_name` (str, optional) - Handler name
- `step_data` (dict, optional) - Additional step data

**Returns:** `{status_code}`

**Example:**
```python
orchestrator_api_manager.call(
    "record_event_fingerprint",
    event_uuid=str(event.event_id),
    step_type="lambda_invocation",
    step_name="process_payment",
    service_name="lambda",
    handler_name="lambda_function.handler",
    step_data={"duration_ms": 1500},
    access_token=event.access_token,
    organization_id=event.organization.organization_id
)
```

### Session & Event Management

#### get_single_orchestration_session
Fetch complete orchestration session details.

**Parameters:**
- `orchestration_session_id` (str) - Session ID (UUID or numeric)

**Returns:** `{session_data, status_code}`

**Example:**
```python
response = orchestrator_api_manager.call(
    "get_single_orchestration_session",
    orchestration_session_id=session_uuid,
    access_token=token,
    organization_id=org_id
)
```

#### get_single_orchestration_event
Fetch complete orchestration event details.

**Parameters:**
- `orchestration_event_uuid` (str) - Event UUID

**Returns:** `{event_data, status_code}`

#### get_orchestration_events
Get all events for an orchestration session.

**Parameters:**
- `orchestration_session_id` (str) - Session ID

**Returns:** `{events, status_code}`

#### get_orchestration_session_user_data
Retrieve user data stored for a session.

**Parameters:**
- `orchestration_session_uuid` (str, optional) - External session UUID
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID

**Returns:** `{user_data, status_code}`

#### save_orchestration_session_user_data
Save user data for a session.

**Parameters:**
- `orchestration_session_id` (str) - Session ID
- `user_data` (dict) - User data to save

**Returns:** `{status_code}`

**Example:**
```python
orchestrator_api_manager.call(
    "save_orchestration_session_user_data",
    orchestration_session_id=session_uuid,
    user_data={"customer_preferences": {"language": "es"}},
    access_token=token,
    organization_id=org_id
)
```

#### create_orchestration_session
Create a new orchestration session.

**Parameters:**
- `organization_customer_id` (str) - Customer UUID
- `pipeline_id` (str) - Pipeline UUID

**Returns:** `{session_uuid, status_code}`

#### create_orchestration_event
Create a new orchestration event.

**Parameters:**
- `**payload` - Event data (event_type, source, target, prompt, etc.)

**Returns:** `{event_uuid, status_code}`

#### assign_orchestration_session_to_event
Link a session to an event.

**Parameters:**
- `orchestration_session_id` (str) - Session ID
- `orchestration_event_id` (str) - Event ID

**Returns:** `{status_code}`

#### change_orchestration_session_status
Update session status.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `status` (str) - New status

**Returns:** `{status_code}`

### Pipeline Integration

#### get_pipelines_from_oe
Get pipelines associated with an orchestration session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{pipelines, status_code}`

#### assign_pipeline_to_orchestration
Link a pipeline to an orchestration session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `cognito_id` (str) - User Cognito ID
- `pipeline_id` (str) - Pipeline UUID

**Returns:** `{status_code}`

#### get_orchestration_session_channels
Get communication channels for a session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{channels, status_code}`

#### assign_wsp_channel_to_orchestration
Assign WhatsApp channel to session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `channel_id` (str) - Channel ID

**Returns:** `{status_code}`

### Plan Management

#### get_orchestration_plan
Get orchestration plan for a session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{plan, status_code}`

#### save_orchestration_plan
Save an orchestration plan.

**Parameters:**
- `orchestration_plan` (dict) - Plan data

**Returns:** `{plan_uuid, status_code}`

#### create_plan_for_os
Create a detailed orchestration plan.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `internal_orchestration_session_uuid` (str) - Internal session UUID
- `brief_summary` (str) - Plan summary
- `clear_objective` (str) - Plan objective
- `critical_data` (str) - Critical data
- `n_steps` (int) - Number of steps
- `task_per_step` (int) - Tasks per step
- `current_step` (int) - Current step

**Returns:** `{plan_uuid, status_code}`

#### update_plan_current_step
Update current step in plan.

**Parameters:**
- `orchestration_plan_uuid` (str) - Plan UUID
- `current_step` (int) - New current step

**Returns:** `{status_code}`

#### toggle_orchestration_plan
Pause/resume orchestration plan.

**Parameters:**
- `orchestration_plan_uuid` (str) - Plan UUID
- `action` (str) - "pause" or "resume"
- `reason` (str) - Reason for action

**Returns:** `{status_code}`

#### update_orchestration_plan_task
Update a task in the plan.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `task_number` (int) - Task number
- `status` (str) - New status
- `comment` (str) - Status comment
- `current_step` (int) - Current step

**Returns:** `{status_code}`

#### update_orchestration_task
Update orchestration task status.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `task_number` (int) - Task number
- `status` (str) - New status
- `comment` (str) - Status comment

**Returns:** `{status_code}`

### Internal Orchestration Sessions

#### get_internal_orchestration_events
Get events for internal orchestration session.

**Parameters:**
- `internal_orchestration_session_id` (str) - Internal session ID

**Returns:** `{events, status_code}`

#### get_internal_orchestration_session_user_data
Get user data for internal session.

**Parameters:**
- `internal_orchestration_session_uuid` (str) - Internal session UUID

**Returns:** `{user_data, status_code}`

#### get_active_requirement_for_os
Get active requirement for orchestration session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{requirement, status_code}`

### Assistants Integration

#### get_orchestrator_events
Get orchestrator events for assistants UI.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{events, status_code}`

#### get_orchestration_data_from_uuid
Get complete orchestration data.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{orchestration_data, status_code}`

#### save_orchestrator_event
Save any type of orchestrator event.

**Parameters:**
- `**event_payload` - Event data

**Returns:** `{event_uuid, status_code}`

#### save_orchestrator_ai_message
Save AI message for assistants.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `ai_message_data` (dict) - AI message data

**Returns:** `{message_uuid, status_code}`

#### update_oe_summary_color
Update event summary and color for UI.

**Parameters:**
- `orchestration_event_uuid` (str) - Event UUID
- `summary` (str) - Event summary
- `color` (str) - Color code

**Returns:** `{status_code}`

### Claude Conversations

#### retrieve_claude_conversation
Retrieve Claude conversation for a node.

**Parameters:**
- `node_id` (int|str) - Pipeline node ID
- `orchestration_session_uuid` (str, optional) - Session UUID (provide one)
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID (provide one)
- `include_conversation_data` (bool, optional) - Include full conversation (default: False)

**Returns:** `{conversation_metadata, conversation_data?, status_code}`

**Example:**
```python
response = orchestrator_api_manager.call(
    "retrieve_claude_conversation",
    node_id=42,
    orchestration_session_uuid=session_uuid,
    include_conversation_data=True,
    access_token=token,
    organization_id=org_id
)
if response.get("status_code") == 200:
    # Conversation exists, can resume
    conversation = response.get("conversation_data")
```

### Utilities

#### validate_rut
Validate Chilean RUT format.

**Parameters:**
- `rut` (str) - RUT to validate

**Returns:** `{valid, status_code}`

#### user_exist_and_belongs_to_organization
Check if user exists in organization.

**Parameters:**
- `user_email` (str) - User email

**Returns:** `{exists, user_data?, status_code}`

---

## Pipeline API

**Import (v2):** `from chask_foundation.api.pipeline_requests import pipeline_api_manager`

**Import (legacy):** `from chask_foundation.api.pipeline_requests import legacy_api_manager`

**Base URL (v2):** `https://{BASE_DOMAIN}/api/v2/pipelines`

**Base URL (legacy):** `https://{BASE_DOMAIN}`

### Modern API (v2) - Recommended

#### create_pipeline
Create an empty pipeline board.

**Parameters:**
- `is_template` (bool) - Whether pipeline is a template
- `organization_customer_uuid` (str) - Customer UUID

**Returns:** `{pipeline_id, status_code}`

**Example:**
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

#### assign_pipeline_to_orchestration
Attach pipeline to orchestration session.

**Parameters:**
- `pipeline_id` (str) - Pipeline UUID
- `orchestration_session_uuid` (str, optional) - Session UUID (provide one)
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID (provide one)

**Returns:** `{status_code}`

#### get_pipeline
Fetch single pipeline by ID.

**Parameters:**
- `pipeline_id` (str) - Pipeline UUID

**Returns:** `{pipeline_data, status_code}`

#### get_node
Fetch single node with widget data.

**Parameters:**
- `node_id` (str) - Node ID

**Returns:** `{node_data, widget_data, status_code}`

**Example:**
```python
response = pipeline_api_manager.call(
    "get_node",
    node_id="42",
    access_token=token,
    organization_id=org_id
)
widget_data = response.get("widget_data", {})
```

#### update_node_widget
Post lambda execution results to node widget.

**Parameters:**
- `node_id` (str) - Node ID
- `lambda_result` (dict) - Lambda execution results (free-form JSON)
- `lambda_name` (str, optional) - Lambda function name (default: "unknown")
- `execution_context` (dict, optional) - Additional execution context

**Returns:** `{widget_data, result_count, status_code}`

**Example:**
```python
pipeline_api_manager.call(
    "update_node_widget",
    node_id="42",
    lambda_result={
        "success": True,
        "payment_id": "pay_123",
        "amount": 1500.00
    },
    lambda_name="process_payment",
    execution_context={"duration_ms": 1200},
    access_token=token,
    organization_id=org_id
)
```

#### operate_node
Transition to specific node in pipeline.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `node_id` (str) - Target node ID

**Returns:** `{status_code}`

#### update_node_status
Update node status.

**Parameters:**
- `node_id` (str) - Node ID
- `status` (str) - New status

**Returns:** `{status_code}`

#### assign_node
Assign human to specific node.

**Parameters:**
- `node_id` (str) - Node ID
- `pipeline_id` (str) - Pipeline UUID

**Returns:** `{status_code}`

#### get_pipeline_status
Get current pipeline status for session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{status, current_node, status_code}`

#### validate_pipeline
Validate pipeline exists and belongs to organization.

**Parameters:**
- `pipeline_uuid` (str) - Pipeline UUID

**Returns:** `{valid, status_code}`

#### save_pipeline
Save pipeline structure with nodes and edges. Used for creating new nodes.

**Parameters:**
- `pipeline_id` (str) - Pipeline ID
- `pipeline` (dict) - Pipeline structure containing nodes and edges. Each node must have:
  - `temp_id` (str) - Temporary ID for new nodes (mapped to real ID on save)
  - `title` (str) - Node title
  - `description` (str) - Node description
  - `position` (dict) - Position with x and y coordinates
  - `node_type` (str) - Type of node (e.g., "custom")
- `pipeline_title` (str, optional) - Pipeline title
- `pipeline_description` (str, optional) - Pipeline description
- `pipeline_prompt` (str, optional) - Pipeline prompt
- `template_used` (str, optional) - Template identifier if created from template

**Returns:** `{pipeline, temp_id_mapping, status_code}`

**Example:**
```python
from chask_foundation.api.pipeline_requests import pipeline_api_manager

response = pipeline_api_manager.call(
    "save_pipeline",
    pipeline_id="123",
    pipeline={
        "nodes": [
            {
                "temp_id": "temp-1",
                "title": "New Step",
                "description": "First step",
                "position": {"x": 100, "y": 100},
                "node_type": "custom"
            }
        ],
        "edges": []
    },
    pipeline_title="My Pipeline",
    access_token=token,
    organization_id=org_id
)
# temp_id_mapping maps "temp-1" to real node ID
```

#### update_pipeline
Update existing pipeline with versioning. Archives nodes/edges not included in payload.

**Parameters:**
- `pipeline` (dict) - Pipeline structure with required fields:
  - `id` (str) - Pipeline ID
  - `nodes` (list) - List of node dicts, each requiring:
    - `id` (str) - Existing node ID
    - `title` (str) - Node title (required)
    - `description` (str) - Node description (required)
    - `position` (dict) - Position with x and y coordinates (required)
  - `edges` (list) - List of edge dicts with source and target

**Returns:** `{pipeline, status_code}`

**Example:**
```python
# Get current pipeline, modify, then update
current = pipeline_api_manager.call("get_pipeline", pipeline_id="123", ...)
nodes = current["nodes"]
# Remove node and add edge
nodes = [n for n in nodes if n["id"] != "node-to-delete"]
edges = current["edges"]
edges.append({"source": "node-1", "target": "node-3"})

pipeline_api_manager.call(
    "update_pipeline",
    pipeline={"id": "123", "nodes": nodes, "edges": edges},
    access_token=token,
    organization_id=org_id
)
```

#### create_node
Create a new node in a pipeline.

**Parameters:**
- `pipeline_id` (str) - Pipeline ID to add the node to
- `title` (str) - Node title
- `positions` (dict) - Position with x and y coordinates
- `description` (str, optional) - Node description
- `node_type` (str, optional) - Type of node (default: "custom")
- `functions` (list, optional) - List of functions attached to the node
- `analyst_uuid` (str, optional) - Analyst UUID to assign
- `widget_data` (dict, optional) - Widget configuration data

**Returns:** `{node, status_code}`

**Example:**
```python
response = pipeline_api_manager.call(
    "create_node",
    pipeline_id="123",
    title="Review Step",
    positions={"x": 300, "y": 200},
    description="Human review required",
    node_type="custom",
    access_token=token,
    organization_id=org_id
)
new_node_id = response["node"]["id"]
```

#### update_node
Update node properties (title, description, position, functions, analyst).

**Parameters:**
- `node_id` (str) - Node ID to update
- `pipeline_id` (str) - Pipeline ID containing the node
- `user_id` (str) - User ID making the update (required)
- `title` (str, optional) - New title
- `description` (str, optional) - New description
- `position` (dict, optional) - New position with x and y
- `node_type` (str, optional) - New node type
- `functions` (list, optional) - New list of functions
- `analyst` (dict, optional) - Analyst dict to assign
- `widget_data` (dict, optional) - Widget configuration data

**Returns:** `{node, status_code}`

**Example:**
```python
pipeline_api_manager.call(
    "update_node",
    node_id="456",
    pipeline_id="123",
    user_id="user-789",
    title="Updated Review Step",
    description="Updated description",
    position={"x": 400, "y": 300},
    access_token=token,
    organization_id=org_id
)
```

#### edit_title_description
Quick update for pipeline title and description only.

**Parameters:**
- `pipeline_id` (str) - Pipeline ID to update
- `title` (str, optional) - New title
- `description` (str, optional) - New description

**Returns:** `{status_code}`

**Example:**
```python
pipeline_api_manager.call(
    "edit_title_description",
    pipeline_id="123",
    title="New Pipeline Title",
    description="Updated pipeline description",
    access_token=token,
    organization_id=org_id
)
```

#### update_assignees
Update the list of users who can be assigned to a node.

**Parameters:**
- `node_id` (str) - Node ID to update
- `assignees` (list) - List of user email addresses who can be assigned

**Returns:** `{status_code}`

**Example:**
```python
pipeline_api_manager.call(
    "update_assignees",
    node_id="456",
    assignees=["user1@example.com", "user2@example.com"],
    access_token=token,
    organization_id=org_id
)
```

### Legacy API - For Compatibility Only

#### create_new_pipeline_board
Create pipeline through pre-v2 endpoint.

**Parameters:**
- `organization_client_uuid` (str) - Client UUID
- `cognito_id` (str) - User Cognito ID
- `organization_id` (str) - Organization ID
- `initiated_from` (str, optional) - Initiation source
- `conversation_id` (str, optional) - Conversation ID

**Returns:** `{pipeline_id, status_code}`

#### assign_pipeline_to_orchestration_legacy
Legacy pipeline assignment.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID
- `cognito_id` (str) - User Cognito ID
- `pipeline_id` (str) - Pipeline UUID

**Returns:** `{status_code}`

#### get_pipeline_conversation_data
Get pipeline conversation data.

**Parameters:**
- `pipeline_conversation_id` (str) - Conversation ID
- `cognito_id` (str) - User Cognito ID

**Returns:** `{conversation_data, status_code}`

#### get_pipeline_from_conversation
Get pipeline from conversation ID.

**Parameters:**
- `pipeline_conversation_id` (str) - Conversation ID
- `cognito_id` (str) - User Cognito ID

**Returns:** `{pipeline_data, status_code}`

#### delegate_pipeline_board
Mark pipeline as delegated.

**Parameters:**
- `pipeline_id` (str) - Pipeline UUID
- `cognito_id` (str) - User Cognito ID
- `organization_id` (str) - Organization ID
- `delegated` (bool, optional) - Delegation status (default: True)

**Returns:** `{status_code}`

#### share_pipeline
Share pipeline with user by email.

**Parameters:**
- `pipeline_id` (str) - Pipeline UUID
- `cognito_id` (str) - User Cognito ID
- `email` (str) - Recipient email

**Returns:** `{status_code}`

#### save_message
Save message to node conversation.

**Parameters:**
- `node_conversation_id` (str) - Node conversation ID
- `content` (str) - Message content
- `is_user` (bool) - Is user message
- `is_bot` (bool) - Is bot message
- `is_system` (bool) - Is system message
- `is_hidden` (bool) - Is hidden message
- `cognito_id` (str) - User Cognito ID
- `email` (str) - User email

**Returns:** `{message_id, status_code}`

#### get_node_conversation_from_node_id
Get node conversation from node ID.

**Parameters:**
- `node_id` (str) - Node ID
- `cognito_id` (str) - User Cognito ID

**Returns:** `{conversation_data, status_code}`

---

## LLM API

**Import:** `from chask_foundation.api.llm_requests import llm_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/llm`

#### log_call
Log a single LLM call for cost tracking and observability.

**Parameters:**
- `provider` (str) - LLM provider ("openai", "anthropic", "azure", "other")
- `model` (str) - Model identifier
- `call_type` (str) - Call type ("chat", "completion", "embedding", "function_call")
- `latency_ms` (int) - Request latency in milliseconds
- `success` (bool, optional) - Whether call succeeded (default: True)
- `request_messages` (list, optional) - Message dictionaries
- `request_functions` (list, optional) - Function definitions
- `request_parameters` (dict, optional) - Additional request parameters
- `response_content` (str, optional) - Text response
- `response_function_call` (dict, optional) - Function call details
- `response_raw` (dict, optional) - Raw response object
- `input_tokens` (int, optional) - Input tokens (default: 0)
- `output_tokens` (int, optional) - Output tokens (default: 0)
- `total_tokens` (int, optional) - Total tokens (auto-calculated if not provided)
- `embedding_dimensions` (int, optional) - Embedding dimensions
- `time_to_first_token_ms` (int, optional) - Time to first token
- `error_message` (str, optional) - Error message if failed
- `error_code` (str, optional) - Error code if failed
- `request_id` (str, optional) - Provider request ID
- `caller_function` (str, optional) - Calling function name
- `orchestration_session_uuid` (str, optional) - Session to link
- `internal_orchestration_session_uuid` (str, optional) - Internal session
- `metadata` (dict, optional) - Additional metadata
- `client_call_id` (str, optional) - Client ID for idempotency

**Returns:** `{uuid, status_code}`

**Example:**
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
    total_cost=0.015,
    success=True,
    caller_function="process_payment",
    orchestration_session_uuid=session_uuid,
    access_token=token,
    organization_id=org_id
)
```

#### batch_log_calls
Batch log multiple LLM calls for efficiency.

**Parameters:**
- `calls` (list) - List of call dictionaries (each with fields from `log_call`)

**Returns:** `{created_count, failed_count, uuids, status_code}`

**Example:**
```python
llm_api_manager.call(
    "batch_log_calls",
    calls=[
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "call_type": "chat",
            "latency_ms": 1200,
            "input_tokens": 800,
            "output_tokens": 400
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "call_type": "chat",
            "latency_ms": 900,
            "input_tokens": 500,
            "output_tokens": 250
        }
    ],
    access_token=token,
    organization_id=org_id
)
```

#### get_contexts_for_function
Get all LLM contexts assigned to a Lambda function.

**Parameters:**
- `function_uuid` (str) - Lambda function UUID

**Returns:** `{function_info, contexts, status_code}`

**Example:**
```python
response = llm_api_manager.call(
    "get_contexts_for_function",
    function_uuid="func-uuid-123",
    access_token=token,
    organization_id=org_id
)
contexts = response.get("contexts", [])
```

---

## Files API

**Import:** `from chask_foundation.api.files_requests import files_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/files`

#### upload_file
Upload a file to the system.

**Parameters:**
- `file` (file object) - File to upload
- `orchestration_session_uuids` (list|str, optional) - Session UUIDs to link
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID
- `shared` (bool, optional) - Whether file is shared (default: False)

**Returns:** `{file_uuid, file_url, status_code}`

**Example:**
```python
from chask_foundation.api.files_requests import files_api_manager

with open("report.pdf", "rb") as f:
    response = files_api_manager.call(
        "upload_file",
        file=f,
        orchestration_session_uuids=[session_uuid],
        shared=False,
        access_token=token,
        organization_id=org_id
    )
file_uuid = response.get("file_uuid")
file_url = response.get("file_url")
```

#### read_single_pdf_text
Extract text from PDF file.

**Parameters:**
- `attachment_uuid` (str) - Attachment UUID
- `origin` (str) - Origin identifier

**Returns:** `{text, pages, status_code}`

**Example:**
```python
response = files_api_manager.call(
    "read_single_pdf_text",
    attachment_uuid="att-uuid-123",
    origin="email",
    access_token=token,
    organization_id=org_id
)
pdf_text = response.get("text")
```

#### get_all_files_for_session
Get all files linked to a session.

**Parameters:**
- `orchestration_session_uuid` (str, optional) - Session UUID (provide one)
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID (provide one)

**Returns:** `{files, status_code}`

#### get_email_attachements_for_session
Get email attachments for a session.

**Parameters:**
- `orchestration_session_uuid` (str, optional) - Session UUID (provide one)
- `internal_orchestration_session_uuid` (str, optional) - Internal session UUID (provide one)

**Returns:** `{attachments, status_code}`

**Example:**
```python
response = files_api_manager.call(
    "get_email_attachements_for_session",
    orchestration_session_uuid=session_uuid,
    access_token=token,
    organization_id=org_id
)
attachments = response.get("attachments", [])
for att in attachments:
    print(f"File: {att['filename']}, UUID: {att['uuid']}")
```

---

## Organizations API

**Import:** `from chask_foundation.api.organizations_requests import organizations_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/organizations`

### Customer Management

#### get_single_organization_customer
Retrieve customer details.

**Parameters:**
- `customer_uuid` (str) - Customer UUID

**Returns:** `{customer_data, status_code}`

**Example:**
```python
from chask_foundation.api.organizations_requests import organizations_api_manager

response = organizations_api_manager.call(
    "get_single_organization_customer",
    customer_uuid="cust-uuid-123",
    access_token=token,
    organization_id=org_id
)
customer = response.get("customer_data")
```

#### create_target_customer_group
Create customer group for bulk campaigns.

**Parameters:**
- `customer_uuids` (list) - List of customer UUIDs
- `name` (str, optional) - Group name (auto-generated if not provided)
- `description` (str, optional) - Group description
- `organization_uuid` (str, optional) - Organization UUID

**Returns:** `{group_uuid, status_code}`

**Example:**
```python
response = organizations_api_manager.call(
    "create_target_customer_group",
    customer_uuids=["uuid1", "uuid2", "uuid3"],
    name="VIP Customers",
    description="High-value customer segment",
    access_token=token,
    organization_id=org_id
)
group_uuid = response.get("group_uuid")
```

#### get_single_target_customer_group_from_ios
Get customer group from internal orchestration session.

**Parameters:**
- `internal_orchestration_session_uuid` (str) - Internal session UUID

**Returns:** `{group_data, status_code}`

### Function Management

#### get_organization_functions
Retrieve all organization functions with parameter requirements.

**Parameters:** None

**Returns:** `{functions, status_code}`

**Example:**
```python
response = organizations_api_manager.call(
    "get_organization_functions",
    access_token=token,
    organization_id=org_id
)
functions = response.get("functions", [])
```

#### get_selected_agent_function
Get currently selected agent function for channel.

**Parameters:**
- `channel_uuid` (str, optional) - Channel UUID

**Returns:** `{function_data, status_code}`

**Example:**
```python
response = organizations_api_manager.call(
    "get_selected_agent_function",
    channel_uuid="ch-uuid-123",
    access_token=token,
    organization_id=org_id
)
agent_function = response.get("function_data")
```

#### get_selected_operator_function
Get currently selected operator function.

**Parameters:** None (uses 'orquestador' channel)

**Returns:** `{function_data, status_code}`

#### clear_operator_function
Clear operator function assignment (revert to template).

**Parameters:** None

**Returns:** `{status_code}`

#### get_agent_equipped_functions
Get functions equipped to an agent.

**Parameters:**
- `agent_function_uuid` (str) - Agent function UUID

**Returns:** `{functions, status_code}`

#### get_operator_equipped_functions
Get functions equipped to an operator.

**Parameters:**
- `operator_function_uuid` (str) - Operator function UUID

**Returns:** `{functions, status_code}`

#### equip_function_to_agent
Equip a function to an agent.

**Parameters:**
- `agent_function_uuid` (str) - Agent function UUID
- `function_uuid` (str) - Function UUID to equip

**Returns:** `{status_code}`

**Example:**
```python
organizations_api_manager.call(
    "equip_function_to_agent",
    agent_function_uuid="agent-uuid",
    function_uuid="func-uuid",
    access_token=token,
    organization_id=org_id
)
```

#### unequip_function_from_agent
Unequip a function from an agent.

**Parameters:**
- `agent_function_uuid` (str) - Agent function UUID
- `function_uuid` (str) - Function UUID to unequip

**Returns:** `{status_code}`

### Secrets Management

#### retrieve_secret
Retrieve a secret value by UUID.

**Parameters:**
- `uuid` (str) - Secret UUID

**Returns:** `{value, status_code}`

**Example:**
```python
response = organizations_api_manager.call(
    "retrieve_secret",
    uuid="secret-uuid-123",
    access_token=token,
    organization_id=org_id
)
secret_value = response.get("value")
```

---

## Functions API

**Import:** `from chask_foundation.api.function_requests import function_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/organizations`

#### search_functions
Search for Lambda functions with fuzzy matching.

**Parameters:**
- `search` (str, optional) - Search term (matches name, logical_id, description)
- `assigned_only` (bool, optional) - Show only assigned functions (default: False)
- `fuzzy_threshold` (int, optional) - Fuzzy match threshold 0-100 (default: 70)

**Returns:** `{functions, status_code}`

**Example:**
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
for func in functions:
    print(f"{func['name']}: {func['description']}")
```

#### get_function_by_uuid
Get specific Lambda function metadata.

**Parameters:**
- `function_uuid` (str) - Function UUID

**Returns:** `{name, description, parameters, logical_id, status_code}`

**Example:**
```python
response = function_api_manager.call(
    "get_function_by_uuid",
    function_uuid="func-uuid-123",
    access_token=token,
    organization_id=org_id
)
function_name = response.get("name")
parameters = response.get("parameters", {})
```

### Placeholder Lambda Functions

#### create_placeholder_function
Create a placeholder Lambda function for pipeline planning.

**Parameters:**
- `logical_name` (str) - Clean function name (e.g., "GetInventory", "ProcessOrder")
- `description` (str, optional) - Human-readable description
- `required_parameters` (dict, optional) - Required parameter definitions
- `optional_parameters` (dict, optional) - Optional parameter definitions

**Returns:** `{function_uuid, logical_name, deployment_state, is_placeholder, status_code}`

**Example:**
```python
response = function_api_manager.call(
    "create_placeholder_function",
    logical_name="ProcessPayment",
    description="Process customer payment transactions",
    required_parameters={
        "amount": {
            "type": "number",
            "description": "Payment amount"
        },
        "customer_id": {
            "type": "string",
            "description": "Customer UUID"
        }
    },
    optional_parameters={
        "currency": {
            "type": "string",
            "description": "Payment currency",
            "default": "USD"
        }
    },
    access_token=token,
    organization_id=org_id
)

function_uuid = response.get("function_uuid")
# is_placeholder: true, deployment_state: "not_deployed"

# Use function_uuid to attach to pipeline nodes
# Later deploy real Lambda code without changing references
```

#### publish_to_placeholder
Deploy a real Lambda to a placeholder function.

**Parameters:**
- `function_uuid` (str) - Placeholder function UUID
- `github_repo_url` (str) - GitHub repository URL
- `git_ref` (str, optional) - Branch/tag/commit (default: "main")
- `manifest_path` (str, optional) - Manifest path (default: "manifest.yml")

**Returns:** `{job_id, status, status_code}`

**Example:**
```python
response = function_api_manager.call(
    "publish_to_placeholder",
    function_uuid="func-uuid-123",
    github_repo_url="https://github.com/org/payment-lambda",
    git_ref="main",
    access_token=token,
    organization_id=org_id
)

job_id = response.get("job_id")
# Poll deployment status using get_deployment_status
```

#### get_deployment_status
Check deployment status for a function deployment job.

**Parameters:**
- `job_id` (str) - Deployment job ID

**Returns:** `{status, progress, logs, function_arn, function_name, status_code}`

**Example:**
```python
response = function_api_manager.call(
    "get_deployment_status",
    job_id="job-123",
    access_token=token,
    organization_id=org_id
)

status = response.get("status")  # "pending", "running", "completed", "failed"
progress = response.get("progress")  # 0-100

if status == "completed":
    function_arn = response.get("function_arn")
    print(f"Deployment complete: {function_arn}")
elif status == "failed":
    logs = response.get("logs", [])
    print(f"Deployment failed: {logs[-1]}")
```

**Placeholder Workflow:**
```python
# 1. Create placeholder during pipeline design
placeholder = function_api_manager.call(
    "create_placeholder_function",
    logical_name="GetInventory",
    description="Fetches inventory from ERP",
    access_token=token,
    organization_id=org_id
)
func_uuid = placeholder["function_uuid"]

# 2. Use func_uuid in pipeline node configuration
# (node references the UUID)

# 3. Later: deploy real Lambda to placeholder
deploy = function_api_manager.call(
    "publish_to_placeholder",
    function_uuid=func_uuid,
    github_repo_url="https://github.com/org/inventory-lambda",
    access_token=token,
    organization_id=org_id
)

# 4. Monitor deployment
import time
while True:
    status = function_api_manager.call(
        "get_deployment_status",
        job_id=deploy["job_id"],
        access_token=token,
        organization_id=org_id
    )
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Pipeline nodes now use deployed Lambda (UUID unchanged)
```

---

## WhatsApp API

**Import:** `from chask_foundation.api.internal_whatsapp_requests import internal_whatsapp_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/channels/whatsapp`

### Messaging

#### save_sent_message
Save a bot-sent WhatsApp message.

**Parameters:**
- `message_id` (str) - WhatsApp message ID
- `sender_id` (str) - Sender phone number
- `message_content` (str) - Message content
- `whatsapp_conversation_id` (str) - Conversation ID
- `raw_data` (dict, optional) - Raw WhatsApp data
- `sender_name` (str, optional) - Sender name

**Returns:** `{message_uuid, status_code}`

**Example:**
```python
from chask_foundation.api.internal_whatsapp_requests import internal_whatsapp_api_manager

internal_whatsapp_api_manager.call(
    "save_sent_message",
    message_id="wamid.123",
    sender_id="+56912345678",
    message_content="Payment processed successfully",
    whatsapp_conversation_id="conv-uuid",
    access_token=token,
    organization_id=org_id
)
```

#### retrieve_messages
Get messages from WhatsApp conversation.

**Parameters:**
- `whatsapp_conversation_id` (str) - Conversation ID

**Returns:** `{messages, status_code}`

#### save_orchestration_message
Save orchestration message to WhatsApp.

**Parameters:**
- `whatsapp_conversation_id` (str) - Conversation ID
- `content` (str) - Message content
- `cognito_id` (str) - User Cognito ID

**Returns:** `{message_uuid, status_code}`

### Conversations

#### get_bot_whatsapp_conversation
Get bot WhatsApp conversation details.

**Parameters:**
- `whatsapp_conversation_uuid` (str) - Conversation UUID

**Returns:** `{conversation_data, status_code}`

#### assign_client_to_conversation
Assign customer to WhatsApp conversation.

**Parameters:**
- `cognito_id` (str) - User Cognito ID
- `whatsapp_conversation_id` (str) - Conversation ID
- `organization_client_uuid` (str) - Customer UUID

**Returns:** `{status_code}`

#### get_latest_conversation_by_client_uuid
Get latest conversation for a customer.

**Parameters:**
- `cognito_id` (str) - User Cognito ID
- `organization_client_uuid` (str) - Customer UUID

**Returns:** `{conversation_data, status_code}`

### Phone Numbers

#### get_organization_phone_number_from_conversation_uuid
Get organization phone from conversation.

**Parameters:**
- `whatsapp_conversation_uuid` (str) - Conversation UUID

**Returns:** `{phone_number, status_code}`

#### get_phone_from_conversation
Get organization phone from conversation (alternative endpoint).

**Parameters:**
- `conversation_uuid` (str) - Conversation UUID

**Returns:** `{phone_number, status_code}`

### Templates

#### get_whatsapp_template_instance
Get WhatsApp template instance from internal session.

**Parameters:**
- `internal_orchestration_session_uuid` (str) - Internal session UUID

**Returns:** `{template_instance, status_code}`

#### create_master_template_instance
Create master template instance for bulk campaigns.

**Parameters:**
- `**payload` - Template data (internal_orchestration_session_uuid, template_uuid, parameters)

**Returns:** `{instance_uuid, status_code}`

**Example:**
```python
internal_whatsapp_api_manager.call(
    "create_master_template_instance",
    internal_orchestration_session_uuid=internal_session_uuid,
    template_uuid="tmpl-uuid",
    parameters={"1": "Customer Name", "2": "Payment Amount"},
    access_token=token,
    organization_id=org_id
)
```

---

## Email API

**Import:** `from chask_foundation.api.internal_email_requests import internal_email_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/channels/email`

#### send_email_to_customer
Send email to customer.

**Parameters:**
- `customer_uuid` (str) - Customer UUID
- `thread_uuid` (str) - Email thread UUID
- `body` (str) - Email body (HTML or plain text)
- `orchestration_session_uuid` (str, optional) - Session to link

**Returns:** `{message_id, status_code}`

**Example:**
```python
from chask_foundation.api.internal_email_requests import internal_email_api_manager

internal_email_api_manager.call(
    "send_email_to_customer",
    customer_uuid="cust-uuid",
    thread_uuid="thread-uuid",
    body="<p>Your payment has been processed.</p>",
    orchestration_session_uuid=session_uuid,
    access_token=token,
    organization_id=org_id
)
```

#### retrieve_threads_by_os
Get email threads for orchestration session.

**Parameters:**
- `orchestration_session_uuid` (str) - Session UUID

**Returns:** `{threads, status_code}`

**Example:**
```python
response = internal_email_api_manager.call(
    "retrieve_threads_by_os",
    orchestration_session_uuid=session_uuid,
    access_token=token,
    organization_id=org_id
)
threads = response.get("threads", [])
```

---

## Outbound API

**Import:** `from chask_foundation.api.outbound_requests import outbound_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/outbound`

#### create_outbound_job
Create a new outbound messaging job.

**Parameters:**
- `**payload` - Job configuration

**Returns:** `{job_uuid, status_code}`

**Example:**
```python
from chask_foundation.api.outbound_requests import outbound_api_manager

response = outbound_api_manager.call(
    "create_outbound_job",
    name="Payment Reminders",
    target_group_uuid="group-uuid",
    template_uuid="tmpl-uuid",
    access_token=token,
    organization_id=org_id
)
job_uuid = response.get("job_uuid")
```

#### list_outbound_jobs
List outbound jobs with filters.

**Parameters:**
- `**params` - Filter parameters

**Returns:** `{jobs, status_code}`

#### get_outbound_job
Get single outbound job details.

**Parameters:**
- `outbound_job_uuid` (str) - Job UUID

**Returns:** `{job_data, status_code}`

#### launch_batch_whatsapp
Launch batch WhatsApp send with per-contact parameters.

**Parameters:**
- `**payload` - Batch configuration (job_uuid, contact_parameters)

**Returns:** `{batch_uuid, status_code}`

**Example:**
```python
outbound_api_manager.call(
    "launch_batch_whatsapp",
    job_uuid="job-uuid",
    contact_parameters=[
        {
            "customer_uuid": "cust1",
            "parameters": {"1": "John Doe", "2": "$150"}
        },
        {
            "customer_uuid": "cust2",
            "parameters": {"1": "Jane Smith", "2": "$200"}
        }
    ],
    access_token=token,
    organization_id=org_id
)
```

---

## Internal Channels API

**Import:** `from chask_foundation.api.internal_requests import internal_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/channels/internal`

#### retrieve_latest_internal_conversation
Get latest internal conversation for session.

**Parameters:**
- `internal_orchestration_session_uuid` (str) - Internal session UUID

**Returns:** `{conversation_data, status_code}`

**Example:**
```python
from chask_foundation.api.internal_requests import internal_api_manager

response = internal_api_manager.call(
    "retrieve_latest_internal_conversation",
    internal_orchestration_session_uuid=internal_session_uuid,
    access_token=token,
    organization_id=org_id
)
conversation = response.get("conversation_data")
```

#### retrieve_internal_conversation_messages
Get messages from internal conversation.

**Parameters:**
- `internal_conversation_uuid` (str) - Conversation UUID

**Returns:** `{messages, status_code}`

#### create_internal_message
Create message in internal conversation.

**Parameters:**
- `message` (str) - Message content
- `sender` (str) - Sender identifier
- `conversation_uuid` (str, optional) - Conversation UUID

**Returns:** `{message_uuid, status_code}`

**Example:**
```python
internal_api_manager.call(
    "create_internal_message",
    message="Task completed successfully",
    sender="system",
    conversation_uuid="conv-uuid",
    access_token=token,
    organization_id=org_id
)
```

---

## Channels API

**Import:** `from chask_foundation.api.channels_requests import channels_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/channels`

#### get_channels
Get all channels for organization.

**Parameters:** None

**Returns:** `{channels, status_code}`

**Example:**
```python
from chask_foundation.api.channels_requests import channels_api_manager

response = channels_api_manager.call(
    "get_channels",
    access_token=token,
    organization_id=org_id
)
channels = response.get("channels", [])
```

#### retrieve_template
Retrieve WhatsApp template with structure details.

**Parameters:**
- `template_uuid` (str) - Template UUID

**Returns:** `{template_data, structure, status_code}`

**Example:**
```python
response = channels_api_manager.call(
    "retrieve_template",
    template_uuid="tmpl-uuid",
    access_token=token,
    organization_id=org_id
)
template = response.get("template_data")
structure = response.get("structure")
```

---

## Canvas API

**Import:** `from chask_foundation.api.canvas_requests import canvas_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/canvas`

### Selection Management

#### get_selection_detail
Retrieve full selection content including all element IDs.

**Parameters:**
- `selection_uuid` (str) - Selection UUID

**Returns:** `{uuid, name, element_ids, node_count, edge_count, canvas_uuid, created_by_email, created_at, updated_at, status_code}`

**Example:**
```python
from chask_foundation.api.canvas_requests import canvas_api_manager

response = canvas_api_manager.call(
    "get_selection_detail",
    selection_uuid="sel-uuid-123",
    access_token=token,
    organization_id=org_id
)

# Extract element IDs for processing
element_ids = response.get("element_ids", {})
nodes = element_ids.get("nodes", [])
edges = element_ids.get("edges", [])

print(f"Selection contains {len(nodes)} nodes and {len(edges)} edges")
```

#### list_canvas_selections
List all selections for a canvas (lightweight, no element_ids).

**Parameters:**
- `canvas_uuid` (str) - Canvas UUID

**Returns:** `{selections, status_code}`

**Example:**
```python
response = canvas_api_manager.call(
    "list_canvas_selections",
    canvas_uuid="canvas-uuid",
    access_token=token,
    organization_id=org_id
)

selections = response.get("selections", [])
for selection in selections:
    print(f"{selection['name']}: {selection['uuid']}")
```

#### create_selection
Create a new canvas selection.

**Parameters:**
- `canvas_uuid` (str) - Canvas UUID
- `name` (str) - Selection name
- `element_ids` (dict) - Element IDs: `{"nodes": [...], "edges": [...]}`

**Returns:** `{uuid, name, element_ids, canvas_uuid, status_code}`

**Example:**
```python
response = canvas_api_manager.call(
    "create_selection",
    canvas_uuid="canvas-uuid",
    name="Payment Flow Nodes",
    element_ids={
        "nodes": ["node-1", "node-2", "node-3"],
        "edges": ["edge-1", "edge-2"]
    },
    access_token=token,
    organization_id=org_id
)

selection_uuid = response.get("uuid")
print(f"Created selection: {selection_uuid}")
```

#### update_selection
Update selection name and/or element_ids.

**Parameters:**
- `selection_uuid` (str) - Selection UUID
- `name` (str, optional) - New name
- `element_ids` (dict, optional) - New element IDs

**Returns:** `{uuid, name, element_ids, updated_at, status_code}`

**Note:** At least one of `name` or `element_ids` must be provided.

**Example:**
```python
# Update just the name
canvas_api_manager.call(
    "update_selection",
    selection_uuid="sel-uuid",
    name="Updated Flow Name",
    access_token=token,
    organization_id=org_id
)

# Update just the elements
canvas_api_manager.call(
    "update_selection",
    selection_uuid="sel-uuid",
    element_ids={"nodes": ["node-1", "node-4"], "edges": []},
    access_token=token,
    organization_id=org_id
)

# Update both
canvas_api_manager.call(
    "update_selection",
    selection_uuid="sel-uuid",
    name="New Name",
    element_ids={"nodes": ["node-5"], "edges": ["edge-3"]},
    access_token=token,
    organization_id=org_id
)
```

#### delete_selection
Delete a canvas selection.

**Parameters:**
- `selection_uuid` (str) - Selection UUID

**Returns:** `{message, status_code}`

**Example:**
```python
response = canvas_api_manager.call(
    "delete_selection",
    selection_uuid="sel-uuid",
    access_token=token,
    organization_id=org_id
)

if response.get("status_code") == 200:
    print("Selection deleted successfully")
```

---

## CRM API

**Import:** `from chask_foundation.api.crm_requests import crm_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/crm`

#### validate_customers_bulk
Validate customer UUIDs and check WhatsApp numbers.

**Parameters:**
- `**payload` - Validation data (customer_uuids list)

**Returns:** `{valid_customers, invalid_customers, status_code}`

**Example:**
```python
from chask_foundation.api.crm_requests import crm_api_manager

response = crm_api_manager.call(
    "validate_customers_bulk",
    customer_uuids=["uuid1", "uuid2", "uuid3"],
    access_token=token,
    organization_id=org_id
)
valid = response.get("valid_customers", [])
invalid = response.get("invalid_customers", [])
```

---

## Agents API

**Import:** `from chask_foundation.api.agent_requests import agent_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/agents`

#### get_context_by_channel
Get agent context for a specific channel.

**Parameters:**
- `channel_id` (str) - Channel ID

**Returns:** `{context, status_code}`

**Example:**
```python
from chask_foundation.api.agent_requests import agent_api_manager

response = agent_api_manager.call(
    "get_context_by_channel",
    channel_id="ch-uuid-123",
    access_token=token,
    organization_id=org_id
)
context = response.get("context")
```

---

## Analysts API

**Import:** `from chask_foundation.api.analysts_requests import analysts_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/analysts`

#### get_for_organization
Get all analysts for the organization.

**Parameters:** None

**Returns:** `{analysts, status_code}`

**Example:**
```python
from chask_foundation.api.analysts_requests import analysts_api_manager

response = analysts_api_manager.call(
    "get_for_organization",
    access_token=token,
    organization_id=org_id
)
analysts = response.get("analysts", [])
```

---

## Fintoc API

**Import:** `from chask_foundation.api.fintoc_requests import fintoc_api_manager`

**Base URL:** `https://{BASE_DOMAIN}/api/v2/fintoc`

#### get_organization_link_token
Get decrypted Fintoc link token for organization.

**Parameters:**
- `link_id` (str, optional) - Specific link ID to retrieve

**Returns:** `{link_token, status_code}`

**Example:**
```python
from chask_foundation.api.fintoc_requests import fintoc_api_manager

response = fintoc_api_manager.call(
    "get_organization_link_token",
    access_token=token,
    organization_id=org_id
)
link_token = response.get("link_token")
```

---

## Firebase API

**Import:** `from chask_foundation.api.firebase_requests import firebase_api_manager`

**Base URL:** Configured via `FIREBASE_BASE_URL`

#### get_user_tokens
Get Firebase notification tokens for user.

**Parameters:**
- `cognito_username` (str) - User's Cognito username

**Returns:** `{tokens, status_code}`

**Example:**
```python
from chask_foundation.api.firebase_requests import firebase_api_manager

response = firebase_api_manager.call(
    "get_user_tokens",
    cognito_username="user123",
    access_token=token,
    organization_id=org_id
)
tokens = response.get("tokens", [])
```

---

## Error Handling

All API calls return a `status_code` field. Always check this before processing results.

### Standard Error Pattern

```python
response = api_manager.call(
    "endpoint_name",
    param1=value1,
    access_token=token,
    organization_id=org_id
)

if response.get("status_code") not in (200, 201):
    error_message = response.get("error", "Unknown error")
    error_code = response.get("error_code", "UNKNOWN")

    # Log the error
    logger.error(f"API call failed: {error_message} (code: {error_code})")

    # Handle specific error codes
    if error_code == "NOT_FOUND":
        # Handle not found
        pass
    elif error_code == "UNAUTHORIZED":
        # Handle auth error
        pass
    else:
        # Handle generic error
        raise Exception(f"API call failed: {error_message}")
```

### Common Status Codes

- `200` - Success (GET)
- `201` - Created (POST)
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error

### Retry Configuration

The LLM API manager includes automatic retry for logging endpoints:

```python
# Automatically retries on 500, 502, 503, 504
# Max 3 retries with exponential backoff
llm_api_manager.call("log_call", ...)
```

For other APIs, implement manual retry logic:

```python
import time

def call_with_retry(api_manager, endpoint, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        response = api_manager.call(endpoint, **kwargs)

        if response.get("status_code") in (200, 201):
            return response

        if response.get("status_code") >= 500 and attempt < max_retries - 1:
            # Exponential backoff
            sleep_time = 2 ** attempt
            time.sleep(sleep_time)
            continue

        # Non-retryable error or max retries reached
        break

    return response
```

---

## Common Patterns

### Pattern 1: Event Lifecycle

```python
from chask_foundation.api.orchestrator_requests import orchestrator_api_manager

def handle_function_call(orchestration_event):
    # 1. Process the function call
    result = process_payment(orchestration_event)

    # 2. Evolve the event
    evolved = orchestrator_api_manager.call(
        "evolve_event",
        parent_event_uuid=str(orchestration_event.event_id),
        event_type="function_call_response",
        source="agent",
        target="orchestrator",
        prompt="Payment processed",
        extra_params={"result": result, "is_error": False},
        access_token=orchestration_event.access_token,
        organization_id=orchestration_event.organization.organization_id
    )

    # 3. Forward to Kafka
    response_event = build_response_event(orchestration_event, evolved["uuid"], result)
    orchestrator_api_manager.call(
        "forward_oe_to_kafka",
        orchestration_event=response_event.model_dump(),
        topic="orchestrator",
        access_token=response_event.access_token,
        organization_id=response_event.organization.organization_id
    )
```

### Pattern 2: Pipeline Widget Update

```python
from chask_foundation.api.pipeline_requests import pipeline_api_manager

def update_pipeline_with_results(node_id, results, token, org_id):
    # Get current node data
    node = pipeline_api_manager.call(
        "get_node",
        node_id=node_id,
        access_token=token,
        organization_id=org_id
    )

    # Update widget with results
    pipeline_api_manager.call(
        "update_node_widget",
        node_id=node_id,
        lambda_result={
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "data": results
        },
        lambda_name="process_data",
        access_token=token,
        organization_id=org_id
    )
```

### Pattern 3: Function Search and Invoke

```python
from chask_foundation.api.function_requests import function_api_manager

def find_and_get_function(search_term, token, org_id):
    # Search for functions
    search_response = function_api_manager.call(
        "search_functions",
        search=search_term,
        assigned_only=True,
        fuzzy_threshold=80,
        access_token=token,
        organization_id=org_id
    )

    functions = search_response.get("functions", [])
    if not functions:
        return None

    # Get detailed metadata for first match
    function_uuid = functions[0]["uuid"]
    detail_response = function_api_manager.call(
        "get_function_by_uuid",
        function_uuid=function_uuid,
        access_token=token,
        organization_id=org_id
    )

    return detail_response
```

### Pattern 4: Batch WhatsApp Campaign

```python
from chask_foundation.api.organizations_requests import organizations_api_manager
from chask_foundation.api.outbound_requests import outbound_api_manager
from chask_foundation.api.crm_requests import crm_api_manager

def launch_whatsapp_campaign(customer_uuids, template_uuid, token, org_id):
    # 1. Validate customers
    validation = crm_api_manager.call(
        "validate_customers_bulk",
        customer_uuids=customer_uuids,
        access_token=token,
        organization_id=org_id
    )
    valid_uuids = validation.get("valid_customers", [])

    # 2. Create target group
    group = organizations_api_manager.call(
        "create_target_customer_group",
        customer_uuids=valid_uuids,
        name="Campaign Group",
        access_token=token,
        organization_id=org_id
    )

    # 3. Create outbound job
    job = outbound_api_manager.call(
        "create_outbound_job",
        target_group_uuid=group["group_uuid"],
        template_uuid=template_uuid,
        access_token=token,
        organization_id=org_id
    )

    # 4. Launch batch send
    result = outbound_api_manager.call(
        "launch_batch_whatsapp",
        job_uuid=job["job_uuid"],
        access_token=token,
        organization_id=org_id
    )

    return result
```

### Pattern 5: File Upload and Processing

```python
from chask_foundation.api.files_requests import files_api_manager

def upload_and_read_pdf(file_path, session_uuid, token, org_id):
    # 1. Upload file
    with open(file_path, "rb") as f:
        upload_response = files_api_manager.call(
            "upload_file",
            file=f,
            orchestration_session_uuids=[session_uuid],
            access_token=token,
            organization_id=org_id
        )

    file_uuid = upload_response.get("file_uuid")

    # 2. Extract text from PDF
    if file_path.endswith(".pdf"):
        text_response = files_api_manager.call(
            "read_single_pdf_text",
            attachment_uuid=file_uuid,
            origin="lambda",
            access_token=token,
            organization_id=org_id
        )
        return text_response.get("text")

    return None
```

### Pattern 6: Session Data Persistence

```python
from chask_foundation.api.orchestrator_requests import orchestrator_api_manager

def save_and_retrieve_session_data(session_uuid, data, token, org_id):
    # Save data
    orchestrator_api_manager.call(
        "save_orchestration_session_user_data",
        orchestration_session_id=session_uuid,
        user_data=data,
        access_token=token,
        organization_id=org_id
    )

    # Retrieve data later
    response = orchestrator_api_manager.call(
        "get_orchestration_session_user_data",
        orchestration_session_uuid=session_uuid,
        access_token=token,
        organization_id=org_id
    )

    return response.get("user_data", {})
```

---

## Quick Reference: Most Used Endpoints

### Top 10 Essential Endpoints

1. **evolve_event** - Event tracking and lineage
2. **forward_oe_to_kafka** - Send events to Kafka
3. **create_pipeline** - Create pipeline boards
4. **get_node** - Get pipeline node data
5. **update_node_widget** - Post Lambda results
6. **search_functions** - Find Lambda functions
7. **get_function_by_uuid** - Get function metadata
8. **upload_file** - Upload files
9. **log_call** - Log LLM usage
10. **get_orchestration_session_user_data** - Session persistence

### API Manager Imports Quick Reference

```python
# Orchestrator & Events
from chask_foundation.api.orchestrator_requests import orchestrator_api_manager

# Pipelines
from chask_foundation.api.pipeline_requests import pipeline_api_manager, legacy_api_manager

# Functions
from chask_foundation.api.function_requests import function_api_manager

# Files
from chask_foundation.api.files_requests import files_api_manager

# LLM
from chask_foundation.api.llm_requests import llm_api_manager

# Organizations
from chask_foundation.api.organizations_requests import organizations_api_manager

# WhatsApp
from chask_foundation.api.internal_whatsapp_requests import internal_whatsapp_api_manager

# Email
from chask_foundation.api.internal_email_requests import internal_email_api_manager

# Outbound
from chask_foundation.api.outbound_requests import outbound_api_manager

# Internal Channels
from chask_foundation.api.internal_requests import internal_api_manager

# Channels
from chask_foundation.api.channels_requests import channels_api_manager

# CRM
from chask_foundation.api.crm_requests import crm_api_manager

# Agents
from chask_foundation.api.agent_requests import agent_api_manager

# Analysts
from chask_foundation.api.analysts_requests import analysts_api_manager

# Fintoc
from chask_foundation.api.fintoc_requests import fintoc_api_manager

# Firebase
from chask_foundation.api.firebase_requests import firebase_api_manager
```

---

## Additional Resources

- **Pipeline API Deep Dive**: See `chask_api/documentation/PIPELINES_API.md` for comprehensive pipeline documentation
- **Event Tracking**: See `documentation/how-to/event-tracking-evolution.md` for event evolution patterns
- **LLM Logging**: See `documentation/how-to/cloudwatch-structured-logging.md` for logging best practices
- **Lambda Development**: See `chask-lambdas/CLAUDE.md` for Lambda development guidelines

---

*Last Updated: 2026-01-22*
*Total Endpoints Documented: 98*
*API Version: v2*
