"""
⚠️ INFRASTRUCTURE CODE - DO NOT MODIFY

ReadProjectFileFn - Lambda Handler

Lambda function created with Chask CLI

This Lambda function is deployed as an organization-specific function via Chask.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: This file contains INFRASTRUCTURE CODE and should NOT be modified.

To implement your business logic, edit:
    src/backend/function_logic.py

Specifically, implement the process_request() method in the FunctionBackend class.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
from typing import Dict, Any, Optional
from chask_foundation.backend.models import OrchestrationEvent
from api.orchestrator_requests import orchestrator_api_manager

# Import the backend class that contains business logic
from backend import FunctionBackend

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_event(event: Dict[str, Any]) -> OrchestrationEvent:
    """
    Parse and validate the Lambda event to extract OrchestrationEvent.

    Args:
        event: Raw Lambda event

    Returns:
        Validated OrchestrationEvent instance

    Raises:
        ValueError: If event is malformed or missing required fields
    """
    # Handle stringified JSON
    if isinstance(event, str):
        event = json.loads(event)

    # Handle API Gateway format (nested body)
    if "body" in event:
        body = event["body"]
        event = json.loads(body) if isinstance(body, str) else body

    # Extract and validate orchestration event
    orchestration_event_data = event.get("orchestration_event")
    if not orchestration_event_data:
        raise ValueError("Missing 'orchestration_event' in Lambda event")

    return OrchestrationEvent.model_validate(orchestration_event_data)


def send_response_to_orchestrator(
    orchestration_event: OrchestrationEvent, message: str, is_error: bool = False
) -> bool:
    """
    Send the function result back to the orchestrator via Kafka.

    Uses evolve_event API to create proper parent-child event linkage
    for event traceability in the Event Tracking System.

    This function is called by the handler (NOT by developer code) to ensure:
    - Test flags are properly preserved
    - Responses are always sent even when developer code fails
    - Consistent response format
    - Events are properly persisted via evolve_event

    Args:
        orchestration_event: The original orchestration event
        message: Result message to send back
        is_error: Whether this is an error response

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
        # CRITICAL: Preserve test execution flags for test commands
        # ====================================================================
        # These flags MUST be preserved to ensure test executions are properly
        # tracked and results are returned to the CLI. There are TWO types of tests:
        #
        # 1. Function tests: 'chask function test'
        #    - Uses is_test=True and test_execution_uuid
        #
        # 2. Pipeline node tests: 'chask pipeline test <pipeline_id> <node_id>'
        #    - Uses is_node_test=True and node_test_execution_uuid
        #
        # Both MUST be preserved or the CLI will never receive the result.
        # DO NOT REMOVE OR MODIFY THIS CODE.

        # Preserve function test flags
        if original_extra_params.get("is_test"):
            extra_params["is_test"] = True
            if original_extra_params.get("test_execution_uuid"):
                extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]

        # Preserve node test flags
        if original_extra_params.get("is_node_test"):
            extra_params["is_node_test"] = True
            if original_extra_params.get("node_test_execution_uuid"):
                extra_params["node_test_execution_uuid"] = original_extra_params["node_test_execution_uuid"]
            # Also preserve pipeline and node IDs for node tests
            if original_extra_params.get("pipeline_id"):
                extra_params["pipeline_id"] = original_extra_params["pipeline_id"]
            if original_extra_params.get("node_id"):
                extra_params["node_id"] = original_extra_params["node_id"]
        # ====================================================================

        # Evolve the event to create a linked function_call_response child event
        # This maintains parent-child relationships via evolved_from_uuid
        # and PERSISTS the event in the database
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


def notify_agent_available(orchestration_event: OrchestrationEvent) -> None:
    """
    Notify the orchestrator that the agent is available again.

    This function MUST be called in the finally block to ensure the agent
    is freed even if the Lambda crashes, times out, or raises an exception.

    Uses evolve_event API to create proper parent-child event linkage
    for event traceability in the Event Tracking System.

    IMPORTANT: This function must NEVER raise an exception, as it runs in
    the finally block and any exception would mask the original error.

    Args:
        orchestration_event: The orchestration event
    """
    try:
        # Check if this is a test execution - skip agent liberation for tests
        extra_params = orchestration_event.extra_params or {}
        is_test = extra_params.get("is_test", False)
        is_node_test = extra_params.get("is_node_test", False)

        if is_test or is_node_test:
            test_type = "node test" if is_node_test else "function test"
            logger.info(f"Skipping agent liberation for {test_type} execution")
            return

        # Evolve the event to create a linked agent_available child event
        # This maintains parent-child relationships via evolved_from_uuid
        evolve_response = orchestrator_api_manager.call(
            "evolve_event",
            parent_event_uuid=str(orchestration_event.event_id),
            event_type="agent_available",
            source="agent",
            target="agent_manager",
            prompt="",
            extra_params={},
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

        agent_event = orchestration_event.model_copy(deep=True)
        agent_event.event_id = evolved_uuid
        agent_event.event_type = "agent_available"
        agent_event.source = "agent"
        agent_event.target = "agent_manager"
        agent_event.prompt = ""
        agent_event.extra_params = evolve_response.get("extra_params", {})

        # Send the evolved orchestration event to the agent manager
        orchestrator_api_manager.call(
            "forward_oe_to_kafka",
            orchestration_event=agent_event.model_dump(),
            topic="agent_manager",
            access_token=agent_event.access_token,
            organization_id=agent_event.organization.organization_id,
        )

        logger.info(
            f"Agent marked as available "
            f"[evolved from {orchestration_event.event_id} -> {evolved_uuid}]"
        )

    except Exception as e:
        # CRITICAL: Suppress all exceptions to prevent masking original errors
        # Log the error but DO NOT re-raise
        logger.error(f"Failed to notify agent available (non-fatal): {e}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function entry point.

    This handler provides a resilient wrapper around your business logic:
    - Parses the event and extracts the OrchestrationEvent
    - Instantiates your backend class and calls process_request()
    - Catches any exceptions and sends error responses
    - ALWAYS frees the agent via the finally block (even on crash/timeout)

    Args:
        event: Event data from AWS Lambda containing:
            - body.orchestration_event: Full orchestration event context
            - body.access_token: API access token
            - body.organization_id: Organization UUID

        context: Lambda context object

    Returns:
        Dictionary with response data (also sent to orchestrator via Kafka)
    """
    backend = None
    orchestration_event = None
    request_id = context.aws_request_id if context else "unknown"

    try:
        logger.info(f"[{request_id}] Lambda invoked")

        # Parse and validate event
        orchestration_event = parse_event(event)

        logger.info(
            f"Processing event for org: {orchestration_event.organization.organization_id}"
        )

        # Instantiate backend with orchestration event
        backend = FunctionBackend(orchestration_event)

        # Execute business logic (developer's code)
        result = backend.process_request()

        # Handler sends the success response (not the backend!)
        response_sent = send_response_to_orchestrator(
            orchestration_event,
            result,
            is_error=False
        )

        # Return success response
        return success_response(
            result={
                "message": result,
                "request_id": request_id
            },
            response_event_sent=response_sent
        )

    except ValueError as e:
        # Parameter validation error (400)
        logger.error(f"Validation error: {str(e)}", exc_info=True)

        # Handler sends error response (guaranteed to preserve test flags)
        response_sent = False
        if orchestration_event:
            response_sent = send_response_to_orchestrator(
                orchestration_event,
                f"Validation error: {str(e)}",
                is_error=True
            )

        return error_response(
            f"Validation error: {str(e)}",
            response_event_sent=response_sent,
            status_code=400
        )

    except Exception as e:
        # Runtime error (500)
        logger.error(f"Lambda error: {str(e)}", exc_info=True)

        # Handler sends error response (guaranteed to preserve test flags)
        response_sent = False
        if orchestration_event:
            response_sent = send_response_to_orchestrator(
                orchestration_event,
                f"Lambda error: {str(e)}",
                is_error=True
            )

        return error_response(
            f"Lambda error: {str(e)}",
            response_event_sent=response_sent,
            status_code=500
        )

    finally:
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Always free the agent, even if Lambda crashes or times out
        # ═══════════════════════════════════════════════════════════════════════
        # This finally block ALWAYS executes, ensuring the agent is never stuck.
        # The agent manager relies on this event to make the agent available again.
        # DO NOT REMOVE OR MODIFY THIS CODE.
        if orchestration_event:
            notify_agent_available(orchestration_event)


def success_response(result: Dict[str, Any], response_event_sent: bool = False, status_code: int = 200) -> Dict[str, Any]:
    """
    Format a successful response.

    Args:
        result: Result data to return
        response_event_sent: Whether the response event was sent to orchestrator (REQUIRED at body level!)
        status_code: HTTP status code

    Returns:
        Formatted response dictionary with response_event_sent flag at body level

    IMPORTANT: response_event_sent MUST be at the body level, not inside result.
    The agent manager checks for body.response_event_sent to determine if it needs
    to send a fallback event.
    """
    return {
        "statusCode": status_code,
        "body": {
            "status": "ok",
            "result": result,
            "response_event_sent": response_event_sent  # Must be at body level!
        }
    }


def error_response(error_message: str, response_event_sent: bool = False, status_code: int = 500) -> Dict[str, Any]:
    """
    Format an error response.

    Args:
        error_message: Error message to return
        response_event_sent: Whether the response event was sent to orchestrator (REQUIRED at body level!)
        status_code: HTTP status code

    Returns:
        Formatted error response dictionary with response_event_sent flag at body level

    IMPORTANT: response_event_sent MUST be at the body level, not inside error.
    The agent manager checks for body.response_event_sent to determine if it needs
    to send a fallback event.
    """
    return {
        "statusCode": status_code,
        "body": {
            "status": "error",
            "error": error_message,
            "response_event_sent": response_event_sent  # Must be at body level!
        }
    }
