# Troubleshooting Guide

## Dependency Issues

### NumPy/Pandas Import Errors

**Problem**: `Unable to import numpy: you should not try to import numpy from its source directory`

**Solution**: See [Common Issues in INSTRUCTIONS.md](./INSTRUCTIONS.md#issue-1-lambda-layer-dependency-conflicts) for detailed guidance on avoiding heavy dependencies and using lightweight alternatives.

### Module Not Found Errors

1. Check `layers/your_layer/python/` contains the package
2. Verify layer is listed in `manifest.yml`
3. Check layer size: `du -sh layers/your_layer/`
4. Ensure dependencies were installed WITH sub-dependencies (no `--no-deps`)
5. Republish: `chask function publish`

## Widget Parameter Issues

### Missing Credentials

1. Check CloudWatch logs for: "Widget data structure: {...}"
2. Verify widget_params in test file match manifest.yml names
3. Use the `extract_widget_params()` helper function for robust parameter access
4. Ensure fallback logic exists for both production and test formats

### Missing or Empty widget_data

**Problem**: Function receives empty `widget_data` or missing secrets even though test file includes `widget_params`

**Cause**: Missing `node_id` parameter in test file

**Solution**: Add `node_id` to your test file:

```json
{
  "function_name": "YourFunctionFn",
  "args": {...},
  "widget_params": {
    "api_token": "test_token"
  },
  "node_id": "your-node-uuid-here"
}
```

**Why**: The Chask operator uses `node_id` to identify which widget configuration to inject into `widget_data`. Without it, the operator doesn't know which configuration to use, resulting in empty or missing widget parameters.

### AttributeError on WidgetParamResolver

**Problem**: `'WidgetParamResolver' object has no attribute 'resolve_by_name'`

**Solution**: Access widget_data directly instead:

```python
params = extract_widget_params(orchestration_event, ["api_token", "api_url"])
api_token = params["api_token"]
```

## File Generation Issues

### Large Files Timeout

- Reduce data size or batch processing
- Increase `timeout` in manifest.yml (max 900s)
- Increase `memory_size` in manifest.yml (more memory = more CPU)
- Consider streaming to S3 instead of in-memory generation

### Memory Errors

- Increase `memory_size` in manifest.yml
- Process data in chunks using generators
- Use `del` to free memory after processing large objects

## External API Issues

### SSL/Certificate Errors

```python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# For testing only
response = requests.get(url, verify=False)
```

### Timeout Errors

- Increase timeout parameter in requests: `requests.get(url, timeout=60)`
- Check network/VPC configuration
- Verify external service is accessible from Lambda

## Testing Issues

### Test Hangs/Timeout

1. Check Lambda logs: `/aws/lambda/ReadProjectFileFn`
2. Verify `response_event_sent` flag is set correctly in response
3. Ensure Kafka response is sent via `send_response_to_orchestrator()`
4. Check test execution flags are preserved (see INSTRUCTIONS.md)

### Test Results Not Returned

**Problem**: CLI hangs waiting for test results

**Solution**: Ensure `is_test` and `test_execution_uuid` flags are preserved in response event:

```python
# This is already in the template's send_response_to_orchestrator()
original_extra_params = orchestration_event.extra_params or {}
if original_extra_params.get("is_test"):
    response_event.extra_params["is_test"] = True
    if original_extra_params.get("test_execution_uuid"):
        response_event.extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]
```

## File Upload/Download Issues

### File Upload Fails

1. Check file size < Lambda limits (50MB response payload)
2. Verify `files_api_manager` has correct access token
3. Check network/VPC configuration
4. Verify S3 permissions

### File Download Fails

1. Check presigned URL hasn't expired
2. Verify network connectivity from Lambda
3. Increase timeout: `requests.get(url, timeout=60)`
4. Check file still exists in S3

## CloudWatch Logs Access

```bash
# View recent logs
aws logs tail /aws/lambda/ReadProjectFileFn \
  --profile development \
  --follow

# Filter by request ID
aws logs tail /aws/lambda/ReadProjectFileFn \
  --profile development \
  --filter-pattern "<request-id>" \
  --format short
```

## Deployment Issues

### Function Not Found

Ensure the function is deployed and assigned to your organization:

```bash
chask function publish
chask function status
```

### Build Failures

1. Check manifest.yml syntax
2. Verify all required files exist (src/handler.py)
3. Check layer paths are correct
4. Review SAM CLI build output

## Performance Issues

### Slow Cold Starts

- Reduce layer sizes (target < 10 MB)
- Minimize import statements
- Increase memory allocation (more CPU allocated proportionally)
- Consider provisioned concurrency for critical functions

### High Memory Usage

- Process data in chunks/batches
- Use generators instead of loading all data into memory
- Clean up large objects with `del` when done
- Monitor memory with: `import psutil; psutil.Process().memory_info().rss`

## Getting Help

1. Check [INSTRUCTIONS.md](./INSTRUCTIONS.md) for detailed guidance
2. Review [README.md](./README.md) for configuration examples
3. Check CloudWatch logs for detailed error messages
4. Contact Chask support with:
   - Function name
   - Lambda Request ID
   - Error message
   - CloudWatch log link
