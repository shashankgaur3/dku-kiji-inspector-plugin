import logging

logger = logging.getLogger(__name__)


def create_payload(model: str, query: dict, settings: dict) -> dict:
    """
    Create the JSON payload for the API request.

    Args:
        model: The model to use for the request.
        query: The query object containing messages.
        settings: Request settings/parameters.

    Returns:
        dict: JSON payload for the API request.
    """
    logger.debug(f"Creating payload for model: {model}")
    # Transform messages to the format expected by OpenAI-compatible APIs for tool calling.
    original_messages = query.get('messages', [])
    transformed_messages = []
    for message in original_messages:
        if message.get('role') == 'assistant' and 'toolCalls' in message:
            # Convert Dataiku 'toolCalls' to OpenAI 'tool_calls' and add 'content: null'.
            transformed_messages.append({
                'role': 'assistant',
                'content': None,
                'tool_calls': message['toolCalls']
            })
        elif message.get('role') == 'tool' and 'toolOutputs' in message:
            # Convert Dataiku 'toolOutputs' array into individual tool messages.
            for tool_output in message.get('toolOutputs', []):
                transformed_messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_output.get('callId'),
                    'content': str(tool_output.get('output', '')) # Content must be a string.
                })
        else:
            # Keep other messages (e.g., user messages) as they are.
            transformed_messages.append(message)

    logger.debug(f"Transformed {len(original_messages)} messages to {len(transformed_messages)} messages")

    payload = {
        "model": model,
        "messages": transformed_messages,
        "return_explanations": True
    }

    # Add optional settings if provided
    if settings:
        # Add common LLM parameters from settings
        if 'temperature' in settings:
            payload['temperature'] = settings['temperature']
        if 'maxOutputTokens' in settings:
            payload['max_tokens'] = settings['maxOutputTokens']
        if 'topP' in settings:
            payload['top_p'] = settings['topP']
        if 'presencePenalty' in settings:
            payload['presence_penalty'] = settings['presencePenalty']
        if 'frequencyPenalty' in settings:
            payload['frequency_penalty'] = settings['frequencyPenalty']
        if 'tools' in settings:
            payload['tools'] = settings['tools']
            payload['tool_choice'] = settings.get('tool_choice', 'auto')
            logger.debug(f"Added {len(settings['tools'])} tools to payload")
        if 'stream' in settings:
            payload['stream'] = settings.get('stream', False)

    logger.debug(f"Created payload with {len(payload)} keys")
    return payload


def parse_response(response_data: dict, prompt_cost: float = 0.0, completion_cost: float = 0.0) -> dict:
    """
    Parse the API response and extract relevant information.

    Args:
        response_data: Raw JSON response from the API.
        prompt_cost: Cost per 1000 prompt tokens.
        completion_cost: Cost per 1000 completion tokens.

    Returns:
        dict: Parsed result with text, tokens, cost, and tool calls.
    """
    logger.debug("Parsing API response")
    result = {
        "text": "",  # Default fallback
        "promptTokens": 0,
        "completionTokens": 0,
        "estimatedCost": 0.0,
        "toolCalls": [],
    }

    if 'choices' in response_data and len(response_data['choices']) > 0:
        choice = response_data['choices'][0]
        message = choice.get('message', {})

        # Extract text content
        content = message.get('content')
        if content:
            result["text"] = content
            logger.debug(f"Extracted text content of length: {len(content)}")

        # Extract tool calls if present
        tool_calls = message.get('tool_calls', [])
        if tool_calls:
            # For non-streaming, return the tool_calls object directly as per API response
            result["toolCalls"] = tool_calls
            logger.debug(f"Extracted {len(tool_calls)} tool calls")

        # Correct the finish reason. If tool calls are present, it must be 'tool_calls'.
        if result["toolCalls"]:
            result["finishReason"] = "tool_calls"
        else:
            result["finishReason"] = choice.get('finish_reason', 'stop')
        logger.debug(f"Finish reason: {result['finishReason']}")


    usage = response_data.get('usage', {})
    if usage:
        result["promptTokens"] = usage.get('prompt_tokens', 0)
        result["completionTokens"] = usage.get('completion_tokens', 0) or (usage.get('total_tokens', 0) - result["promptTokens"])

    logger.info("Successfully parsed API response")
    return result
