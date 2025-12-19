"""
Read-only Slack API assistant.
Handles user queries via DM, @mention, or slash commands using GPT-4o with function calling.
"""
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
try:
    import google.generativeai as genai
    from google.generativeai.types import FunctionDeclaration, Tool, content_types
except ImportError:
    genai = None
    FunctionDeclaration = None
    Tool = None
    content_types = None
from .tools import (
    get_router,
    TOOL_DEFINITIONS,
    get_planning_prompt
)
from .tools.tracer import get_tracer
# Import slack_client lazily to avoid circular dependencies
# Will be imported only if needed for auth_test()
bolt_app = None

def _get_bolt_app():
    """Lazy import of bolt_app to avoid circular dependencies."""
    global bolt_app
    if bolt_app is None:
        try:
            from ..shared.slack_client import get_app
            bolt_app = get_app()
        except (ImportError, Exception):
            pass
    return bolt_app

logger = logging.getLogger(__name__)

# Conversation history storage
# Format: {(user_id, channel_id): [message1, message2, ...]}
_conversation_history: Dict[tuple, List[Dict[str, Any]]] = {}
_max_history_per_conversation = 10  # Keep last 10 user-assistant exchanges
_history_ttl = 3600  # 1 hour TTL for conversations
_last_cleanup = time.time()

# Provider configuration - read dynamically to allow .env.openai to override
def get_llm_provider() -> str:
    """Get LLM provider dynamically from environment variable."""
    return os.getenv("LLM_PROVIDER", "openai").lower()

def get_openai_model() -> str:
    """Get OpenAI model name dynamically."""
    return os.getenv("MODEL_NAME", "gpt-4o")

def get_claude_model() -> str:
    """Get Claude model name dynamically."""
    return os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")

def get_claude_max_output_tokens() -> int:
    """Get Claude max output tokens dynamically."""
    try:
        return int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "800"))
    except ValueError:
        return 800

def get_openai_max_output_tokens() -> int:
    """Get OpenAI max output tokens dynamically."""
    try:
        return int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096"))
    except ValueError:
        return 4096

def get_claude_temperature() -> float:
    """Get Claude temperature dynamically."""
    try:
        return float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))
    except ValueError:
        return 0.2
def get_gemini_model() -> str:
    """Get Gemini model name dynamically."""
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def get_gemini_max_output_tokens() -> int:
    """Get Gemini max output tokens dynamically."""
    try:
        return int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
    except ValueError:
        return 1024

def get_gemini_temperature() -> float:
    """Get Gemini temperature dynamically."""
    try:
        return float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    except ValueError:
        return 0.7
def get_model_name() -> str:
    """Get the active model name based on provider."""
    provider = get_llm_provider()
    if provider == "claude":
        return get_claude_model()
    elif provider == "gemini":
        return get_gemini_model()
    return get_openai_model()

# LLM_PROVIDER and related variables are now read dynamically via functions above
# This ensures .env.openai settings are picked up even if loaded after module import

# Lazy initialization of clients to avoid errors if .env not loaded yet
_openai_client = None
_anthropic_client = None
_gemini_model_instance = None

def get_openai_client():
    """Get or create OpenAI client (lazy initialization)."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_anthropic_client():
    """Get or create Anthropic client (lazy initialization)."""
    global _anthropic_client
    if _anthropic_client is None:
        if Anthropic is None:
            raise ImportError("anthropic package is not installed. Please `pip install anthropic`.")
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY not set in environment variables")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client

def get_gemini_client():
    """Get or create Gemini model (lazy initialization)."""
    global _gemini_model_instance
    if _gemini_model_instance is None:
        if genai is None:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Please run: pip install google-generativeai"
            )
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        genai.configure(api_key=api_key)
        _gemini_model_instance = genai.GenerativeModel(get_gemini_model())
    return _gemini_model_instance

# Get router instance
router = get_router()


def _get_tool_definitions_for_provider() -> List[Dict[str, Any]]:
    """Return tool definitions formatted for the active provider."""
    provider = get_llm_provider()

    if provider == "claude":
        claude_tools = []
        for tool in TOOL_DEFINITIONS:
            fn = tool.get("function", {})
            claude_tools.append({
                "name": fn.get("name"),
                "description": fn.get("description"),
                "input_schema": fn.get("parameters", {"type": "object"})
            })
        return claude_tools

    elif provider == "gemini":
        # Return OpenAI format - we'll convert in _call_gemini_llm
        # This keeps the interface consistent
        return TOOL_DEFINITIONS

    # OpenAI format (default)
    return TOOL_DEFINITIONS


def _ensure_arguments_string(arguments: Any) -> str:
    """Ensure tool arguments are serialized as a JSON string."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return json.dumps({"raw": str(arguments)})


def _normalize_tool_call(tool_call_id: Optional[str], name: Optional[str], arguments: Any) -> Dict[str, Any]:
    """Normalize provider-specific tool calls to a common dict structure."""
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _ensure_arguments_string(arguments)
        }
    }


def _extract_tool_metadata(tool_call: Any) -> Tuple[Optional[str], Optional[str], str]:
    """Extract (id, name, arguments) from either dict or OpenAI object."""
    if hasattr(tool_call, "function"):
        return (
            getattr(tool_call, "id", None),
            getattr(tool_call.function, "name", None),
            _ensure_arguments_string(getattr(tool_call.function, "arguments", "{}"))
        )
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        return (
            tool_call.get("id"),
            function.get("name"),
            _ensure_arguments_string(function.get("arguments", "{}"))
        )
    return (None, None, "{}")


def _convert_messages_for_claude(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert OpenAI-style message history into Anthropic's format."""
    system_blocks: List[str] = []
    claude_messages: List[Dict[str, Any]] = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            if content:
                system_blocks.append(content)
            continue
        
        if role == "user":
            claude_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": content or ""}]
            })
            continue
        
        if role == "assistant":
            content_blocks: List[Dict[str, Any]] = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            
            for tool_call in msg.get("tool_calls") or []:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    parsed_arguments = {"raw": arguments}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id") or name,
                    "name": name,
                    "input": parsed_arguments or {}
                })
            
            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]
            
            claude_messages.append({
                "role": "assistant",
                "content": content_blocks
            })
            continue
        
        if role == "tool":
            claude_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content", "")
                }]
            })
    
    system_prompt = "\n\n".join(system_blocks) if system_blocks else None
    return system_prompt, claude_messages


def _convert_json_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON Schema (OpenAI format) to Gemini's schema format.

    Gemini uses a subset of OpenAPI 3.0 schema.
    """
    if not schema:
        return {"type": "OBJECT", "properties": {}}

    # Map JSON Schema types to Gemini types
    type_mapping = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    result = {}

    # Handle type
    json_type = schema.get("type", "object")
    if isinstance(json_type, list):
        # Handle nullable types like ["string", "null"]
        json_type = [t for t in json_type if t != "null"][0] if json_type else "string"
    result["type"] = type_mapping.get(json_type, "STRING")

    # Handle description
    if "description" in schema:
        result["description"] = schema["description"]

    # Handle enum
    if "enum" in schema:
        result["enum"] = schema["enum"]

    # Handle properties (for objects)
    if "properties" in schema:
        result["properties"] = {}
        for prop_name, prop_schema in schema["properties"].items():
            result["properties"][prop_name] = _convert_json_schema_to_gemini(prop_schema)

    # Handle required fields
    if "required" in schema:
        result["required"] = schema["required"]

    # Handle array items
    if "items" in schema:
        result["items"] = _convert_json_schema_to_gemini(schema["items"])

    return result


def _convert_tools_for_gemini(openai_tools: List[Dict[str, Any]]) -> List:
    """
    Convert OpenAI-format tool definitions to Gemini FunctionDeclaration objects.
    """
    if not openai_tools or genai is None:
        return []

    gemini_functions = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        fn = tool.get("function", {})
        name = fn.get("name")
        description = fn.get("description", "")
        parameters = fn.get("parameters", {})

        if not name:
            continue

        # Convert parameters schema
        gemini_params = _convert_json_schema_to_gemini(parameters)

        # Create FunctionDeclaration
        func_decl = FunctionDeclaration(
            name=name,
            description=description,
            parameters=gemini_params if gemini_params.get("properties") else None
        )
        gemini_functions.append(func_decl)

    return gemini_functions


def _convert_messages_for_gemini(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List]:
    """
    Convert OpenAI-style messages to Gemini's content format.

    Returns:
        Tuple of (system_instruction, gemini_contents)
    """
    system_instruction = None
    gemini_contents = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            # Gemini uses system_instruction parameter
            system_instruction = content
            continue

        elif role == "user":
            gemini_contents.append({
                "role": "user",
                "parts": [{"text": content}]
            })

        elif role == "assistant":
            parts = []

            # Add text content if present
            if content:
                parts.append({"text": content})

            # Add function calls if present
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name")
                func_args = func.get("arguments", "{}")

                # Parse arguments
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        func_args = {}

                parts.append({
                    "function_call": {
                        "name": func_name,
                        "args": func_args
                    }
                })

            if parts:
                gemini_contents.append({
                    "role": "model",  # Gemini uses "model" instead of "assistant"
                    "parts": parts
                })

        elif role == "tool":
            # Tool results in Gemini are function_response parts
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = msg.get("content", "")

            # Parse the content if it's JSON
            try:
                response_data = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
            except json.JSONDecodeError:
                response_data = {"result": tool_content}

            gemini_contents.append({
                "role": "user",  # Function responses come from "user" role in Gemini
                "parts": [{
                    "function_response": {
                        "name": tool_call_id.split("_")[0] if "_" in tool_call_id else tool_call_id,
                        "response": response_data
                    }
                }]
            })

    return system_instruction, gemini_contents


def _call_openai_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
    """Call OpenAI Chat Completions API."""
    # Get model name dynamically
    model_name = get_openai_model()
    logger.info(f"[LLM] Using OpenAI model: {model_name}")
    
    # Get max_tokens from environment or use default
    max_output_tokens = get_openai_max_output_tokens()
    
    try:
        response = get_openai_client().chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_output_tokens
        )
        # Log the actual model used (OpenAI may resolve to a specific version)
        if hasattr(response, 'model') and response.model != model_name:
            logger.info(f"[LLM] Model resolved to: {response.model}")
    except Exception as e:
        # Log the error with model name for debugging
        error_str = str(e)
        logger.error(f"[LLM] OpenAI API call failed with model '{model_name}': {error_str}")
        # Check if it's a model-related error
        if 'model' in error_str.lower() and 'invalid' in error_str.lower():
            logger.warning(f"[LLM] Model '{model_name}' may be invalid. Check available models.")
        raise
    message = response.choices[0].message
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(_normalize_tool_call(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments
            ))
    
    message_dict = {
        "role": message.role,
        "content": message.content,
    }
    if tool_calls:
        message_dict["tool_calls"] = tool_calls
    
    return message_dict, tool_calls


def _call_claude_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
    """Call Anthropic Messages API with tool support."""
    system_prompt, claude_messages = _convert_messages_for_claude(messages)
    if not claude_messages:
        raise ValueError("No user messages available for Claude call")
    
    client = get_anthropic_client()
    request_payload: Dict[str, Any] = {
        "model": get_claude_model(),
        "messages": claude_messages,
        "tools": tools,
        "max_tokens": get_claude_max_output_tokens(),
        "temperature": get_claude_temperature(),
    }
    if system_prompt:
        request_payload["system"] = system_prompt
    
    response = client.messages.create(**request_payload)
    
    text_parts: List[str] = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(_normalize_tool_call(
                tool_call_id=block.id,
                name=block.name,
                arguments=block.input
            ))
    
    content_text = "\n".join(part.strip() for part in text_parts if part.strip())
    message_dict = {
        "role": "assistant",
        "content": content_text or None,
    }
    if tool_calls:
        message_dict["tool_calls"] = tool_calls
    
    return message_dict, tool_calls


def _call_gemini_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
    """
    Call Google Gemini API with full function calling support.

    Args:
        messages: OpenAI-format message history
        tools: OpenAI-format tool definitions

    Returns:
        Tuple of (message_dict, tool_calls) matching OpenAI/Claude format
    """
    # Get the base model
    base_model = get_gemini_client()

    # Convert tools to Gemini format
    gemini_functions = _convert_tools_for_gemini(tools)

    # Create model with tools if we have any
    if gemini_functions:
        gemini_tools = [Tool(function_declarations=gemini_functions)]
        model = genai.GenerativeModel(
            model_name=get_gemini_model(),
            tools=gemini_tools,
            system_instruction=None  # We'll set this from messages
        )
    else:
        model = base_model

    # Convert messages to Gemini format
    system_instruction, gemini_contents = _convert_messages_for_gemini(messages)

    # If we have a system instruction, recreate model with it
    if system_instruction:
        if gemini_functions:
            gemini_tools = [Tool(function_declarations=gemini_functions)]
            model = genai.GenerativeModel(
                model_name=get_gemini_model(),
                tools=gemini_tools,
                system_instruction=system_instruction
            )
        else:
            model = genai.GenerativeModel(
                model_name=get_gemini_model(),
                system_instruction=system_instruction
            )

    # Configure generation
    generation_config = genai.types.GenerationConfig(
        temperature=get_gemini_temperature(),
        max_output_tokens=get_gemini_max_output_tokens(),
    )

    try:
        # Start chat with history (all but last message)
        if len(gemini_contents) > 1:
            chat = model.start_chat(history=gemini_contents[:-1])
            # Send the last message
            last_content = gemini_contents[-1]
            last_parts = last_content.get("parts", [])
            # Extract text from parts
            last_text = ""
            for part in last_parts:
                if isinstance(part, dict) and "text" in part:
                    last_text = part["text"]
                    break
            response = chat.send_message(last_text, generation_config=generation_config)
        elif gemini_contents:
            # Just one message
            last_content = gemini_contents[0]
            last_parts = last_content.get("parts", [])
            last_text = ""
            for part in last_parts:
                if isinstance(part, dict) and "text" in part:
                    last_text = part["text"]
                    break
            response = model.generate_content(last_text, generation_config=generation_config)
        else:
            raise ValueError("No messages to send to Gemini")

    except Exception as e:
        logger.error(f"[LLM] Gemini API call failed: {e}")
        raise

    # Parse response
    text_parts = []
    tool_calls = []

    # Handle the response
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            for part in candidate.content.parts:
                # Check for text
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    # Generate a unique ID for the tool call
                    tool_call_id = f"{fc.name}_{len(tool_calls)}"

                    # Convert args to dict
                    args_dict = dict(fc.args) if fc.args else {}

                    tool_calls.append(_normalize_tool_call(
                        tool_call_id=tool_call_id,
                        name=fc.name,
                        arguments=args_dict
                    ))

    # Also try direct text access for simpler responses
    if not text_parts and hasattr(response, 'text'):
        try:
            text_parts.append(response.text)
        except Exception:
            pass

    content_text = "\n".join(part.strip() for part in text_parts if part and part.strip())

    message_dict = {
        "role": "assistant",
        "content": content_text or None,
    }
    if tool_calls:
        message_dict["tool_calls"] = tool_calls

    logger.info(f"[LLM] Gemini response: {len(content_text)} chars, {len(tool_calls)} tool calls")

    return message_dict, tool_calls


def _call_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
    """Dispatch to the appropriate LLM provider."""
    provider = get_llm_provider()

    logger.info(f"[LLM] Using provider: {provider}")

    if provider == "claude":
        return _call_claude_llm(messages, tools)
    elif provider == "gemini":
        return _call_gemini_llm(messages, tools)
    return _call_openai_llm(messages, tools)

# Tracer configuration from environment variables
TRACER_ENABLE_CONSOLE = os.getenv("TRACER_ENABLE_CONSOLE", "true").lower() == "true"
TRACER_ENABLE_FILE = os.getenv("TRACER_ENABLE_FILE", "false").lower() == "true"
TRACER_LOG_FILE = os.getenv("TRACER_LOG_FILE", "function_trace.log")

# System prompt for the assistant
ASSISTANT_SYSTEM_PROMPT = f"""You are a helpful Slack assistant that can answer questions about the workspace using read-only tools.

{get_planning_prompt()}

When answering questions:
- Use tools to gather information before responding
- Provide clear, concise answers
- Include relevant details like channel names, user names, timestamps
- Timestamps are already formatted as human-readable dates (YYYY-MM-DD HH:MM:SS format)

**CRITICAL - Channel and User ID Usage:**
- When you call list_channels, you will receive a response with channel objects containing "id" and "name" fields
- **ALWAYS use the EXACT "id" value from the list_channels response** - do NOT make up, guess, or hallucinate channel IDs
- Copy the channel ID character-for-character from the tool response - channel IDs are case-sensitive and must match exactly
- If you need to find a channel by name, call list_channels first, then find the channel with the matching name in the response, and use its exact "id" field
- The same applies to user IDs from list_users or get_user_info - always use the exact "id" from the response
- If a channel or user ID doesn't work, verify you copied it exactly from the tool response - do not modify or abbreviate IDs

**IMPORTANT - Tool Result Handling:**
- Check the tool result's "success" field first
- If success is true: Use the "data" field to answer the question. Do NOT say you're unable to retrieve data.
- If success is false: Check the "error.message" field and use that exact message to explain the issue to the user.
- Never say "I'm unable to retrieve" or "technical issue" unless the tool explicitly returns success: false with an error.

- For search results, summarize key findings and verify they're relevant to the query
- When search results are returned, check if they actually match what the user asked for
- If search results don't seem relevant, mention this to the user and suggest refining the query
- **IMPORTANT**: If search_messages fails with "not_allowed_token_type" or "missing_scope" (these errors 
  only occur with API mode, not JSON mode), this means the search API requires a user token. In this case, 
  offer to search specific channels instead using get_channel_history. For example: "I can't use the global 
  search, but I can search specific channels for you. Which channels should I check?" 
  NOTE: In JSON mode, search_messages should always work - if it fails, report the actual error message.
- **CRITICAL - Message Filtering**: When using get_channel_history to search for specific topics, you MUST 
  filter the messages to only include those that are actually relevant to the user's query. Do NOT return 
  all messages from a channel - only return messages that contain keywords or phrases related to what the 
  user asked for. If a channel has no relevant messages, say so clearly rather than returning unrelated messages.
- For message history, provide context and highlights
- For channel lists, list the channel names and brief details

**CONVERSATION CONTEXT:**
- You have access to previous messages in this conversation
- When a user asks a follow-up question, refer back to previous context
- If a user mentions something from earlier (like "that thread", "the cornering thread", or "what we discussed"), 
  use the conversation history to understand what they're referring to
- Maintain context across multiple questions in the same conversation
- If a user asks about something mentioned earlier, you don't need to ask them to repeat details

Always be helpful and respect user privacy.
"""


def execute_tool_call(tool_call, user_id: str, channel_id: Optional[str] = None, tracer=None) -> Dict[str, Any]:
    """
    Execute a tool call from GPT-4o.
    
    Args:
        tool_call: Tool call object from GPT-4o (ChatCompletionMessageFunctionToolCall)
        user_id: Slack user ID making the request
        channel_id: Channel ID where the request came from
        tracer: Optional FunctionCallTracer instance for tracing
    
    Returns:
        Tool execution result formatted for GPT-4o
    """
    # Access attributes directly (tool_call is an object, not a dict)
    tool_call_id, tool_name, arguments_str = _extract_tool_metadata(tool_call)
    
    if not tool_name:
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "content": json.dumps({
                "success": False,
                "error": {
                    "type": "invalid_tool",
                    "message": "Tool name missing from tool call"
                }
            })
        }
    
    try:
        # Parse arguments
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {arguments_str}, error: {e}")
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "content": json.dumps({
                "success": False,
                "error": {
                    "type": "invalid_arguments",
                    "message": f"Failed to parse arguments: {str(e)}"
                }
            })
        }
    
    # Execute tool via router
    logger.info(f"Executing tool: {tool_name} with args: {arguments}")
    start_time = time.time()
    result = router.execute_tool(
        tool_name=tool_name,
        params=arguments,
        user_id=user_id,
        channel_id=channel_id
    )
    duration_ms = (time.time() - start_time) * 1000
    
    # Log function result to tracer
    if tracer:
        tracer.log_function_result(
            tool_name=tool_name,
            success=result.get("success", False),
            result_data=result if result.get("success") else None,
            error=result.get("error") if not result.get("success") else None,
            duration_ms=duration_ms
        )
    
    # Format result for GPT-4o
    # Ensure the result is properly formatted
    try:
        content = json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"Error serializing tool result: {e}")
        # Return a minimal error response if serialization fails
        content = json.dumps({
            "success": False,
            "error": {
                "type": "serialization_error",
                "message": f"Error formatting tool response: {str(e)}",
                "tool": tool_name
            },
            "data": None
        })
    
    return {
        "tool_call_id": tool_call_id,
        "role": "tool",
        "content": content
    }


def _get_conversation_key(user_id: str, channel_id: Optional[str]) -> tuple:
    """Generate a unique key for conversation history."""
    return (user_id, channel_id or "dm")


def _get_conversation_history(user_id: str, channel_id: Optional[str]) -> List[Dict[str, Any]]:
    """Get conversation history for a user/channel."""
    key = _get_conversation_key(user_id, channel_id)
    return _conversation_history.get(key, [])


def _cleanup_old_conversations() -> None:
    """Remove conversations older than TTL."""
    global _conversation_history
    try:
        current_time = time.time()
        keys_to_remove = []
        
        for key, messages in _conversation_history.items():
            if messages:
                last_timestamp = messages[-1].get("timestamp", 0)
                if current_time - last_timestamp > _history_ttl:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del _conversation_history[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old conversations")
    except Exception as e:
        logger.warning(f"Error during conversation cleanup: {e}")


def _add_to_conversation_history(
    user_id: str,
    channel_id: Optional[str],
    user_message: str,
    assistant_response: str
) -> None:
    """Add a user-assistant exchange to conversation history."""
    global _last_cleanup
    key = _get_conversation_key(user_id, channel_id)
    
    if key not in _conversation_history:
        _conversation_history[key] = []
    
    # Add user message and assistant response
    _conversation_history[key].append({
        "role": "user",
        "content": user_message,
        "timestamp": time.time()
    })
    _conversation_history[key].append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": time.time()
    })
    
    # Trim to max history
    if len(_conversation_history[key]) > _max_history_per_conversation * 2:
        _conversation_history[key] = _conversation_history[key][-_max_history_per_conversation * 2:]
    
    # Clean up old conversations periodically
    current_time = time.time()
    if current_time - _last_cleanup > 300:  # Every 5 minutes
        _cleanup_old_conversations()
        _last_cleanup = current_time


def handle_user_query(
    user_query: str,
    user_id: str,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    max_iterations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Handle a user query using GPT-4o with function calling.
    
    Args:
        user_query: The user's question/request
        user_id: Slack user ID
        channel_id: Channel ID (None for DM)
        thread_ts: Thread timestamp if in a thread
        max_iterations: Maximum number of tool call iterations (defaults to config or 5)
    
    Returns:
        {
            "text": str,  # Final answer text
            "tool_calls": int,  # Number of tool calls made
            "success": bool,
            "error": Optional[str]
        }
    """
    # Get max_iterations from environment variable dynamically
    if max_iterations is None:
        from .tools.config import get_max_iterations
        max_iterations = get_max_iterations()
    
    try:
        # Initialize tracer for this session (create new instance per session)
        from .tools.tracer import FunctionCallTracer
        tracer = FunctionCallTracer(
            enable_console=TRACER_ENABLE_CONSOLE,
            enable_file=TRACER_ENABLE_FILE,
            log_file=TRACER_LOG_FILE
        )
        session_start_time = time.time()
        
        # Log tracer status for debugging
        if TRACER_ENABLE_CONSOLE:
            logger.info(f"[TRACER] Function call tracing enabled (console output)")
        else:
            logger.info(f"[TRACER] Function call tracing disabled (console output)")
        
        # Start tracing session
        tracer.start_session(user_query, user_id, channel_id)
        
        # Get conversation history (graceful degradation if it fails)
        try:
            history = _get_conversation_history(user_id, channel_id)
        except Exception as e:
            logger.warning(f"Failed to retrieve conversation history: {e}")
            history = []
        
        # Build messages list with system prompt, history, and current query
        messages = [
            {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}
        ]
        
        # Add conversation history (only user/assistant messages, not tool calls)
        for msg in history:
            # Only include user and assistant messages, skip tool calls
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        tool_calls_count = 0
        iteration = 0
        
        tool_definitions = _get_tool_definitions_for_provider()
        
        while iteration < max_iterations:
            iteration += 1
            
            # Log iteration start
            tracer.log_iteration_start(iteration, max_iterations)
            
            # Call active LLM provider
            message_dict, llm_tool_calls = _call_llm(messages, tool_definitions)
            
            # Log model response
            tracer.log_model_response(
                content=message_dict.get("content"),
                tool_calls=llm_tool_calls
            )
            
            messages.append(message_dict)
            
            # Check if model wants to call tools
            if llm_tool_calls:
                tool_calls_count += len(llm_tool_calls)
                
                # Execute all tool calls (tracer will log them via execute_tool_call)
                for idx, tool_call in enumerate(llm_tool_calls):
                    # Log function call before execution
                    call_id, tool_name, arguments_str = _extract_tool_metadata(tool_call)
                    tracer.log_function_call(
                        tool_name=tool_name or "unknown_tool",
                        arguments=arguments_str,
                        tool_call_id=call_id,
                        call_index=idx + 1,
                        total_calls=len(llm_tool_calls)
                    )
                    
                    tool_result = execute_tool_call(tool_call, user_id, channel_id, tracer=tracer)
                    messages.append(tool_result)
                
                # Continue loop to get model's response to tool results
                continue
            else:
                # Model has finished (no more tool calls, has final answer)
                final_text = message_dict.get("content") or "I'm sorry, I couldn't generate a response."
                
                # Log final answer
                total_duration = time.time() - session_start_time
                tracer.log_final_answer(final_text, tool_calls_count, total_duration)
                
                # Save to conversation history
                try:
                    _add_to_conversation_history(user_id, channel_id, user_query, final_text)
                except Exception as e:
                    logger.warning(f"Failed to save conversation history: {e}")
                
                return {
                    "text": final_text,
                    "tool_calls": tool_calls_count,
                    "success": True,
                    "error": None
                }
        
        # Max iterations reached
        error_text = "I'm sorry, I reached the maximum number of tool calls. Please try rephrasing your question."
        if tracer:
            total_duration = time.time() - session_start_time
            tracer.log_final_answer(error_text, tool_calls_count, total_duration)
        return {
            "text": error_text,
            "tool_calls": tool_calls_count,
            "success": False,
            "error": "max_iterations_reached"
        }
    
    except Exception as e:
        logger.error(f"Error handling user query: {e}", exc_info=True)
        error_text = f"I encountered an error: {str(e)}. Please try again."
        # Try to log error if tracer exists
        try:
            if 'tracer' in locals() and tracer:
                tracer.log_final_answer(error_text, 0, None)
        except:
            pass
        return {
            "text": error_text,
            "tool_calls": 0,
            "success": False,
            "error": str(e)
        }


def format_response_for_slack(
    response: Dict[str, Any],
    include_metadata: bool = False
) -> str:
    """
    Format the assistant's response for Slack.
    
    Args:
        response: Response from handle_user_query
        include_metadata: Whether to include tool call metadata
    
    Returns:
        Formatted text for Slack
    """
    text = response.get("text", "No response generated.")
    
    if include_metadata and response.get("tool_calls", 0) > 0:
        text += f"\n\n_(Used {response['tool_calls']} tool call(s) to answer)_"
    
    return text


# Track recent responses to prevent duplicates
from collections import defaultdict
_response_cache: Dict[str, float] = {}
_response_cache_ttl = 60  # 1 minute

def _get_response_key(user_id: str, channel_id: str, text: str) -> str:
    """Generate a unique key for a response to prevent duplicates."""
    # Use first 50 chars of text + user + channel to create unique key
    text_hash = hash(text[:50]) if text else 0
    return f"{user_id}:{channel_id}:{text_hash}"

def _is_duplicate_response(response_key: str) -> bool:
    """Check if we've recently sent this exact response."""
    import time
    global _response_cache
    
    current_time = time.time()
    
    # Clean up old entries
    _response_cache = {k: v for k, v in _response_cache.items() if current_time - v < _response_cache_ttl}
    
    if response_key in _response_cache:
        logger.warning(f"[DM] Duplicate response detected, skipping: {response_key}")
        return True
    
    _response_cache[response_key] = current_time
    return False


def handle_dm(event: Dict[str, Any], say) -> None:
    """
    Handle a direct message to the bot.
    
    Args:
        event: Slack event
        say: Slack say function
    """
    user_id = event.get("user")
    channel_id = event.get("channel")  # DM channel ID
    text = event.get("text", "").strip()
    thread_ts = event.get("thread_ts")
    event_ts = event.get("ts")
    
    if not text:
        say(text="Hi! I can help you search Slack, find channels, look up users, and more. What would you like to know?")
        return
    
    logger.info(f"Handling DM from user {user_id}: {text[:100]}")
    
    # Handle query
    response = handle_user_query(
        user_query=text,
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts
    )
    
    # Format and send response
    response_text = format_response_for_slack(response, include_metadata=False)
    
    # Check for duplicate response before sending
    response_key = _get_response_key(user_id, channel_id, response_text)
    if _is_duplicate_response(response_key):
        logger.warning(f"[DM] Skipping duplicate response for user {user_id}")
        return
    
    # Send reply (only once)
    try:
        logger.info(f"[DM] Sending response to user {user_id} in channel {channel_id} (length: {len(response_text)} chars, event_ts: {event_ts})")
        if thread_ts:
            # Reply in thread
            say(text=response_text, thread_ts=thread_ts)
        else:
            # New message
            say(text=response_text)
        logger.info(f"[DM] Response sent successfully")
    except Exception as e:
        logger.error(f"Error sending DM response: {e}", exc_info=True)


def handle_mention(event: Dict[str, Any], say) -> None:
    """
    Handle an @mention of the bot in a channel.
    
    Args:
        event: Slack event
        say: Slack say function
    """
    user_id = event.get("user")
    channel_id = event.get("channel")
    text = event.get("text", "").strip()
    thread_ts = event.get("thread_ts")
    event_ts = event.get("ts")
    
    # Remove bot mention from text
    try:
        app = _get_bolt_app()
        if app:
            # Use Slack API's auth_test, not OpenAI
            auth_result = app.client.auth_test()
            if auth_result and isinstance(auth_result, dict) and auth_result.get("ok"):
                bot_user_id = auth_result.get("user_id")
                if bot_user_id:
                    text = text.replace(f"<@{bot_user_id}>", "").strip()
        # Also remove any other mentions that might be in the text
        import re
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    except Exception as e:
        logger.warning(f"Error getting bot user ID: {e}")
        # Fallback: just remove all mentions
        import re
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        text = text.replace("@slackbench", "").replace("@SlackBench", "").strip()
    
    if not text:
        say(text="Hi! How can I help you? Try asking me about channels, messages, users, or search the workspace.", thread_ts=event_ts)
        return
    
    logger.info(f"Handling mention from user {user_id} in channel {channel_id}: {text[:100]}")
    
    # Handle query
    response = handle_user_query(
        user_query=text,
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts
    )
    
    # Format response
    response_text = format_response_for_slack(response, include_metadata=False)
    
    # Always reply in a thread to avoid channel spam
    try:
        say(text=response_text, thread_ts=event_ts)
    except Exception as e:
        logger.error(f"Error sending mention response: {e}", exc_info=True)


def handle_slash_command(ack, command: Dict[str, Any], respond) -> None:
    """
    Handle a slash command.
    
    Args:
        ack: Slack ack function
        command: Slash command data
        respond: Slack respond function (for ephemeral responses)
    """
    # Acknowledge command immediately
    ack()
    
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")
    text = command.get("text", "").strip()
    
    if not text:
        respond(
            text="Usage: /slackbench <query>\n\nExample: /slackbench search for 'oncall runbook' in #frontend",
            response_type="ephemeral"
        )
        return
    
    logger.info(f"Handling slash command from user {user_id}: {text[:100]}")
    
    # Handle query
    response = handle_user_query(
        user_query=text,
        user_id=user_id,
        channel_id=channel_id
    )
    
    # Format response
    response_text = format_response_for_slack(response, include_metadata=True)
    
    # Send ephemeral response
    try:
        respond(text=response_text, response_type="ephemeral")
    except Exception as e:
        logger.error(f"Error sending slash command response: {e}", exc_info=True)
        respond(
            text=f"Error: {str(e)}",
            response_type="ephemeral"
        )

