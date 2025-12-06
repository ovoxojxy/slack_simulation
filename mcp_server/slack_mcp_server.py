#!/usr/bin/env python3
"""
Slack MCP Server - Bridges Slack and Cursor via Model Context Protocol

This server exposes Slack channels, history, and messaging capabilities to Cursor
through the MCP protocol, allowing AI assistants to read context and respond in Slack.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Optional, Sequence
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
    import mcp.server.stdio
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('mcp_server.log'), logging.StreamHandler()]
)
logger = logging.getLogger("slack-mcp")

# Initialize Slack clients
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
slack_user_token = os.getenv("SLACK_USER_TOKEN")

if not slack_bot_token:
    logger.error("SLACK_BOT_TOKEN not found in environment")
    sys.exit(1)

# Primary client (bot token)
slack_client = WebClient(token=slack_bot_token)

# User client (if user token is available)
slack_user_client = WebClient(token=slack_user_token) if slack_user_token else None

if slack_user_client:
    logger.info("User token detected - can post as actual Slack users")
else:
    logger.info("No user token - will post as bot with custom username/emoji")

# Cache for channel data
CHANNEL_CACHE = {}
CHANNEL_CACHE_TIME = 0
CACHE_TTL = 300  # 5 minutes


def get_channels(force_refresh: bool = False) -> dict:
    """Get list of channels with caching"""
    global CHANNEL_CACHE, CHANNEL_CACHE_TIME
    
    current_time = asyncio.get_event_loop().time()
    if not force_refresh and CHANNEL_CACHE and (current_time - CHANNEL_CACHE_TIME) < CACHE_TTL:
        return CHANNEL_CACHE
    
    try:
        channels = {}
        cursor = None
        
        while True:
            response = slack_client.conversations_list(
                limit=200,
                cursor=cursor,
                types="public_channel,private_channel"
            )
            
            for channel in response.get("channels", []):
                channels[channel["id"]] = {
                    "id": channel["id"],
                    "name": channel["name"],
                    "is_private": channel.get("is_private", False),
                    "is_member": channel.get("is_member", False),
                    "topic": channel.get("topic", {}).get("value", ""),
                    "purpose": channel.get("purpose", {}).get("value", ""),
                }
            
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        CHANNEL_CACHE = channels
        CHANNEL_CACHE_TIME = current_time
        logger.info(f"Loaded {len(channels)} channels into cache")
        return channels
        
    except SlackApiError as e:
        logger.error(f"Error fetching channels: {e.response['error']}")
        return {}


def get_channel_history(channel_id: str, limit: int = 50, oldest: Optional[str] = None) -> list:
    """Fetch recent messages from a channel"""
    try:
        kwargs = {
            "channel": channel_id,
            "limit": limit,
        }
        if oldest:
            kwargs["oldest"] = oldest
        
        response = slack_client.conversations_history(**kwargs)
        return response.get("messages", [])
        
    except SlackApiError as e:
        logger.error(f"Error fetching history for {channel_id}: {e.response['error']}")
        return []


def get_thread_replies(channel_id: str, thread_ts: str, limit: int = 50) -> list:
    """Fetch replies in a thread"""
    try:
        response = slack_client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit
        )
        return response.get("messages", [])
        
    except SlackApiError as e:
        logger.error(f"Error fetching thread {thread_ts}: {e.response['error']}")
        return []


def format_message(msg: dict, include_thread: bool = False) -> str:
    """Format a Slack message for display"""
    user = msg.get("user") or msg.get("username", "unknown")
    text = msg.get("text", "")
    ts = msg.get("ts", "")
    
    # Convert timestamp to readable format
    try:
        dt = datetime.fromtimestamp(float(ts))
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        time_str = ts
    
    formatted = f"[{time_str}] {user}: {text}"
    
    if include_thread and msg.get("thread_ts"):
        formatted += f" [thread: {msg.get('thread_ts')}]"
    
    return formatted


def post_message_to_slack(
    channel: str,
    text: str,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    thread_ts: Optional[str] = None,
    as_user: bool = False
) -> dict:
    """
    Post a message to Slack
    
    Args:
        channel: Channel ID or name (e.g., "C12345" or "#general")
        text: Message text
        username: Custom username (bot token only)
        icon_emoji: Custom emoji (bot token only)
        thread_ts: Thread timestamp to reply to
        as_user: If True and user token available, post as actual user
    """
    try:
        # Determine which client to use
        client = slack_user_client if (as_user and slack_user_client) else slack_client
        
        kwargs = {
            "channel": channel,
            "text": text,
        }
        
        # User token posts as the authenticated user (no custom username/emoji)
        if as_user and slack_user_client:
            logger.info(f"Posting as actual user to {channel}")
            # User tokens don't support username/icon_emoji
            # The message appears from the authenticated user
        else:
            # Bot token allows custom username and emoji
            if username:
                kwargs["username"] = username
            if icon_emoji:
                kwargs["icon_emoji"] = icon_emoji
        
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        
        response = client.chat_postMessage(**kwargs)
        return {
            "success": True,
            "ts": response.get("ts"),
            "channel": response.get("channel"),
            "posted_as": "user" if (as_user and slack_user_client) else "bot",
        }
        
    except SlackApiError as e:
        logger.error(f"Error posting message: {e.response['error']}")
        return {
            "success": False,
            "error": e.response['error'],
        }


# Initialize MCP Server
app = Server("slack-mcp")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """
    List available Slack channels as resources.
    Each channel can be read to get its recent history.
    """
    channels = get_channels()
    
    resources = []
    for channel_id, channel_info in channels.items():
        channel_name = channel_info["name"]
        is_private = " (private)" if channel_info["is_private"] else ""
        
        resources.append(Resource(
            uri=f"slack://channel/{channel_id}",
            name=f"#{channel_name}{is_private}",
            description=f"Recent messages from #{channel_name}. Topic: {channel_info['topic'][:100]}",
            mimeType="text/plain",
        ))
    
    logger.info(f"Listed {len(resources)} channel resources")
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """
    Read a Slack channel's recent history.
    URI format: slack://channel/{channel_id}
    Optional query params: ?limit=N&oldest=timestamp
    """
    logger.info(f"Reading resource: {uri}")
    
    # Parse URI
    if not uri.startswith("slack://channel/"):
        raise ValueError(f"Invalid URI format: {uri}")
    
    # Extract channel ID and parameters
    parts = uri.replace("slack://channel/", "").split("?")
    channel_id = parts[0]
    
    # Parse query parameters
    limit = 50
    oldest = None
    if len(parts) > 1:
        params = dict(param.split("=") for param in parts[1].split("&"))
        limit = int(params.get("limit", 50))
        oldest = params.get("oldest")
    
    # Get channel info
    channels = get_channels()
    channel_info = channels.get(channel_id)
    
    if not channel_info:
        raise ValueError(f"Channel not found: {channel_id}")
    
    channel_name = channel_info["name"]
    
    # Fetch history
    messages = get_channel_history(channel_id, limit=limit, oldest=oldest)
    
    # Format output
    output_lines = [
        f"# Slack Channel: #{channel_name}",
        f"Channel ID: {channel_id}",
        f"Topic: {channel_info.get('topic', 'N/A')}",
        f"Purpose: {channel_info.get('purpose', 'N/A')}",
        f"Messages: {len(messages)} (most recent first)",
        "",
        "## Recent Messages:",
        ""
    ]
    
    # Add messages (reverse to show oldest first)
    for msg in reversed(messages):
        if msg.get("subtype") in ["channel_join", "channel_leave"]:
            continue
        output_lines.append(format_message(msg, include_thread=True))
    
    return "\n".join(output_lines)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available Slack tools for interaction.
    """
    return [
        Tool(
            name="slack_list_channels",
            description="List all available Slack channels with their metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "refresh": {
                        "type": "boolean",
                        "description": "Force refresh the channel cache",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="slack_get_history",
            description="Get recent message history from a specific Slack channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or channel name (with or without #)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to retrieve (default: 50)",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200
                    },
                    "oldest": {
                        "type": "string",
                        "description": "Only messages after this Unix timestamp"
                    }
                },
                "required": ["channel"]
            }
        ),
        Tool(
            name="slack_get_thread",
            description="Get all replies in a specific Slack thread",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or channel name (with or without #)"
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp (parent message timestamp)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of replies to retrieve (default: 50)",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200
                    }
                },
                "required": ["channel", "thread_ts"]
            }
        ),
        Tool(
            name="slack_post_message",
            description="Post a message to a Slack channel (optionally as a persona and/or in a thread). Can post as bot with custom username/emoji or as actual user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or channel name (with or without #)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to post"
                    },
                    "username": {
                        "type": "string",
                        "description": "Username to post as (bot mode only, ignored if as_user=true)"
                    },
                    "icon_emoji": {
                        "type": "string",
                        "description": "Emoji icon for the message (e.g., ':robot_face:') (bot mode only, ignored if as_user=true)"
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp to reply to (optional)"
                    },
                    "as_user": {
                        "type": "boolean",
                        "description": "If true and user token is available, post as authenticated Slack user (ignores username/icon_emoji)",
                        "default": False
                    }
                },
                "required": ["channel", "text"]
            }
        ),
        Tool(
            name="slack_search_messages",
            description="Search for messages across all channels",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """
    Handle tool calls from Cursor/Claude.
    """
    logger.info(f"Tool called: {name} with args: {arguments}")
    
    try:
        if name == "slack_list_channels":
            refresh = arguments.get("refresh", False)
            channels = get_channels(force_refresh=refresh)
            
            output_lines = [f"Found {len(channels)} Slack channels:\n"]
            for ch_id, ch_info in sorted(channels.items(), key=lambda x: x[1]["name"]):
                private = " (private)" if ch_info["is_private"] else ""
                member = " ✓" if ch_info["is_member"] else ""
                output_lines.append(
                    f"  #{ch_info['name']}{private}{member}\n"
                    f"    ID: {ch_id}\n"
                    f"    Topic: {ch_info['topic'][:80]}\n"
                )
            
            return [TextContent(type="text", text="".join(output_lines))]
        
        elif name == "slack_get_history":
            channel = arguments["channel"]
            limit = arguments.get("limit", 50)
            oldest = arguments.get("oldest")
            
            # Resolve channel name to ID if needed
            if channel.startswith("#"):
                channel = channel[1:]
            
            channels = get_channels()
            channel_id = None
            
            # Try to find by name or ID
            if channel in channels:
                channel_id = channel
            else:
                for ch_id, ch_info in channels.items():
                    if ch_info["name"] == channel:
                        channel_id = ch_id
                        break
            
            if not channel_id:
                return [TextContent(type="text", text=f"Channel not found: {channel}")]
            
            messages = get_channel_history(channel_id, limit=limit, oldest=oldest)
            channel_name = channels[channel_id]["name"]
            
            output_lines = [
                f"# Messages from #{channel_name} (Channel ID: {channel_id})\n",
                f"Retrieved {len(messages)} messages:\n\n"
            ]
            
            for msg in reversed(messages):
                if msg.get("subtype") in ["channel_join", "channel_leave"]:
                    continue
                output_lines.append(format_message(msg, include_thread=True) + "\n")
            
            return [TextContent(type="text", text="".join(output_lines))]
        
        elif name == "slack_get_thread":
            channel = arguments["channel"]
            thread_ts = arguments["thread_ts"]
            limit = arguments.get("limit", 50)
            
            # Resolve channel name to ID if needed
            if channel.startswith("#"):
                channel = channel[1:]
            
            channels = get_channels()
            channel_id = None
            
            if channel in channels:
                channel_id = channel
            else:
                for ch_id, ch_info in channels.items():
                    if ch_info["name"] == channel:
                        channel_id = ch_id
                        break
            
            if not channel_id:
                return [TextContent(type="text", text=f"Channel not found: {channel}")]
            
            messages = get_thread_replies(channel_id, thread_ts, limit=limit)
            channel_name = channels[channel_id]["name"]
            
            output_lines = [
                f"# Thread in #{channel_name}\n",
                f"Thread timestamp: {thread_ts}\n",
                f"Retrieved {len(messages)} messages:\n\n"
            ]
            
            for msg in messages:
                output_lines.append(format_message(msg) + "\n")
            
            return [TextContent(type="text", text="".join(output_lines))]
        
        elif name == "slack_post_message":
            channel = arguments["channel"]
            text = arguments["text"]
            username = arguments.get("username")
            icon_emoji = arguments.get("icon_emoji")
            thread_ts = arguments.get("thread_ts")
            as_user = arguments.get("as_user", False)
            
            # Resolve channel name to ID if needed
            if channel.startswith("#"):
                channel = channel[1:]
            
            channels = get_channels()
            channel_id = None
            
            if channel in channels:
                channel_id = channel
            else:
                for ch_id, ch_info in channels.items():
                    if ch_info["name"] == channel:
                        channel_id = ch_id
                        break
            
            if not channel_id:
                return [TextContent(type="text", text=f"Channel not found: {channel}")]
            
            result = post_message_to_slack(
                channel=channel_id,
                text=text,
                username=username,
                icon_emoji=icon_emoji,
                thread_ts=thread_ts,
                as_user=as_user
            )
            
            if result["success"]:
                posted_mode = result.get("posted_as", "bot")
                mode_text = "as actual user" if posted_mode == "user" else "as bot"
                return [TextContent(
                    type="text",
                    text=f"✓ Message posted successfully to #{channels[channel_id]['name']} {mode_text}\n"
                         f"Message timestamp: {result['ts']}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"✗ Failed to post message: {result['error']}"
                )]
        
        elif name == "slack_search_messages":
            query = arguments["query"]
            count = arguments.get("count", 20)
            
            try:
                response = slack_client.search_messages(query=query, count=count)
                matches = response.get("messages", {}).get("matches", [])
                
                output_lines = [
                    f"Found {len(matches)} messages matching '{query}':\n\n"
                ]
                
                for match in matches:
                    channel = match.get("channel", {})
                    channel_name = channel.get("name", "unknown")
                    user = match.get("username", "unknown")
                    text = match.get("text", "")
                    ts = match.get("ts", "")
                    
                    output_lines.append(
                        f"[#{channel_name}] {user}: {text}\n"
                        f"  (ts: {ts})\n\n"
                    )
                
                return [TextContent(type="text", text="".join(output_lines))]
                
            except SlackApiError as e:
                return [TextContent(
                    type="text",
                    text=f"Search failed: {e.response['error']}"
                )]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    logger.info("Starting Slack MCP Server...")
    
    # Pre-load channels
    channels = get_channels()
    logger.info(f"Initialized with {len(channels)} channels")
    
    # Run server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running on stdio")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

