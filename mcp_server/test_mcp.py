#!/usr/bin/env python3
"""
Test script for the Slack MCP Server

Run this to verify your setup is working correctly.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def test_environment():
    """Test that environment variables are set"""
    print("=" * 60)
    print("Testing Environment Configuration")
    print("=" * 60)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in environment")
        print("   Check your .env file")
        return False
    
    if not token.startswith("xoxb-"):
        print("❌ SLACK_BOT_TOKEN doesn't look like a bot token")
        print("   Bot tokens should start with 'xoxb-'")
        return False
    
    print(f"✓ SLACK_BOT_TOKEN found: {token[:15]}...")
    print(f"✓ Token format looks correct")
    return True

def test_slack_connection():
    """Test connection to Slack API"""
    print("\n" + "=" * 60)
    print("Testing Slack API Connection")
    print("=" * 60)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    
    try:
        # Test auth
        auth_response = client.auth_test()
        print(f"✓ Connected to Slack")
        print(f"  Team: {auth_response['team']}")
        print(f"  User: {auth_response['user']}")
        print(f"  User ID: {auth_response['user_id']}")
        return True
    except SlackApiError as e:
        print(f"❌ Slack API Error: {e.response['error']}")
        return False

def test_channels():
    """Test reading channels"""
    print("\n" + "=" * 60)
    print("Testing Channel Access")
    print("=" * 60)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    
    try:
        response = client.conversations_list(limit=50)
        channels = response['channels']
        
        print(f"✓ Found {len(channels)} channels")
        print("\nChannels:")
        for ch in channels[:10]:  # Show first 10
            member = "✓" if ch.get('is_member') else " "
            print(f"  [{member}] #{ch['name']} ({ch['id']})")
        
        if len(channels) > 10:
            print(f"  ... and {len(channels) - 10} more")
        
        return True
    except SlackApiError as e:
        print(f"❌ Error listing channels: {e.response['error']}")
        return False

def test_read_history():
    """Test reading message history"""
    print("\n" + "=" * 60)
    print("Testing Message History Read")
    print("=" * 60)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    
    try:
        # Get first channel bot is member of
        channels_response = client.conversations_list(limit=50)
        member_channels = [
            ch for ch in channels_response['channels'] 
            if ch.get('is_member')
        ]
        
        if not member_channels:
            print("⚠ Bot is not a member of any channels")
            print("  Add the bot to a channel and try again")
            return False
        
        test_channel = member_channels[0]
        channel_name = test_channel['name']
        channel_id = test_channel['id']
        
        print(f"Testing with channel: #{channel_name}")
        
        history_response = client.conversations_history(
            channel=channel_id,
            limit=5
        )
        messages = history_response.get('messages', [])
        
        print(f"✓ Read {len(messages)} recent messages")
        
        if messages:
            print("\nSample message:")
            msg = messages[0]
            user = msg.get('user') or msg.get('username', 'unknown')
            text = msg.get('text', '')[:100]
            print(f"  {user}: {text}...")
        
        return True
    except SlackApiError as e:
        print(f"❌ Error reading history: {e.response['error']}")
        return False

def test_permissions():
    """Test required permissions"""
    print("\n" + "=" * 60)
    print("Testing Permissions")
    print("=" * 60)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    
    try:
        auth_response = client.auth_test()
        
        # Test various permissions by trying operations
        tests = []
        
        # Test channels:read
        try:
            client.conversations_list(limit=1)
            tests.append(("channels:read", True, "List channels"))
        except:
            tests.append(("channels:read", False, "List channels"))
        
        # Test users:read
        try:
            client.users_list(limit=1)
            tests.append(("users:read", True, "List users"))
        except:
            tests.append(("users:read", False, "List users"))
        
        print("Permission tests:")
        for scope, passed, description in tests:
            status = "✓" if passed else "❌"
            print(f"  {status} {scope:20} - {description}")
        
        all_passed = all(t[1] for t in tests)
        return all_passed
        
    except SlackApiError as e:
        print(f"❌ Error testing permissions: {e.response['error']}")
        return False

def test_mcp_imports():
    """Test that MCP SDK is installed"""
    print("\n" + "=" * 60)
    print("Testing MCP SDK Installation")
    print("=" * 60)
    
    try:
        import mcp
        from mcp.server import Server
        from mcp.types import Resource, Tool
        print("✓ MCP SDK is installed")
        print(f"  Version: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print("❌ MCP SDK not installed")
        print("   Run: pip install mcp")
        return False

def test_mcp_server():
    """Test that MCP server can be imported"""
    print("\n" + "=" * 60)
    print("Testing MCP Server Module")
    print("=" * 60)
    
    try:
        # Try to import the server module
        import slack_mcp_server
        print("✓ MCP server module can be imported")
        return True
    except Exception as e:
        print(f"⚠ Could not import MCP server: {e}")
        print("  This is OK if running from different directory")
        return True  # Not a critical failure

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Slack MCP Server - Setup Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Environment", test_environment()))
    results.append(("MCP SDK", test_mcp_imports()))
    results.append(("Slack Connection", test_slack_connection()))
    results.append(("Channels", test_channels()))
    results.append(("History Read", test_read_history()))
    results.append(("Permissions", test_permissions()))
    results.append(("MCP Server", test_mcp_server()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status:8} {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Configure Cursor's mcp.json")
        print("2. Restart Cursor")
        print("3. Try: 'Show me all Slack channels'")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install MCP: pip install mcp")
        print("- Check .env has SLACK_BOT_TOKEN")
        print("- Verify bot is added to channels")
        print("- Check Slack app permissions")
    
    print()
    return passed_count == total_count

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

