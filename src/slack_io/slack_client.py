import time
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt import App
from slack_sdk.errors import SlackApiError
import os, logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

log = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))

def get_persona_user_token(persona: str) -> str:
    """Get user token for a specific persona"""
    if not persona:
        return None
    env_key = f"SLACK_USER_TOKEN_{persona.upper()}"
    return os.getenv(env_key)

def post_message(channel: str, text: str, username: str, icon_emoji: str=None, thread_ts: str=None, persona: str=None):
    # Check if bots are enabled
    bots_enabled = os.getenv("BOTS_ENABLED", "true").lower() == "true"
    if not bots_enabled:
        log.info("Bots are disabled - skipping message post")
        return {"ok": False, "error": "bots_disabled"}
    
    # Try to use persona-specific user token first
    client = app.client
    use_user_token = False
    
    if persona:
        user_token = get_persona_user_token(persona)
        if user_token:
            from slack_sdk import WebClient
            client = WebClient(token=user_token)
            use_user_token = True
            log.info(f"Using user token for persona: {persona}")
    
    args = {
        "channel": channel,
        "text": text,
    }

    # Only add username/icon for bot tokens, not user tokens
    if not use_user_token:
        args["username"] = username
    if icon_emoji:
        args["icon_emoji"] = icon_emoji
    
    if thread_ts:
        args["thread_ts"] = thread_ts
        
    while True:
        try:
            return client.chat_postMessage(**args)
        except SlackApiError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", "1"))
                time.sleep(wait + 0.1)
                continue
            raise

def fetch_history(channel: str, oldest: str=None, latest: str=None, limit: int=200):
    return app.client.conversations.history(channel=channel, oldest=oldest, latest=latest, limit=limit)

def fetch_thread(channel: str, parent_ts: str, limit: int=200):
    return app.client.conversations.replies(channel=channel, ts=parent_ts, limit=limit)
    