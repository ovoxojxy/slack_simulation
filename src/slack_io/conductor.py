# conductor.py
import time, random, json, logging
from typing import Dict, List
from .slack_client import app as bolt_app
from .persona_registry import PERSONAS, CHANNEL_POLICY, CHANNEL_ID_TO_NAME
from .agent_engine import generate_reply
from .queue import ChannelQueue

logger = logging.getLogger(__name__)

# No strict citation requirements - let messages flow naturally

CHANNEL_QUEUES: Dict[str, ChannelQueue] = {}
THREAD_STATE: Dict[str, dict] = {}  # key: thread_ts -> {turns, last_persona, last_ts}
PERSONA_COOLDOWN: Dict[str, float] = {}  # persona -> epoch seconds when they can speak again

# knobs
MAX_TURNS_PER_THREAD = 8
MIN_DELAY_S, MAX_DELAY_S = 1, 3         # Reduced from 2-8s to 1-3s for faster responses
PERSONA_COOLDOWN_S = 12                 # Reduced from 25s to 12s for more activity
SELF_REPLY_GRACE_S = 2                  # Reduced from 4s to 2s for quicker replies
MAX_ACTIVE_THREADS = 8                  # Increased from 5 to 8 for more concurrent threads

# Proactive posting (to create more conversation opportunities)
PROACTIVE_POST_INTERVAL_S = 90  # Reduced from 180s to 90s - check every 1.5 minutes
LAST_PROACTIVE_CHECK = 0  # Last time we checked for proactive posts

def _queue_for(channel_id: str) -> ChannelQueue:
    if channel_id not in CHANNEL_QUEUES:
        CHANNEL_QUEUES[channel_id] = ChannelQueue(bolt_app.client, cooldown=0.8)  # Reduced from 1.1s to 0.8s
    return CHANNEL_QUEUES[channel_id]

def _channel_name(channel_id: str) -> str:
    name = CHANNEL_ID_TO_NAME.get(channel_id)
    if name: return name
    try:
        info = bolt_app.client.conversations_info(channel=channel_id)
        name = info["channel"]["name"]
        CHANNEL_ID_TO_NAME[channel_id] = name
        return name
    except Exception:
        return channel_id

def _eligible_personas(ch_name: str, exclude: List[str]) -> List[str]:
    policy = CHANNEL_POLICY.get(ch_name)
    if not policy:
        return []
    pool = [p for p in policy["candidates"] if p not in exclude]
    now = time.time()
    pool = [p for p in pool if PERSONA_COOLDOWN.get(p, 0) < now]
    return pool

def _update_state(thread_ts: str, persona: str, ts: float):
    st = THREAD_STATE.setdefault(thread_ts, {"turns": 0, "last_persona": None, "last_ts": 0.0})
    st["turns"] += 1
    st["last_persona"] = persona
    st["last_ts"] = ts
    PERSONA_COOLDOWN[persona] = time.time() + PERSONA_COOLDOWN_S

def _count_active_threads(channel_id: str, within_seconds: float = 60) -> int:
    """Count threads that have been active recently in this channel"""
    now = time.time()
    count = 0
    for thread_ts, state in THREAD_STATE.items():
        # Simple heuristic: if thread had activity in last N seconds, it's active
        if state.get("last_ts", 0) > (now - within_seconds):
            # We don't track channel_id in thread state, so this is approximate
            # but sufficient for basic throttling
            count += 1
    return count

def _should_skip(event: dict) -> bool:
    # Avoid infinite loops and bursts
    thread_ts = event.get("thread_ts") or event.get("ts")
    st = THREAD_STATE.get(thread_ts)
    if st and st["turns"] >= MAX_TURNS_PER_THREAD:
        logger.info(f"[CONDUCTOR] Thread {thread_ts} reached max turns ({MAX_TURNS_PER_THREAD})")
        return True
    # If the last post in thread is very recent, back off a bit
    if st and (time.time() - st["last_ts"] < SELF_REPLY_GRACE_S):
        logger.info(f"[CONDUCTOR] Thread {thread_ts} too recent (grace period)")
        return True
    
    # Check if we have too many active threads
    active_threads = _count_active_threads(event.get("channel", ""))
    if active_threads >= MAX_ACTIVE_THREADS:
        logger.info(f"[CONDUCTOR] Too many active threads ({active_threads} >= {MAX_ACTIVE_THREADS})")
        return True
    
    return False

def _fanout_count(ch_name: str) -> int:
    # 1–3 responders based on channel “busyness”
    if ch_name in ("sre-ops","eng-backend","deployments"): return random.choice([1,2,2,3])
    if ch_name in ("product","qa-testing","eng-frontend"): return random.choice([1,2])
    return 1

def maybe_handle_event(event: dict):
    logger.info(f"{'='*60}")
    logger.info(f"[CONDUCTOR] NEW EVENT RECEIVED")
    logger.info(f"[CONDUCTOR] Channel: {event.get('channel')}")
    logger.info(f"[CONDUCTOR] Timestamp: {event.get('ts')}")
    logger.info(f"[CONDUCTOR] Thread TS: {event.get('thread_ts')}")
    logger.info(f"[CONDUCTOR] Text: {event.get('text', '')[:100]}")
    logger.info(f"[CONDUCTOR] Subtype: {event.get('subtype')}")
    logger.info(f"[CONDUCTOR] Username: {event.get('username')}")
    logger.info(f"{'='*60}")
    
    # Periodically check if we should trigger proactive posts
    maybe_trigger_proactive_post()
    
    channel_id = event["channel"]
    ts = event["ts"]
    text = event.get("text","") or ""
    thread_ts = event.get("thread_ts") or ts
    ch_name = _channel_name(channel_id)
    
    # Only skip if this is a known non-persona bot message
    # We WANT personas to be able to reply to each other
    subtype = event.get("subtype")
    username = event.get("username") or ""
    user = event.get("user", "")
    
    # Get list of persona usernames for matching
    persona_usernames = [PERSONAS[p]["username"] for p in PERSONAS]
    
    logger.info(f"[CONDUCTOR] Event details - subtype: {subtype}, username: {username}, user: {user}")
    logger.info(f"[CONDUCTOR] Persona usernames: {persona_usernames}")
    
    # Skip only if this is a bot message from a bot we don't know about
    if subtype == "bot_message":
        # Allow if username matches a persona
        if username in persona_usernames:
            logger.info(f"[CONDUCTOR] Allowing bot message from persona: {username}")
        elif user and user.startswith("B"):  # Slack bot user IDs start with 'B'
            logger.info(f"[CONDUCTOR] Skipping bot message from unknown bot (user: {user})")
            return
        # If no clear indicator, be permissive and allow it
        else:
            logger.info(f"[CONDUCTOR] Ambiguous bot message, allowing: subtype={subtype}, username={username}, user={user}")
    

    # Gate by channel policy & reply probability
    policy = CHANNEL_POLICY.get(ch_name)
    if not policy:
        logger.info(f"[CONDUCTOR] No policy for #{ch_name} ({channel_id})")
        return
    if random.random() > policy["p_reply"]:
        logger.info(f"[CONDUCTOR] Random skip (p_reply threshold)")
        return
    if _should_skip(event):
        return

    # Check if message is from a known persona (to avoid self-replies)
    sender_username = event.get("username") or event.get("user", "")
    sender_is_persona = sender_username and any(PERSONAS[p]["username"] == sender_username for p in PERSONAS)
    
    # If the sender is one of our personas, we need to be more careful
    # Allow replies to persona messages, but exclude the sender from replying
    logger.info(f"[CONDUCTOR] Processing event from {'persona ' + sender_username if sender_is_persona else 'human'} in #{ch_name}")

    # Determine how many personas reply
    n_repliers = _fanout_count(ch_name)

    # Avoid the same persona replying twice in a row
    exclude = []
    st = THREAD_STATE.get(thread_ts)
    if st and st.get("last_persona"):
        exclude.append(st["last_persona"])
    
    # Also exclude the sender if they're a persona
    if sender_is_persona:
        for p in PERSONAS:
            if PERSONAS[p]["username"] == sender_username:
                exclude.append(p)
                break

    eligible = _eligible_personas(ch_name, exclude)
    if not eligible:
        logger.info(f"[CONDUCTOR] No eligible personas for #{ch_name}")
        logger.info(f"[CONDUCTOR] Excluded personas: {exclude}")
        logger.info(f"[CONDUCTOR] Persona cooldowns: {[(p, round(PERSONA_COOLDOWN.get(p, 0) - time.time(), 1)) for p in PERSONAS if PERSONA_COOLDOWN.get(p, 0) > time.time()]}")
        return

    repliers = random.sample(eligible, k=min(n_repliers, len(eligible)))
    logger.info(f"[CONDUCTOR] Selected {len(repliers)} repliers: {repliers}")

    # For each chosen persona, generate + post with small staggered delay
    # Determine if this is a thread reply (thread_ts != ts means it's a reply in a thread)
    is_thread = event.get("thread_ts") is not None
    original_ts = event.get("thread_ts") if is_thread else ts
    
    logger.info(f"[CONDUCTOR] Scheduling {len(repliers)} replies")
    logger.info(f"[CONDUCTOR] is_thread: {is_thread}, original_ts: {original_ts}")
    
    for i, persona in enumerate(repliers):
        delay = random.uniform(MIN_DELAY_S, MAX_DELAY_S) + i * 0.5  # Reduced stagger from 1.2s to 0.5s
        logger.info(f"[CONDUCTOR] Scheduling {persona} with {delay:.1f}s delay, is_thread={is_thread}")
        _schedule_reply(persona, ch_name, channel_id, text, original_ts, delay, is_thread)


def mark_persona_cooldown(persona: str, seconds: float = PERSONA_COOLDOWN_S):
    PERSONA_COOLDOWN[persona] = time.time() + seconds

def schedule_followups_for_thread(ch_name: str, channel_id: str, starter_persona: str, event_text: str, thread_ts: str, max_repliers: int | None = None):

    exclude = [starter_persona]
    eligible = _eligible_personas(ch_name, exclude)
    if not eligible:
        return
    
    n = max_repliers or _fanout_count(ch_name)
    repliers = random.sample(eligible, k=min(n, len(eligible)))

    for i, persona in enumerate(repliers):
        delay =  random.uniform(MIN_DELAY_S, MAX_DELAY_S) + i * 1.2
        _schedule_reply(persona, ch_name, channel_id, event_text, thread_ts, delay, is_thread=True)

def maybe_trigger_proactive_post():
    """Periodically have a persona post something new to keep conversations going"""
    import time as time_module
    global LAST_PROACTIVE_CHECK
    
    now = time_module.time()
    if now - LAST_PROACTIVE_CHECK < PROACTIVE_POST_INTERVAL_S:
        return
    
    LAST_PROACTIVE_CHECK = now
    
    try:
        # Pick a random channel and persona
        active_channels = [ch for ch, pol in CHANNEL_POLICY.items() if pol.get("p_reply", 0) > 0.4]
        if not active_channels:
            return
        
        ch_name = random.choice(active_channels)
        ch_id = CHANNEL_ID_TO_NAME.get(ch_name)
        if not ch_id:
            # Try to get it
            ch_id = _channel_name_inverse(ch_name)
        
        if not ch_id:
            logger.info(f"[CONDUCTOR] Could not get ID for channel {ch_name}")
            return
        
        # Get eligible personas for this channel
        eligible = _eligible_personas(ch_name, [])
        if not eligible:
            logger.info(f"[CONDUCTOR] No eligible personas for proactive post in {ch_name}")
            return
        
        persona = random.choice(eligible)
        
        # Generate a proactive message (e.g., status update, question, observation)
        prompts = [
            "Share a brief status update about your current work",
            "Ask for help or input on something you're working on",
            "Share an observation or insight related to your work",
            "Post a quick update about progress on your tasks",
        ]
        
        digest = _get_recent_digest(ch_id, limit=8)
        prompt = random.choice(prompts)
        
        from .agent_engine import generate_reply
        result = generate_reply(persona, ch_name, ch_id, prompt, thread_ts=None)
        
        # Post it
        username = PERSONAS[persona]["username"]
        icon = PERSONAS[persona]["icon"]
        
        _queue_for(ch_id).enqueue(
            bolt_app.client.chat_postMessage,
            channel=ch_id, text=result["text"], username=username, icon_emoji=icon
        )
        
        mark_persona_cooldown(persona)
        logger.info(f"[CONDUCTOR] {persona} posted proactive message in #{ch_name}")
        
    except Exception as e:
        logger.error(f"[CONDUCTOR] Error in proactive post: {e}", exc_info=True)

def _channel_name_inverse(ch_name: str) -> str:
    """Get channel ID from name"""
    ch_id = CHANNEL_NAME_TO_ID.get(ch_name)
    if ch_id:
        return ch_id
    # Try to look it up
    try:
        for ch_id, cached_name in CHANNEL_ID_TO_NAME.items():
            if cached_name == ch_name:
                return ch_id
        
        # If not found, try the API
        resp = bolt_app.client.conversations_list(types="public_channel,private_channel")
        for ch in resp.get("channels", []):
            if ch["name"] == ch_name:
                CHANNEL_ID_TO_NAME[ch["id"]] = ch["name"]
                CHANNEL_NAME_TO_ID[ch["name"]] = ch["id"]
                return ch["id"]
    except Exception:
        pass
    return None

def _get_recent_digest(ch_id: str, limit: int = 8) -> str:
    """Get recent messages as a digest string"""
    try:
        r = bolt_app.client.conversations_history(channel=ch_id, limit=limit)
        lines = []
        for m in reversed(r.get("messages", [])):
            if m.get("subtype") in {"message_changed", "channel_join", "channel_leave"}:
                continue
            u = m.get("user") or m.get("username", "user")
            t = (m.get("text") or "").replace("\n", " ")
            lines.append(f"{u}: {t[:120]}")
        return "\n".join(lines[-limit:])
    except Exception:
        return ""

def _schedule_reply(persona: str, ch_name: str, channel_id: str, event_text: str, thread_ts: str, delay_s: float, is_thread: bool = False):
    def _do():
        logger.info(f"[CONDUCTOR] {persona} replying {'in thread' if is_thread else 'top-level'} in #{ch_name}")
        
        # generate natural reply
        out = generate_reply(persona, ch_name, channel_id, event_text, thread_ts=thread_ts if is_thread else None)
        visible_text = out["text"]

        # post - only use thread_ts if this is actually a thread
        username = PERSONAS[persona]["username"]
        icon = PERSONAS[persona]["icon"]
        
        # Use persona-specific posting (will use user token if available)
        from .slack_client import post_message
        
        if is_thread:
            _queue_for(channel_id).enqueue(
                post_message,
                channel=channel_id, text=visible_text, username=username, icon_emoji=icon, thread_ts=thread_ts, persona=persona
            )
        else:
            # Top-level reply in channel
            _queue_for(channel_id).enqueue(
                post_message,
                channel=channel_id, text=visible_text, username=username, icon_emoji=icon, persona=persona
            )
        
        _update_state(thread_ts, persona, time.time())
        logger.info(f"[CONDUCTOR] {persona} posted reply")

        # (optional) local provenance log
        try:
            rec = {"t": time.time(), "persona": persona, "chan": ch_name, "thread_ts": thread_ts,
                   "text": visible_text, "supports": supports}
            with open("data/slack_runs_raw.jsonl","a",encoding="utf-8") as f:
                f.write(json.dumps(rec)+"\n")
        except Exception:
            pass

    # crude delay using the queue thread (non-blocking)
    t0 = time.time()
    while time.time() - t0 < delay_s:
        time.sleep(0.2)
    _do()