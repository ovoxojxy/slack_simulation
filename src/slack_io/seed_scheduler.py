import threading, time, random
from typing import Optional
from .slack_client import app as bolt_app
from .persona_registry import PERSONAS, CHANNEL_NAME_TO_ID
from .conductor import mark_persona_cooldown, schedule_followups_for_thread
from .agent_engine import client as llm_client, MODEL

def _post_root(persona: str, channel_name: str, text:str) -> Optional[str]:
    ch_id = CHANNEL_NAME_TO_ID.get(channel_name)
    if not ch_id:
        return None
    username = PERSONAS[persona]["username"]
    icon = PERSONAS[persona]["icon"]
    
    # Use persona-specific posting (will use user token if available)
    from .slack_client import post_message
    resp = post_message(channel=ch_id, text=text, username=username, icon_emoji=icon, persona=persona)
    
    if not resp.get("ok", True):
        return None
        
    ts = resp["ts"]
    mark_persona_cooldown(persona)
    return ts

def _digest_recent(ch_id: str, limit: int = 12) -> str:
    try:
        r = bolt_app.client.conversations_history(channel=ch_id, limit=limit)
        lines = []
        for m in reversed(r.get("messages", [])):
            u = m.get("user") or m.get("username", "user")
            t = (m.get("text") or "").replace("\n"," ")
            lines.append(f"{u}: {t[:160]}")
        return "\n".join(lines[-12:])
    except Exception:
        return ""
    
def _llm_root(persona: str, channel_name: str, prompt_goal: str, digest: str) -> str:
    sys = (f"You are {persona} posting a new *root* message in #{channel_name} of an internal Slack workspace. "
           f"Be concise (1–3 sentences), concrete, and actionable.")
    user = (f"Goal: {prompt_goal}\n\n"
            f"Recent context (optional):\n{digest}\n\n"
            f"Rules:\n- Start a new thread (no replies).\n- No citations or IDs.\n")
    resp = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.5,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

def standup_loop(minutes: int = 60):
    def run():
        while True:
            try:
                ch_name = random.choice(["eng-backend","eng-frontend","qa-testing","product","deployments","design-ux"])
                ch_id = CHANNEL_NAME_TO_ID.get(ch_name)
                if not ch_id:
                    time.sleep(minutes*60); continue
                persona = random.choice(["Gabriella_PM","Tara_TPM","Mike_BE","Sarah_FE"])
                digest = _digest_recent(ch_id, limit=10)
                text = _llm_root(
                    persona, ch_name,
                    "Kick off a quick standup thread. Ask for blockers and today’s focus.",
                    digest
                )
                ts = _post_root(persona, ch_name, text)
                if ts:
                    # Let 1–2 others reply in the thread
                    schedule_followups_for_thread(ch_name, ch_id, persona, text, thread_ts=ts, max_repliers=2)
            except Exception:
                pass
            time.sleep(minutes*60)
    threading.Thread(target=run, daemon=True).start()

def announcements_loop(minutes: int = 90):
    """Periodically post an announcements summary from PM/TPM."""
    def run():
        while True:
            try:
                ch_name = "announcements"
                ch_id = CHANNEL_NAME_TO_ID.get(ch_name)
                if not ch_id:
                    time.sleep(minutes*60); continue
                persona = random.choice(["Tara_TPM","Gabriella_PM"])
                digest = _digest_recent(ch_id, limit=8)  # you could also pull from product/eng channels
                text = _llm_root(
                    persona, ch_name,
                    "Post a concise status update summarizing key decisions and next steps.",
                    digest
                )
                ts = _post_root(persona, ch_name, text)
                # Typically no immediate follow-ups needed in announcements, but you could add one:
                # if ts: schedule_followups_for_thread(ch_name, ch_id, persona, text, thread_ts=ts, max_repliers=1)
            except Exception:
                pass
            time.sleep(minutes*60)
    threading.Thread(target=run, daemon=True).start()

def noise_loop(every_seconds: int = 20, prob: float = 0.03):
    """Occasional #random post."""
    LINKS = [
        "https://example.com/blog/how-we-cut-p95",
        "https://example.com/meme/latency-cat",
        "https://example.com/paper/caching-tradeoffs",
        "https://example.com/music/lofi"
    ]
    def run():
        while True:
            time.sleep(every_seconds)
            try:
                if random.random() >= prob:
                    continue
                ch_name = "random"
                ch_id = CHANNEL_NAME_TO_ID.get(ch_name)
                if not ch_id:
                    continue
                persona = random.choice(["Sarah_FE"])  # or "NoiseBot" if you use a bot identity
                link = random.choice(LINKS)
                text = f"TIL: {link}"
                ts = _post_root(persona, ch_name, text)
                # usually no follow-ups for noise
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()

def start_seeders(standup_every_min=60, announcements_every_min=90, noise_every_s=20, noise_prob=0.03):
    standup_loop(standup_every_min)
    announcements_loop(announcements_every_min)
    noise_loop(noise_every_s, noise_prob)