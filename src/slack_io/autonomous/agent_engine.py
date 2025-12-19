# src/slack_io/agent_engine.py
import os, re
import numpy as np
from typing import List, Dict, Set
from openai import OpenAI
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from ..shared.slack_client import app as bolt_app

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_AGENT_TEMPERATURE", "0.9"))
OPENAI_PRESENCE_PENALTY = float(os.getenv("OPENAI_AGENT_PRESENCE_PENALTY", "0.6"))
OPENAI_FREQUENCY_PENALTY = float(os.getenv("OPENAI_AGENT_FREQUENCY_PENALTY", "0.3"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_AGENT_TEMPERATURE", "0.9"))
CLAUDE_MAX_OUTPUT_TOKENS = int(os.getenv("CLAUDE_AGENT_MAX_TOKENS", "400"))
MAX_AGENT_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "200"))
ENABLE_EMBEDDING_CHECKS = os.getenv("ENABLE_EMBEDDING_CHECKS", "true").lower() == "true"
USE_EMBEDDINGS = ENABLE_EMBEDDING_CHECKS and LLM_PROVIDER != "claude"

_openai_client = None
_anthropic_client = None
_gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def get_openai_client():
    """Lazy-load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_anthropic_client():
    """Lazy-load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        if Anthropic is None:
            raise ImportError("anthropic package is required when LLM_PROVIDER=claude. Install with `pip install anthropic`.")
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY not set in environment variables")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client

def _get_gemini_model():
    """Lazy-load the Gemini model."""
    global _gemini_model
    if _gemini_model is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    return _gemini_model

def get_embedding(text: str) -> List[float]:
    """Get embedding for text (OpenAI only)."""
    if not USE_EMBEDDINGS:
        raise RuntimeError("Embedding checks are disabled for this provider")
    response = get_openai_client().embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _call_openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = MAX_AGENT_TOKENS) -> str:
    """Call OpenAI chat completion."""
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=OPENAI_TEMPERATURE,
        presence_penalty=OPENAI_PRESENCE_PENALTY,
        frequency_penalty=OPENAI_FREQUENCY_PENALTY,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _call_claude_chat(system_prompt: str, user_prompt: str, max_tokens: int = MAX_AGENT_TOKENS) -> str:
    """Call Anthropic Claude messages API."""
    client = get_anthropic_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=CLAUDE_TEMPERATURE,
        max_tokens=min(max_tokens, CLAUDE_MAX_OUTPUT_TOKENS),
    )
    text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    return "\n".join(part.strip() for part in text_parts if part.strip())


def _call_gemini_chat(
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
) -> str:
    """Call Gemini API for chat completion."""
    model = _get_gemini_model()

    # Gemini doesn't have separate system messages, so prepend to user message
    full_prompt = f"{system_prompt}\n\n{user_message}"

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise


def _call_model(
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
) -> str:
    """Call the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return _call_openai_chat(system_prompt, user_message, temperature, max_tokens)
    elif provider == "claude":
        return _call_claude_chat(system_prompt, user_message, temperature, max_tokens)
    elif provider == "gemini":  # <-- ADD THIS
        return _call_gemini_chat(system_prompt, user_message, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai', 'claude', or 'gemini'")


def call_agent_model(system_prompt: str, user_prompt: str, max_tokens: int = MAX_AGENT_TOKENS) -> str:
    """Public helper for other modules (e.g., seed scheduler) to call the agent LLM."""
    return _call_model(system_prompt, user_prompt, max_tokens=max_tokens)

def detect_new_entities(text: str, prior_texts: List[str]) -> int:
    """Detect if new technical entities are present (simplified heuristic)"""
    # Look for technical patterns: service names, versions, numbers, IDs
    tech_patterns = [r'\d+\.\d+\.\d+',  # versions like 1.2.3
                     r'#\d+',  # PR numbers
                     r'[A-Z]+-\d+',  # JIRA IDs
                     r'[a-z-]+\.(com|io|net)',  # domains
                     r'\d+ms', r'\d+%', r'\d+k',  # metrics
                     r'[A-Z][a-z]+_[A-Z][a-z]+']  # PascalCase entities
    
    text_matches = set()
    for pattern in tech_patterns:
        text_matches.update(re.findall(pattern, text))
    
    prior_matches = set()
    for prior in prior_texts:
        for pattern in tech_patterns:
            prior_matches.update(re.findall(pattern, prior))
    
    # Count unique new entities
    new_entities = text_matches - prior_matches
    return len(new_entities)

# Slack timestamps look like "1730071234.56789" (digits dot digits)
REF_RE = re.compile(r"\[\[ref:(\d{10,}\.\d{1,6})\]\]")   # capture TS values
MAX_CTX = 10

def fetch_recent_context(channel_id: str, thread_ts: str | None, k: int = MAX_CTX) -> List[Dict]:
    """Prefer thread replies if thread_ts is set; else fall back to channel history."""
    cl = bolt_app.client
    msgs = []
    
    try:
        if thread_ts:
            r = cl.conversations_replies(channel=channel_id, ts=thread_ts, limit=50)
            msgs = r.get("messages", [])
        else:
            r = cl.conversations_history(channel=channel_id, limit=50)
            msgs = r.get("messages", [])
    except Exception as e:
        # Handle cases where bot is not in channel (not_in_channel error)
        print(f"[CONTEXT] Could not fetch context from {channel_id}: {e}")
        return []
    
    # newest last; keep non-edit, non-join messages
    out = []
    for m in msgs:
        if m.get("subtype") in {"message_changed", "channel_join", "channel_leave"}:
            continue
        out.append(m)
    # keep the last k user-visible messages
    return out[-k:]

def format_ctx_for_prompt(msgs: List[Dict]) -> str:
    """Format as simple, natural lines."""
    lines = []
    for m in reversed(msgs):  # most recent first
        u = m.get("user") or m.get("username", "user")
        t = (m.get("text") or "").replace("\n", " ")
        lines.append(f"{u}: {t[:200]}")
    return "\n".join(lines)

def extract_4grams(text: str) -> Set[str]:
    """Extract 4-grams from text"""
    words = text.lower().split()
    if len(words) < 4:
        return set()
    return set(tuple(words[i:i+4]) for i in range(len(words) - 3))

def calculate_overlap(grams1: Set[str], grams2: Set[str]) -> float:
    """Calculate percentage overlap between two sets of n-grams"""
    if not grams1 or not grams2:
        return 0.0
    intersection = grams1 & grams2
    union = grams1 | grams2
    return len(intersection) / len(union) if union else 0.0

def check_duplicate(proposed_text: str, recent_msgs: List[Dict], threshold: float = 0.35) -> bool:
    """Check if proposed text shares >threshold 4-grams with recent messages"""
    if len(recent_msgs) == 0:
        return False
    
    proposed_grams = extract_4grams(proposed_text)
    recent_texts = [(m.get("text") or "").replace("\n", " ") for m in recent_msgs[-6:]]
    
    for recent_text in recent_texts:
        recent_grams = extract_4grams(recent_text)
        overlap = calculate_overlap(proposed_grams, recent_grams)
        if overlap > threshold:
            return True
    
    return False

def too_similar(new_txt: str, recent_txts: List[str]) -> bool:
    """Anti-parrot filter: 4-gram Jaccard + cosine on embeddings (either one can block)"""
    def shingles(t):
        """Extract 4-grams as set of tuples"""
        toks = t.lower().split()
        return set(tuple(toks[i:i+4]) for i in range(max(0, len(toks)-3)))
    
    if not recent_txts:
        return False
    
    # Check 4-gram Jaccard similarity
    new_shingles = shingles(new_txt)
    max_jaccard = 0.0
    
    for r in recent_txts:
        if not r.strip():
            continue
        r_shingles = shingles(r)
        intersection = len(new_shingles & r_shingles)
        union = len(new_shingles | r_shingles)
        if union > 0:
            jaccard = intersection / union
            max_jaccard = max(max_jaccard, jaccard)
    
    if max_jaccard > 0.35:
        return True
    
    # Optional: cosine similarity on embeddings (> 0.90 → block)
    if USE_EMBEDDINGS:
        try:
            new_embedding = get_embedding(new_txt)
            for r in recent_txts:
                if not r.strip():
                    continue
                r_embedding = get_embedding(r)
                cosine_sim = cosine_similarity(new_embedding, r_embedding)
                if cosine_sim > 0.90:
                    return True
        except Exception:
            # If embedding fails, continue
            pass
    
    return False

def persona_system_prompt(persona_name: str, persona_cfg: dict) -> str:
    """Build rich persona prompt with role, expertise, and communication style"""
    role = persona_cfg.get("role", "Team Member")
    tone_ticks = ", ".join(persona_cfg.get("tone_ticks", []))
    knowledge = ", ".join(persona_cfg.get("knowledge_domains", []))
    behaviors = ", ".join(persona_cfg.get("behaviors", []))
    seed = "\n".join(f'• "{s}"' for s in persona_cfg.get("seed_snippets", []))
    
    # Voice style based on role (hedges vs decisive)
    decisive_roles = ["Staff Engineer", "TPM"]
    if role in decisive_roles:
        voice_style = "Use decisive, confident language. Make clear recommendations and commitments."
    else:
        voice_style = "Use soft hedges naturally ('I suspect...', 'One idea is...', 'Maybe we should...'). Be collaborative, not prescriptive."
    
    # Personality hints based on role
    personality_hints = {
        "Backend Engineer": "You're technical, direct, and focused on root causes. You think in terms of systems and data.",
        "Product Manager": "You're strategic, stakeholder-focused, and always thinking about business impact and timelines.",
        "QA Tester": "You're methodical, detail-oriented, and focused on reproducibility and verification.",
        "SRE / DevOps": "You're incident-focused, pragmatic, and always thinking about stability and rollback plans.",
        "Frontend Engineer": "You're user-focused, design-aware, and think about accessibility and user experience.",
        "Staff Engineer": "You're senior, architectural, and think about long-term solutions and technical debt.",
        "UX Designer": "You're user-centric, empathetic, and focused on usability and design consistency.",
        "TPM": "You're process-oriented, organized, and focused on coordination and project management.",
        "Data Scientist": "You're analytical, data-driven, and focused on experimentation and user behavior insights."
    }
    
    personality = personality_hints.get(role, "You're professional and collaborative.")
    
    return (
        f"You are {persona_name} ({role}). {personality}\n"
        f"Your expertise: {knowledge}\n"
        f"Your typical contributions: {behaviors}\n"
        f"Voice style: {voice_style}\n"
        f"Communication style: Use these phrases naturally: {tone_ticks}\n"
        f"\nIMPORTANT: Start with a brief acknowledgment, then add one new thing.\n"
        f"Keep messages concise (alternate between short/medium). Use @mentions when relevant.\n"
        f"Only output the message text, no markdown fences.\n\n"
        f"Example messages (emulate this tone):\n{seed}"
    )

def find_relevant_artifacts(context: str, limit: int = 2) -> str:
    """Find relevant artifacts for grounding based on context keywords"""
    from ..legacy.artifacts import search_artifacts
    
    # Extract potential keywords from context
    keywords = context.lower().split()
    # Look for technical terms
    tech_keywords = ['error', 'bug', 'fix', 'pr', 'deploy', 'query', 'sql', 'log', 'decision', 'rollback']
    search_terms = [kw for kw in keywords if any(tech in kw for tech in tech_keywords)]
    
    if not search_terms:
        return ""
    
    # Search for artifacts
    artifacts = []
    for term in search_terms[:3]:  # Try up to 3 search terms
        found = search_artifacts(term, limit=2)
        artifacts.extend(found)
    
    # Deduplicate and limit
    seen = set()
    unique_artifacts = []
    for art in artifacts:
        if art.id not in seen:
            seen.add(art.id)
            unique_artifacts.append(art)
            if len(unique_artifacts) >= limit:
                break
    
    if not unique_artifacts:
        return ""
    
    # Format artifacts for prompt
    artifact_texts = []
    for art in unique_artifacts:
        artifact_texts.append(f"[Relevant Artifact: {art.title}]\n{art.summary}")
    
    return "\n\n".join(artifact_texts)


def build_user_prompt(channel_name: str, event_text: str, ctx_txt: str, is_thread: bool, extra_guidance: str = "", phase_context: str = "", include_artifact_hint: bool = False, artifact_type: str = None) -> str:
    import random
    
    # Add variety with random guidance
    if not extra_guidance:
        guidance_options = [
            "Respond naturally based on your role and expertise.",
            "Add your unique perspective based on your domain knowledge.",
            "Build on previous messages with specific technical details.",
            "Provide actionable next steps relevant to your role.",
            "Share insights that others might not have considered.",
        ]
        extra_guidance = random.choice(guidance_options)
    
    # Add phase context if provided
    phase_str = f"{phase_context}\n\n" if phase_context else ""
    
    # Add strict rules for threaded replies
    strict_rules = ""
    if is_thread:
        strict_rules = (
            "\n\n[STRICT RULES - MUST FOLLOW]:\n"
            "1. Cite at least one prior message via [[ref:TIMESTAMP]] where TIMESTAMP is a Slack timestamp from the recent messages.\n"
            "2. Novelty requirement: Add exactly one of the following: a new datum, a next step, a decision, or a blocking question.\n"
            "Your reply must contain both elements. If missing, regenerate."
        )
    
    # Add artifact hint if requested
    artifact_hint = ""
    if include_artifact_hint and artifact_type:
        artifact_hint = f"\n[OPTIONAL] Consider including a {artifact_type} artifact in triple backticks if relevant to the conversation."
    
    # Find relevant artifacts for grounding
    artifact_grounding = ""
    if ctx_txt:
        relevant_artifacts = find_relevant_artifacts(ctx_txt, limit=2)
        if relevant_artifacts:
            artifact_grounding = f"\n\n[Relevant Artifacts - Hidden Context]:\n{relevant_artifacts}"
    
    return (
        f"Channel: #{channel_name}\n"
        f"{phase_str}Recent messages:\n{ctx_txt}\n"
        f"{artifact_grounding}\n"
        f"Goal: {event_text}\n"
        f"Guidance: {extra_guidance}\n"
        f"{'Reply in this thread.' if is_thread else 'Write a natural message to this channel.'}\n"
        f"Keep it concise (1-3 sentences).{strict_rules}{artifact_hint}"
    )

def _get_role_guidance(role: str) -> str:
    """Get role-specific response guidance"""
    import random
    
    role_cues = {
        "Backend Engineer": [
            "Provide technical analysis from a systems perspective.",
            "Share debugging insights or code-level details.",
            "Suggest technical solutions or architectural considerations.",
        ],
        "QA Tester": [
            "Share testing insights or verification steps.",
            "Report on test results or edge cases.",
            "Provide quality assurance perspective.",
        ],
        "SRE / DevOps": [
            "Focus on operational impact and mitigation strategies.",
            "Share incident response or monitoring insights.",
            "Provide infrastructure or deployment perspective.",
        ],
        "Frontend Engineer": [
            "Consider user experience and frontend implications.",
            "Share UI/UX insights or accessibility considerations.",
            "Provide frontend technical perspective.",
        ],
        "Product Manager": [
            "Focus on business impact and stakeholder communication.",
            "Share product strategy or prioritization insights.",
            "Provide customer impact and roadmap perspective.",
        ],
        "Staff Engineer": [
            "Focus on long-term architectural solutions.",
            "Consider technical debt and scalability.",
            "Provide senior technical guidance.",
        ],
        "TPM": [
            "Focus on timeline and coordination.",
            "Identify risks and blockers.",
            "Track milestones and deliverables.",
        ],
        "UX Designer": [
            "Focus on user experience and usability.",
            "Share design system and accessibility insights.",
            "Consider user journey implications.",
        ],
        "Data Scientist": [
            "Share data-driven insights and analysis.",
            "Focus on experiment results and metrics.",
            "Provide analytical perspective.",
        ],
    }
    
    cues = role_cues.get(role, [
        "Add your unique perspective.",
        "Provide actionable insights.",
        "Share relevant details from your expertise.",
    ])
    
    return random.choice(cues)

def generate_reply(persona_name: str, channel_name: str, channel_id: str, event_text: str, thread_ts: str | None, phase_context: str = "") -> Dict:
    # Import here to avoid circular dependency
    from .persona_registry import PERSONAS
    
    ctx_msgs = fetch_recent_context(channel_id, thread_ts, k=MAX_CTX)
    ctx_txt  = format_ctx_for_prompt(ctx_msgs)
    
    # Get persona config
    persona_cfg = PERSONAS.get(persona_name, {})
    sys = persona_system_prompt(persona_name, persona_cfg)
    
    # Build role-specific guidance
    extra_guidance = _get_role_guidance(persona_cfg.get("role", ""))
    up = build_user_prompt(channel_name, event_text, ctx_txt, is_thread=bool(thread_ts), extra_guidance=extra_guidance, phase_context=phase_context)

    text = _call_model(sys, up, max_tokens=MAX_AGENT_TOKENS)
    
    # Validation: For threaded replies, check if strict rules are followed
    if thread_ts:
        has_reference = "[[ref:" in text
        # Check for novelty: new datum, next step, decision, or blocking question
        novelty_indicators = ["found", "discovered", "next", "step", "decide", "decision", "can't", "unable", "blocked", "need"]
        has_novelty = any(indicator in text.lower() for indicator in novelty_indicators)
        
        if not has_reference or not has_novelty:
            # Retry with even stricter enforcement
            up_strict = up + "\n\n[MANDATORY - REGENERATE WITH BOTH]:\n1. INCLUDE [[ref:TIMESTAMP]] citation from recent messages.\n2. ADD NEW CONTENT: datum, step, decision, or blocking question.\nYour previous attempt was missing required elements."
            text = _call_model(sys, up_strict, max_tokens=MAX_AGENT_TOKENS)
    
    # Anti-parrot filter: check for excessive similarity to recent messages
    if thread_ts and len(ctx_msgs) > 0:
        recent_texts = [(m.get("text") or "").replace("\n", " ") for m in ctx_msgs[-6:]]
        if too_similar(text, recent_texts):
            # Regenerate with explicit forward-motion instruction
            up_retry = up + "\n\nCRITICAL: Avoid repeating earlier wording; move the task forward by adding one concrete next step with an owner and time."
            text = _call_model(sys, up_retry, max_tokens=MAX_AGENT_TOKENS)

    # Semantic similarity check: compare against last 6 messages
    if thread_ts and len(ctx_msgs) > 0:
        prior_texts = [(m.get("text") or "").replace("\n", " ") for m in ctx_msgs[-6:]]
        
        # Check for new entities
        new_entity_count = detect_new_entities(text, prior_texts)
        regenerate = new_entity_count == 0
        
        # Compute semantic similarity if embeddings are enabled
        if USE_EMBEDDINGS:
            try:
                text_embedding = get_embedding(text)
                max_similarity = 0.0
                
                for prior_text in prior_texts:
                    if prior_text.strip():
                        prior_embedding = get_embedding(prior_text)
                        similarity = cosine_similarity(text_embedding, prior_embedding)
                        max_similarity = max(max_similarity, similarity)
                
                if max_similarity > 0.90:
                    regenerate = True
            except Exception as e:
                # If embedding fails, continue with original text
                print(f"[agent_engine] Embedding check failed: {e}")
        
        if regenerate:
            up_semantic = up + "\n\nCRITICAL: Do not repeat prior phrasing; propose a next concrete step."
            text = _call_model(sys, up_semantic, max_tokens=MAX_AGENT_TOKENS)
    
    # Extract citations from text
    citations = REF_RE.findall(text)
    
    # Return text with citations preserved
    return {"text": text, "supports": citations}