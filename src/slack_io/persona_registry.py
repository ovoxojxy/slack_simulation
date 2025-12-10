PERSONAS = {
    "Gabriella_PM": {"username": "Gabriella_PM", "icon": ":memo:", "channels": ["product","announcements"]},
    "Mike_BE":      {"username": "Mike_BE",      "icon": ":gear:", "channels": ["eng-backend","sre-ops","deployments"]},
    "Sarah_FE":     {"username": "Sarah_FE",     "icon": ":art:",  "channels": ["eng-frontend","product","random"]},
    "Kevin_QA":     {"username": "Kevin_QA",     "icon": ":mag:",  "channels": ["qa-testing","eng-backend","eng-frontend","deployments"]},
    "Nina_SRE":     {"username": "Nina_SRE",     "icon": ":helmet_with_white_cross:", "channels": ["sre-ops","deployments","eng-backend"]},
    "Ravi_Staff":   {"username": "Ravi_Staff",   "icon": ":compass:", "channels": ["eng-backend","product","sre-ops"]},
    "Dana_DS":      {"username": "Dana_DS",      "icon": ":bar_chart:", "channels": ["product","eng-backend","eng-frontend"]},
    "Zoey_UX":      {"username": "Zoey_UX",      "icon": ":lipstick:", "channels": ["design-ux","product","eng-frontend"]},
    "Tara_TPM":     {"username": "Tara_TPM",     "icon": ":calendar:", "channels": ["product","announcements","eng-backend","eng-frontend"]},
    "sam_altman":   {"username": "sam_altman",   "icon": ":technologist:", "channels": ["product","announcements","eng-backend","eng-frontend","random"]},
}

# Lightweight policy: who is likely to respond in which channel
CHANNEL_POLICY = {
    "sre-ops":      {"candidates": ["Nina_SRE","Ravi_Staff","Mike_BE","Tara_TPM"], "p_reply": 0.85},
    "eng-backend":  {"candidates": ["Mike_BE","Ravi_Staff","Kevin_QA","Dana_DS","Tara_TPM"], "p_reply": 0.75},
    "eng-frontend": {"candidates": ["Sarah_FE","Zoey_UX","Dana_DS","Tara_TPM"], "p_reply": 0.60},
    "qa-testing":   {"candidates": ["Kevin_QA","Mike_BE","Sarah_FE"], "p_reply": 0.70},
    "product":      {"candidates": ["Gabriella_PM","Tara_TPM","Ravi_Staff","Sarah_FE","Dana_DS","sam_altman"], "p_reply": 0.55},
    "deployments":  {"candidates": ["Nina_SRE","Mike_BE","Kevin_QA","Tara_TPM"], "p_reply": 0.65},
    "announcements":{"candidates": ["Tara_TPM","Gabriella_PM","sam_altman"], "p_reply": 0.25},  # usually proactive
    "design-ux":    {"candidates": ["Zoey_UX","Sarah_FE"], "p_reply": 0.5},
    "random":       {"candidates": ["Sarah_FE"], "p_reply": 0.2},
}

# conductor.py
DEFAULT_POLICY = {"candidates": ["Mike_BE","Kevin_QA","Tara_TPM","Nina_SRE","Sarah_FE","Gabriella_PM","sam_altman"], "p_reply": 0.5}

# Map Slack channel IDs -> short names once at startup
CHANNEL_ID_TO_NAME = {}   # filled at app start
CHANNEL_NAME_TO_ID = {}   # reverse
