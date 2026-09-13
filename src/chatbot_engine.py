"""
Chatbot engine for KulTur (BiH heritage sites).

Combines two things the two notebooks already trained:
1. Semantic retrieval over the Chroma vector store ("bih_heritage") built in
   01_Heritage_Knowledge_Base.ipynb - matches free-text questions to sites by
   meaning (description / keywords / fun facts), and now also carries the
   cluster_micro / cluster_macro spatial-corridor labels (see
   rag_cluster_integration.py).
2. The RandomForestClassifier + LabelEncoders trained in
   02_Preference_Simulation_Model Training.ipynb - re-ranks the semantic
   candidates by how well they match a user's stated interest/season/region/
   time-budget.

This is intentionally dependency-light (no external LLM call) so it runs
fully offline/free: it's "retrieval + learned re-ranking", not text generation.
"""
from __future__ import annotations

import difflib
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = DATA_DIR / "heritage_db"
COLLECTION_NAME = "bih_heritage"
BRAIN_PATH = DATA_DIR / "heritage_brain_v2.pkl"

INTERESTS = ["Any", "History", "Mystery", "Nature", "Quick Stop", "Culture & Urban"]
SEASONS = ["Any", "Spring", "Summer", "Autumn", "Winter"]

# Maps the recommender's coarse "interest" buckets to the raw site categories
# in the dataset, used only as a graceful fallback if the pickled model can't
# be scored for a given site's category (e.g. brand-new category not seen in training).
_INTEREST_TO_CATEGORIES = {
    "History": {"monument", "fortress", "museum", "archaeological", "modern history", "national monument"},
    "Nature": {"nature", "cultural landscape", "natural"},
    "Mystery": {"hidden gem"},
    "Culture & Urban": {"urban heritage", "religious", "building", "traditional village"},
    "Quick Stop": set(),  # handled via visit_time_num instead of category
}

# Free-text keywords used to detect interests the user mentions in the chat
# itself (e.g. "I like nature too"), so the sidebar selection isn't the only
# signal the recommender uses.
_INTEREST_KEYWORDS = {
    "History": ["history", "historical", "ancient", "medieval", "heritage", "old town"],
    "Nature": ["nature", "natural", "outdoor", "hike", "hiking", "lake", "waterfall", "mountain", "forest", "river"],
    "Mystery": ["mystery", "mysterious", "legend", "hidden", "secret", "myth"],
    "Culture & Urban": ["culture", "urban", "city", "bazaar", "market", "museum", "art"],
    "Quick Stop": ["quick", "short on time", "not much time", "little time"],
}

# Verb phrases that turn a bare topic keyword into an explicit, persistent
# preference statement (e.g. "I'm into nature", "I like history") - stronger
# than just mentioning the word in passing (e.g. "what nature spots are
# there?"), so slot-filling into the conversation profile doesn't over-trigger.
_EXPLICIT_PREFERENCE_VERBS = [
    "i'm into", "im into", "i am into", "i like", "i love", "i enjoy", "i prefer",
    "i'm interested in", "im interested in", "i am interested in", "i'm a fan of", "im a fan of",
]

# Extra signals (beyond interest keywords) that a message is actually asking
# for travel suggestions rather than just chatting.
_TRAVEL_ACTION_WORDS = [
    "recommend", "suggest", "visit", "place", "site", "trip", "go to", "explore",
    "things to do", "itinerary", "plan", "nearby", "around", "near",
]

# Canned replies for messages that aren't travel requests, so the bot can
# hold a basic conversation instead of always dumping recommendations.
_GREETING_WORDS = ["zdravo", "hi", "hello", "hey", "howdy", "yo", "good morning", "good afternoon", "good evening"]
_HOWAREYOU_WORDS = [
    "how are you", "how are you doing", "kako si", "how's it going", "hows it going",
    "what's up", "whats up", "what are you doing", "what are you up to", "whatcha doing",
    "what you doing", "are you ok", "are you okay", "you doing ok", "you alright", "you good",
]
_THANKS_WORDS = ["thank", "thanks", "hvala"]
# "bye"/"goodbye"/"see you" are safe for fuzzy substring matching; very short
# words like "cao"/"ćao" are NOT - at typo-tolerant cutoffs they can match
# random unrelated words (e.g. "can"), so they're only matched as a
# near-exact *whole* message (see _BYE_WHOLE_WORDS below).
_BYE_WORDS = ["bye", "goodbye", "see you"]
_BYE_WHOLE_WORDS = {"cao", "ćao"}
_CAPABILITY_WORDS = ["what can you do", "who are you", "help me", "what is this", "how does this work"]
# Meta questions about the bot's own reasoning - checked BEFORE travel-intent
# detection, since phrases like "why do you think I'm into history" would
# otherwise get misread as a new travel request just because they contain
# an interest keyword.
_META_QUESTION_WORDS = [
    "why do you think", "why did you", "why do you", "why is that", "why history",
    "explain why", "how do you know", "based on what", "why these", "why did it",
]
# Short standalone utterances - matched on the *whole* message (not substring)
# so they don't misfire inside longer travel questions.
_ACK_WORDS = {"cool", "nice", "awesome", "great", "perfect", "sweet", "ok", "okay", "alright", "sounds good", "cool thanks"}
_STOP_WORDS = {"no", "stop", "no stop", "nope", "never mind", "nevermind", "cancel", "that's enough", "thats enough", "no more"}

# Phrases like "aren't these really far?" - a pushback on a *previous*
# recommendation's distance, answered using chat history rather than the
# generic "unclear" fallback.
_DISTANCE_COMPLAINT_WORDS = [
    "too far", "aren't these far", "arent these far", "isn't that far", "isnt that far",
    "that's far", "thats far", "so far away", "too much driving", "too much travel",
    "not close", "quite far", "really far", "far away",
]

# Explicit signals that the user is fine with a wider search radius (car,
# road trip, etc.) - bypasses the strict same-corridor location filter.
_WIDE_RANGE_WORDS = [
    "car", "driving", "drive", "road trip", "willing to travel", "regional",
    "whole region", "further away", "further trip", "farther trip", "day trip",
    "anywhere in", "don't mind traveling", "dont mind traveling",
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.009
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def _fuzzy_phrase_match(text: str, phrases, cutoff: float = 0.65) -> bool:
    """Typo-tolerant phrase matching: exact substring first, then a light
    per-word fuzzy fallback (e.g. "hpw are you doing" still matches "how are
    you doing") using difflib - no external NLP dependency needed.
    """
    lower = text.lower().strip().rstrip("!.?,")
    if any(p in lower for p in phrases):
        return True
    tokens = lower.split()
    for phrase in phrases:
        words = phrase.split()
        if all(any(difflib.SequenceMatcher(None, t, w).ratio() >= cutoff for t in tokens) for w in words):
            return True
    return False


def _fuzzy_whole_match(text: str, whole_phrases, cutoff: float = 0.75) -> bool:
    """For short standalone utterances ("cool", "stop", ...) matched against
    the *entire* stripped message, with a typo-tolerant fallback.
    """
    lower = text.lower().strip().rstrip("!.?,")
    if lower in whole_phrases:
        return True
    return bool(difflib.get_close_matches(lower, whole_phrases, n=1, cutoff=cutoff))

CONVERSATIONAL_REPLIES = {
    "howareyou": [
        "I'm doing great, thanks for asking! 😊 What part of Bosnia & Herzegovina are you curious about - history, nature, food, or something more mysterious?",
        "Doing well, thanks! Are you thinking history, nature, a hidden gem, or something quick?",
        "All good here! What kind of place are you in the mood for - old towns, mountains, legends?",
    ],
    "greeting": [
        "Zdravo! How can I help you explore Bosnia & Herzegovina today?",
        "Zdravo! Looking for somewhere historic, natural, or a bit mysterious?",
        "Hey! Ready to explore Bosnia & Herzegovina - what are you in the mood for?",
    ],
    "thanks": [
        "You're welcome! Let me know if you'd like more suggestions.",
        "Anytime! Happy to find more places if you want.",
        "No problem at all - just ask if you want another idea.",
    ],
    "bye": [
        "Bok! Have a wonderful trip around Bosnia & Herzegovina.",
        "Bok! Safe travels, and come back if you need more ideas.",
    ],
    "capability": [
        (
            "I can recommend heritage sites, hidden gems, and nature spots across Bosnia & Herzegovina "
            "based on your interests, the season, and how much time you have. Just tell me what you're in the mood for, "
            "or use the 'Suggest based on my profile' shortcut in the sidebar."
        ),
    ],
    "ack": [
        "Glad you like it! Want more suggestions, or something different?",
        "Nice! Want another one, or a different vibe altogether?",
    ],
    "stop": [
        "No problem, I'll pause the suggestions. Just ask whenever you're ready again.",
        "Got it, pausing for now - I'm here whenever you want more ideas.",
    ],
    "unclear": [
        (
            "Not sure I caught that - are you after travel suggestions (history, nature, mystery, a quick stop...), "
            "or just chatting? Let me know what you're in the mood for and I'll find some places."
        ),
        "Hmm, not sure what you mean - want some place suggestions, or just chatting?",
    ],
    "meta_why": [
        (
            "I rank sites using your sidebar 'Main interest'/season/region/time settings plus any hints you "
            "mention in chat - feel free to tweak the dropdowns anytime, or just tell me a different vibe."
        ),
    ],
    "distance_pushback": [
        (
            "Fair point - I try to keep suggestions in the same spatial corridor as the place you mentioned. "
            "Want me to narrow it to only the very closest option, or widen the search (just mention you're driving)?"
        ),
    ],
}


def get_conversational_reply(intent: str) -> str:
    """Pick one of the varied canned replies for an intent (avoids repeating
    the exact same sentence every time, without needing a real LLM).
    """
    return random.choice(CONVERSATIONAL_REPLIES[intent])


# ---- Optional Groq LLM for dynamic chit-chat/meta/off-topic replies -------
# Casual, meta, or off-topic messages ("what's up", "aren't you an LLM?",
# "why do you keep repeating yourself") are routed here instead of the static
# templates above, so replies feel natural and non-repetitive. This never
# touches the RAG retrieval or RandomForest re-ranking used for actual travel
# requests - it's purely for the non-recommendation conversational branch.
try:
    from groq import Groq
except ImportError:  # pragma: no cover - groq is an optional dependency
    Groq = None

GROQ_MODEL = "openai/gpt-oss-120b"

_GROQ_SYSTEM_PROMPT = (
    "You are the conversational voice of 'KulTur', a chatbot that recommends heritage "
    "sites, nature spots, and hidden gems across Bosnia & Herzegovina. The user's message is casual, "
    "meta, or off-topic (not a travel request), so just chat naturally - 1-3 sentences, friendly, and "
    "varied - don't repeat earlier replies in this conversation. If it fits, you can invite them to ask "
    "about places to visit, but don't force it every time. If asked whether you're an LLM or why you "
    "repeat yourself, answer honestly: you're a hybrid assistant - simple rules decide whether a message "
    "needs travel recommendations (handled by a retrieval + machine-learning re-ranker) or chit-chat "
    "(handled by you, the LLM)."
)


def get_groq_client(api_key: str | None = None):
    """Build a Groq client if a key is available; otherwise None so callers fall back to templates."""
    if Groq is None or not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def generate_llm_reply(
    client,
    intent: str,
    user_message: str,
    history: list[dict] | None = None,
    context_note: str | None = None,
) -> str:
    """Ask Groq for a natural chit-chat reply; fall back to a canned template on any failure."""
    if client is None:
        return get_conversational_reply(intent)
    try:
        messages = [{"role": "system", "content": _GROQ_SYSTEM_PROMPT}]
        if context_note:
            messages.append({"role": "system", "content": context_note})
        for turn in (history or [])[-6:]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_message})
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=200,
        )
        text = _trim_to_last_complete_sentence((completion.choices[0].message.content or "").strip())
        return text or get_conversational_reply(intent)
    except Exception:
        return get_conversational_reply(intent)


_RAG_SYSTEM_PROMPT = (
    "You are the conversational voice of 'KulTur'. A retrieval system has already found "
    "and ranked real heritage sites from a database for the user's request - they are listed below as "
    "'Retrieved sites'. Write a warm, natural reply (a short paragraph or a light list, 3-6 sentences "
    "total) that presents the top options. Use ONLY the facts given - do not invent sites, regions, or "
    "facts that aren't listed. STRICT RULE ABOUT DISTANCES: never state or estimate a specific distance "
    "(km, meters, miles, 'a short walk', 'a few minutes away', etc.) for a site unless that exact distance "
    "figure is explicitly present in the 'Retrieved sites' data below - if no distance is given for a site, "
    "simply don't mention distance for it at all. Never guess or fabricate distances in kilometers if they "
    "are not explicitly provided in the retrieved context. If a site lists 'Nearby corridor' sites, and the "
    "user is open to a wider/further trip (mentions a car, driving, or a further/farther trip), prefer "
    "recommending from that nearby corridor list over jumping to an unrelated, distant region - the corridor "
    "sites are geographically close to what the user already asked about. Weave in why each fits (from the "
    "'why' field), and end by inviting a follow-up (e.g. narrowing down, or asking for something different). "
    "Always finish your reply with a complete sentence - never trail off mid-word or mid-thought. Do not start "
    "with a robotic phrase like 'Here's what I'd suggest for X:' - just talk naturally, like a knowledgeable "
    "local friend giving tips."
)


def _trim_to_last_complete_sentence(text: str) -> str:
    """Guard against a response getting cut off mid-word/mid-sentence (e.g. hitting
    max_tokens) by trimming back to the last sentence-ending punctuation, if any.
    """
    text = text.strip()
    if not text or text[-1] in ".!?\u201d\u2019\"'":
        return text
    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_end == -1:
        return text
    # Keep a trailing closing quote/bracket right after the punctuation, if any.
    end = last_end + 1
    while end < len(text) and text[end] in "\u201d\u2019\"')":
        end += 1
    trimmed = text[:end].strip()
    return trimmed or text


def _format_results_for_llm(results: list[SiteResult], nearby_by_site: dict[str, list[str]] | None = None) -> str:
    lines = []
    nearby_by_site = nearby_by_site or {}
    for i, r in enumerate(results, start=1):
        distance_bit = f", about {r.distance_km:.0f} km away" if r.distance_km is not None else ""
        nearby = nearby_by_site.get(r.name) or []
        nearby_bit = f" Nearby corridor: {', '.join(nearby)}." if nearby else ""
        lines.append(
            f"{i}. {r.name} ({r.region}, category: {r.category}){distance_bit}. "
            f"Description: {r.description} Fun fact: {r.fun_fact} Why recommended: {r.reason}.{nearby_bit}"
        )
    return "\n".join(lines)


def format_recommendation_fallback(query_text: str, results: list[SiteResult]) -> str:
    """Plain-template rendering of results - used when no Groq key/call is available."""
    if not results:
        return "I couldn't find a good match for that - try mentioning a region, a vibe (history/nature/mystery), or a site name."
    lines = [f"Here's what I'd suggest for *{query_text}*:\n"]
    for i, r in enumerate(results, start=1):
        distance_bit = f" ({r.distance_km:.0f} km away)" if r.distance_km is not None else ""
        lines.append(
            f"**{i}. {r.name}**{distance_bit} ({r.region} · {r.category})\n"
            f"   {r.description}\n"
            f"   *Why:* {r.reason}\n"
            f"   💡 {r.fun_fact}\n"
        )
    lines.append("_See the sidebar 'Match details' dropdown for confidence scores and nearby sites._")
    return "\n".join(lines)


def synthesize_recommendation_reply(
    client,
    query_text: str,
    results: list[SiteResult],
    history: list[dict] | None = None,
    engine: "HeritageChatbotEngine | None" = None,
) -> str:
    """Have Groq write the recommendation reply using only the grounded/retrieved
    site data (RAG-style synthesis) - falls back to the plain template if the
    LLM is unavailable or the call fails for any reason. If `engine` is given,
    each site's same-corridor neighbors are looked up and passed along so the
    LLM can prefer them for "wider/further trip" follow-ups instead of jumping
    to an unrelated region.
    """
    fallback = format_recommendation_fallback(query_text, results)
    if client is None or not results:
        return fallback
    try:
        nearby_by_site = (
            {r.name: engine.find_nearby(r.name, scale="micro", top_n=3) for r in results} if engine else None
        )
        facts_block = _format_results_for_llm(results, nearby_by_site)
        messages = [
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {"role": "system", "content": f"Retrieved sites (grounded facts, in rank order):\n{facts_block}"},
        ]
        for turn in (history or [])[-6:]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": query_text})
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        text = _trim_to_last_complete_sentence((completion.choices[0].message.content or "").strip())
        return text or fallback
    except Exception:
        return fallback


@dataclass
class SiteResult:
    name: str
    region: str
    category: str
    historical_period: str
    description: str
    fun_fact: str
    tourism_season: str
    avg_visit_time: str
    cluster_micro: int
    cluster_macro: int
    score: float
    reason: str
    distance_km: float | None = None
    matched_interests: list[str] = field(default_factory=list)
    text_detected_interests: list[str] = field(default_factory=list)


class HeritageChatbotEngine:
    """Loads once (cached by Streamlit via st.cache_resource) and answers queries."""

    def __init__(self) -> None:
        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(f"Vector DB not found at {CHROMA_DB_PATH}")
        if not BRAIN_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {BRAIN_PATH}")

        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_collection(COLLECTION_NAME)

        brain = joblib.load(BRAIN_PATH)
        self.model = brain["model"]
        self.le_interest = brain["le_interest"]
        self.le_season = brain["le_season"]
        self.le_region = brain["le_region"]
        self.le_category = brain["le_category"]
        self.le_period = brain["le_period"]
        self.le_pop = brain["le_pop"]

        # Pull all metadata once - the dataset is small (a few hundred sites).
        all_docs = self.collection.get(include=["metadatas"])
        self.metadata_by_name = {m["name"]: m for m in all_docs["metadatas"]}
        self.all_regions = sorted({m.get("region", "Any") for m in all_docs["metadatas"]})

        # Derived (not hardcoded) municipality/location keywords from the
        # dataset's own 'location' field, e.g. "olovo", "mostar", "banja luka" -
        # lets a message like "I'm around Olovo" register as travel intent
        # even without an explicit action word like "visit", and doubles as an
        # index to resolve a mentioned place back to its coordinates/cluster.
        location_words: set[str] = set()
        location_to_metas: dict[str, list[dict]] = {}
        for m in all_docs["metadatas"]:
            loc = str(m.get("location", "")).strip().lower()
            if loc:
                for part in re.split(r"[^a-zšđčćž]+", loc):
                    if len(part) > 3:
                        location_words.add(part)
                        location_to_metas.setdefault(part, []).append(m)
        self.all_locations = sorted(location_words)
        self._location_to_metas = location_to_metas

        # PyTorch's very first inference call pays a one-time backend/JIT
        # warm-up cost (tens of seconds). Pay it here, during engine
        # construction (once, cached by Streamlit), instead of during the
        # user's first chat message.
        self.collection.query(query_texts=["warm up"], n_results=1)

    # ---- Conversation vs. recommendation intent ---------------------------

    def detect_interests_in_text(self, text: str) -> set[str]:
        """Public wrapper so callers (e.g. the UI) can explain what was detected."""
        return self._detect_interests_in_text(text)

    def _has_travel_intent(self, text: str) -> bool:
        lower = text.lower()
        if self._detect_interests_in_text(text):
            return True
        if any(word in lower for word in _TRAVEL_ACTION_WORDS):
            return True
        if any(region.lower() in lower for region in self.all_regions):
            return True
        if any(loc in lower for loc in self.all_locations):
            return True
        return any(name.lower() in lower for name in self.metadata_by_name)

    def _wants_wide_range(self, text: str) -> bool:
        lower = text.lower()
        return any(w in lower for w in _WIDE_RANGE_WORDS)

    def wants_wide_range(self, text: str) -> bool:
        """Public wrapper so callers (e.g. the UI) can detect 'I have a car' etc."""
        return self._wants_wide_range(text)

    def find_location_reference(self, text: str) -> dict | None:
        """Resolve a mentioned place (e.g. "Olovo") to a reference point: its
        average coordinates and dominant micro-corridor, derived entirely from
        the dataset (no hardcoded city list).
        """
        lower = text.lower()
        matches = [loc for loc in self.all_locations if loc in lower]
        if not matches:
            return None
        best = max(matches, key=len)
        metas = self._location_to_metas.get(best, [])
        if not metas:
            return None
        lats = [m["latitude"] for m in metas if m.get("latitude") is not None]
        lons = [m["longitude"] for m in metas if m.get("longitude") is not None]
        clusters = [m.get("cluster_micro", -1) for m in metas]
        dominant_cluster = max(set(clusters), key=clusters.count) if clusters else -1
        return {
            "location": best,
            "lat": sum(lats) / len(lats) if lats else None,
            "lon": sum(lons) / len(lons) if lons else None,
            "cluster_micro": dominant_cluster,
        }

    def classify_smalltalk(self, text: str) -> str | None:
        """Return a key into CONVERSATIONAL_REPLIES if this looks like plain
        conversation (greeting, thanks, a meta question, ...) rather than a
        travel request, otherwise None (meaning: go ahead and recommend places).
        """
        lower = text.lower().strip().rstrip("!.?")

        # Checked first: a meta question like "why do you think I'm into
        # history?" would otherwise be misread as a new travel request just
        # because it contains an interest keyword.
        if any(p in lower for p in _META_QUESTION_WORDS):
            return "meta_why"

        # A pushback on a previous recommendation's distance - answer using
        # chat history instead of falling through to "unclear". Exact
        # substring match only (no fuzzy tolerance): short words like "far"
        # are too easily confused with unrelated short words (e.g. "car") at
        # typo-tolerant cutoffs.
        if any(p in lower for p in _DISTANCE_COMPLAINT_WORDS):
            return "distance_pushback"

        if self._has_travel_intent(text):
            return None
        if _fuzzy_whole_match(text, _STOP_WORDS):
            return "stop"
        if _fuzzy_whole_match(text, _ACK_WORDS):
            return "ack"
        if _fuzzy_phrase_match(text, _HOWAREYOU_WORDS):
            return "howareyou"
        if _fuzzy_phrase_match(text, _THANKS_WORDS):
            return "thanks"
        if _fuzzy_whole_match(text, _BYE_WHOLE_WORDS, cutoff=0.85) or _fuzzy_phrase_match(text, _BYE_WORDS):
            return "bye"
        if _fuzzy_phrase_match(text, _CAPABILITY_WORDS):
            return "capability"
        if any(lower == w or lower.startswith(w + " ") or lower.startswith(w + "!") or lower.startswith(w + ",") for w in _GREETING_WORDS):
            return "greeting"
        # No travel signal and no recognized smalltalk pattern - ask for
        # clarification instead of silently dumping recommendations.
        return "unclear"

    # ---- Recommender scoring -------------------------------------------------

    def _safe_encode(self, encoder, value: str, fallback_index: int = 0) -> int:
        """LabelEncoder.transform() raises on unseen labels; fall back gracefully."""
        try:
            return int(encoder.transform([value])[0])
        except ValueError:
            return fallback_index

    def _predict_batch(self, rows: list[dict]) -> np.ndarray:
        # Column order must exactly match model.feature_names_in_ (verified via
        # joblib.load(...)["model"].feature_names_in_).
        features = pd.DataFrame(rows)[
            [
                "user_interest",
                "current_season",
                "user_region",
                "limited_time",
                "site_region",
                "site_category",
                "site_pop",
                "period",
            ]
        ]
        proba = self.model.predict_proba(features)
        # column 1 = probability the model would "recommend" (label=1)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    def _detect_interests_in_text(self, text: str) -> set[str]:
        lower = text.lower()
        return {interest for interest, keywords in _INTEREST_KEYWORDS.items() if any(k in lower for k in keywords)}

    def _detect_explicit_interest_statements(self, text: str) -> set[str]:
        lower = text.lower()
        if not any(v in lower for v in _EXPLICIT_PREFERENCE_VERBS):
            return set()
        return self._detect_interests_in_text(text)

    def detect_explicit_interest_statements(self, text: str) -> set[str]:
        """Slot-filling entry point: only fires on an explicit preference
        statement ("I'm into nature", "I like history", "I love mystery
        legends", ...), not just a passing topic mention in a question - so
        callers can persist it into the conversation profile with confidence
        that the user actually stated it, not just referenced it in passing.
        """
        return self._detect_explicit_interest_statements(text)

    def _resolve_interests(
        self, interest: str, query_text: str, known_interests: set[str] | None = None
    ) -> tuple[list[str], bool, set[str]]:
        """Combine the sidebar interest, anything mentioned in the chat text this
        turn, and any interests accumulated from earlier turns in the same
        conversation (`known_interests`, passed in by the caller - never hardcoded
        here) - so a preference stated once keeps biasing the re-ranker even if
        the user doesn't repeat it every message.

        Returns (interests_to_score, use_max, text_detected). When the user has an
        explicit preference (sidebar choice, this message, and/or earlier chat
        turns), we take the *best* match among those interests ("either/or"
        semantics). With no preference at all ("Any" + nothing detected anywhere),
        we average across every interest for a neutral score. `text_detected` is
        strictly what was found in *this* query_text - never the sidebar choice or
        earlier turns - so callers can tell a fresh chat-stated preference apart
        from carried-over context or the neutral every-interest fallback.
        """
        text_detected = self._detect_interests_in_text(query_text)
        explicit = set(text_detected) | set(known_interests or set())
        if interest != "Any":
            explicit.add(interest)
        if explicit:
            return sorted(explicit), True, text_detected
        return INTERESTS[1:], False, text_detected

    def _score_candidates(
        self,
        metas: list[dict],
        interest: str,
        season: str,
        region_pref: str,
        limited_time: bool,
        query_text: str,
        known_interests: set[str] | None = None,
    ) -> tuple[list[float], list[str], set[str]]:
        interests_to_try, use_max, text_detected = self._resolve_interests(interest, query_text, known_interests)
        seasons_to_try = SEASONS[1:] if season == "Any" else [season]

        rows: list[dict] = []
        owner: list[int] = []
        for cand_idx, meta in enumerate(metas):
            region_value = region_pref if region_pref != "Any" else meta.get("region", "Any")
            for i in interests_to_try:
                for s in seasons_to_try:
                    rows.append(
                        {
                            "user_interest": self._safe_encode(self.le_interest, i),
                            "current_season": self._safe_encode(self.le_season, s),
                            "user_region": self._safe_encode(self.le_region, region_value),
                            "limited_time": int(limited_time),
                            "site_region": self._safe_encode(self.le_region, meta.get("region", "")),
                            "site_category": self._safe_encode(self.le_category, meta.get("category", "")),
                            "period": self._safe_encode(self.le_period, meta.get("historical_period", "")),
                            "site_pop": meta.get("pop_num", 1.0),
                        }
                    )
                    owner.append(cand_idx)

        probs = self._predict_batch(rows) if rows else np.array([])

        buckets: list[list[float]] = [[] for _ in metas]
        for cand_idx, p in zip(owner, probs):
            buckets[cand_idx].append(p)

        reduce_fn = max if use_max else (lambda xs: float(np.mean(xs)))
        scores = [reduce_fn(b) if b else 0.0 for b in buckets]
        return scores, interests_to_try, text_detected

    # ---- Public API ------------------------------------------------------

    def semantic_candidates(self, query_text: str, top_n: int = 20) -> list[dict]:
        metas, _ = self.semantic_candidates_with_distance(query_text, top_n=top_n)
        return metas

    def semantic_candidates_with_distance(self, query_text: str, top_n: int = 20) -> tuple[list[dict], list[float]]:
        result = self.collection.query(query_texts=[query_text], n_results=top_n)
        metas = result["metadatas"][0] if result["metadatas"] else []
        dists = result["distances"][0] if result.get("distances") else [1.0] * len(metas)
        return metas, dists

    def ask(
        self,
        query_text: str,
        interest: str = "Any",
        season: str = "Any",
        region_pref: str = "Any",
        limited_time: bool = False,
        min_score: float = 0.0,
        max_results: int = 10,
        known_interests: set[str] | None = None,
    ) -> list[SiteResult]:
        candidates, distances = self.semantic_candidates_with_distance(query_text, top_n=25)
        if not candidates:
            candidates = list(self.metadata_by_name.values())
            distances = [1.0] * len(candidates)

        # Location filter: if the user mentioned a place (e.g. "Olovo"),
        # restrict candidates to the same micro-corridor - falling back to a
        # tight straight-line radius if that corridor is empty/noise. Mentioning
        # a car/road trip doesn't drop the corridor preference entirely - it
        # just widens the fallback radius, so a "further trip" still prioritizes
        # the same nearby corridor over a random jump to an unrelated region.
        location_ref = self.find_location_reference(query_text)
        wide_range = self._wants_wide_range(query_text)
        if location_ref:
            paired = list(zip(candidates, distances))
            radius_km = 50.0 if wide_range else 25.0
            same_cluster = []
            if location_ref["cluster_micro"] != -1:
                same_cluster = [(m, d) for m, d in paired if m.get("cluster_micro") == location_ref["cluster_micro"]]
            if same_cluster:
                paired = same_cluster
            elif location_ref["lat"] is not None:
                nearby = [
                    (m, d)
                    for m, d in paired
                    if m.get("latitude") is not None
                    and m.get("longitude") is not None
                    and _haversine_km(location_ref["lat"], location_ref["lon"], m["latitude"], m["longitude"]) <= radius_km
                ]
                if nearby:
                    paired = nearby
            candidates, distances = [m for m, _ in paired], [d for _, d in paired]

        raw_scores, interests_used, text_detected = self._score_candidates(
            candidates, interest, season, region_pref, limited_time, query_text, known_interests
        )

        scored: list[SiteResult] = []
        for meta, chroma_distance, rf_score in zip(candidates, distances, raw_scores):
            # Hybrid score: 40% semantic similarity + 40% learned preference
            # match - 20% distance penalty (only applied when a place was
            # actually mentioned; otherwise the penalty term is zero).
            vector_similarity = 1.0 / (1.0 + max(chroma_distance, 0.0))
            distance_km = None
            distance_multiplier = 0.0
            if location_ref and location_ref["lat"] is not None and meta.get("latitude") is not None:
                distance_km = _haversine_km(location_ref["lat"], location_ref["lon"], meta["latitude"], meta["longitude"])
                distance_multiplier = min(distance_km / 100.0, 1.0)
            final_score = max(0.0, 0.4 * vector_similarity + 0.4 * rf_score - 0.2 * distance_multiplier)

            matched_interests = [
                i for i in interests_used if meta.get("category") in _INTEREST_TO_CATEGORIES.get(i, set())
            ]
            # Only interests the user actually typed in *this* message - never the
            # sidebar choice, and never the neutral "evaluate every interest"
            # fallback used when nothing was said - so chat-inferred profile
            # tracking can't be contaminated by scoring internals.
            text_detected_interests = [i for i in matched_interests if i in text_detected]
            reason_bits = (
                [f"since you're into {' / '.join(matched_interests)}, this fits well"]
                if matched_interests
                else ["semantically related to your question"]
            )
            if region_pref != "Any" and meta.get("region") == region_pref:
                reason_bits.append(f"located in {region_pref}")
            if season != "Any" and meta.get(f"is_{season.lower()}") == 1:
                reason_bits.append(f"open/best in {season}")
            if distance_km is not None:
                reason_bits.append(f"~{distance_km:.0f} km from {location_ref['location'].title()}")
            scored.append(
                SiteResult(
                    name=meta.get("name", "Unknown"),
                    region=meta.get("region", "Unknown"),
                    category=meta.get("category", "Unknown"),
                    historical_period=meta.get("historical_period", "Unknown"),
                    description=meta.get("description", ""),
                    fun_fact=meta.get("fun_fact", ""),
                    tourism_season=meta.get("tourism_season", ""),
                    avg_visit_time=meta.get("avg_visit_time", ""),
                    cluster_micro=meta.get("cluster_micro", -1),
                    cluster_macro=meta.get("cluster_macro", -1),
                    score=final_score,
                    reason=", ".join(reason_bits),
                    distance_km=distance_km,
                    matched_interests=matched_interests,
                    text_detected_interests=text_detected_interests,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        confident = [r for r in scored if r.score >= min_score][:max_results]
        # Never leave the user with nothing - if the bar was set too high, at
        # least surface the single best match instead of an empty answer.
        return confident if confident else scored[:1]

    def find_nearby(self, site_name: str, scale: str = "micro", top_n: int = 5) -> list[str]:
        field = f"cluster_{scale}"
        meta = self.metadata_by_name.get(site_name)
        if not meta:
            return []
        target_cluster = meta.get(field)
        if target_cluster is None or target_cluster == -1:
            return []
        return [
            m["name"]
            for m in self.metadata_by_name.values()
            if m.get(field) == target_cluster and m["name"] != site_name
        ][:top_n]
