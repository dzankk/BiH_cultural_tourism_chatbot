# app.py
"""
Unified BiH Cultural Companion: RAG chatbot (default view) + DBSCAN spatial map.

Two pages, chatbot is the primary/default experience:
  - "Cultural Chatbot": ask a free-text question, get sites re-ranked by the
    RandomForest recommender trained in 02_Preference_Simulation_Model Training.ipynb,
    retrieved semantically from the Chroma vector store built in
    01_Heritage_Knowledge_Base.ipynb.
  - "Spatial Cluster Map": the original DBSCAN seminar map/dashboard, click a
    site in the dropdown to pop its name / corridor / fun fact on the map.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.chatbot_engine import HeritageChatbotEngine, INTERESTS, SEASONS, generate_llm_reply, get_conversational_reply, get_groq_client, synthesize_recommendation_reply  # noqa: E402

st.set_page_config(page_title="KulTur: Smart AI Guide for BiH Heritage", layout="wide", page_icon="🇧🇦")

# CARTO now requires a free API key for raster basemap tiles. Never hardcode
# it here - set it via a Streamlit secret (.streamlit/secrets.toml, gitignored)
# or the CARTO_API_KEY environment variable.
try:
    CARTO_API_KEY = st.secrets.get("CARTO_API_KEY", "") or os.environ.get("CARTO_API_KEY", "")
except Exception:
    CARTO_API_KEY = os.environ.get("CARTO_API_KEY", "")

# Groq powers dynamic chit-chat/meta/off-topic replies (never the travel
# recommendations themselves). Never hardcode the key - same secret/env
# pattern as CARTO. Missing key = automatic fallback to canned templates.
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner=False)
def get_llm_client():
    return get_groq_client(GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Page: Cultural Chatbot (default)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Warming up the chatbot (one-time, ~30s on first load)...")
def get_engine() -> HeritageChatbotEngine:
    return HeritageChatbotEngine()


def render_chatbot_page() -> None:
    st.title("💬 KulTur: Smart AI Guide for BiH Heritage")
    st.caption("Ask about places to visit - the answer is retrieved from the heritage knowledge base and re-ranked by the trained preference model.")

    try:
        engine = get_engine()
    except FileNotFoundError as exc:
        st.error(f"Chatbot data not available: {exc}")
        return

    with st.sidebar:
        st.header("Your preferences")
        interest = st.selectbox("Main interest", INTERESTS, index=0, help="Your main travel interest - also detected automatically from what you type in chat.")
        season = st.selectbox("Current season", SEASONS, index=0, help="Sites open/best in this season score higher.")
        region_pref = st.selectbox("Preferred region", ["Any"] + engine.all_regions, index=0, help="Bias suggestions toward one region of Bosnia & Herzegovina.")
        limited_time = st.checkbox("I only have a short time budget", value=False)
        min_confidence = st.slider(
            "Minimum match confidence", 0.0, 1.0, 0.5, 0.05,
            help="Only show suggestions the model is at least this confident about - raise it for fewer, stronger matches.",
        )
        suggest_clicked = st.button("🎯 Suggest based on my profile", use_container_width=True)
        if st.button("Reset chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("last_scores", None)
            st.session_state.pop("profile_detected_interests", None)
            st.rerun()

        if st.session_state.get("last_scores"):
            with st.expander("🔎 Match details (score & nearby corridor)", expanded=False):
                st.dataframe(
                    pd.DataFrame(st.session_state["last_scores"]),
                    hide_index=True,
                    use_container_width=True,
                )

        detected_so_far = sorted(st.session_state.get("profile_detected_interests", set()))
        with st.expander("🧭 Your traveler profile", expanded=False):
            st.markdown(
                f"- **Main interest:** {interest}\n"
                f"- **Season:** {season}\n"
                f"- **Preferred region:** {region_pref}\n"
                f"- **Time budget:** {'Short on time' if limited_time else 'Flexible'}\n"
                f"- **Also mentioned in chat:** {', '.join(detected_so_far) if detected_so_far else 'nothing yet - try mentioning a vibe!'}"
            )

        if get_llm_client() is None:
            st.caption("💡 Set GROQ_API_KEY to enable dynamic chit-chat replies (currently using template replies).")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": get_conversational_reply("greeting"),
            }
        ]

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_query = st.chat_input("Ask about a place, region, or vibe...")
    user_query = typed_query.strip() if typed_query else typed_query
    is_shortcut = False
    if suggest_clicked:
        user_query = "Suggest some great places for me based on my profile."
        is_shortcut = True

    if user_query:
        display_text = "🎯 Suggest based on my profile" if is_shortcut else user_query
        st.session_state["chat_history"].append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)

        intent = None if is_shortcut else engine.classify_smalltalk(user_query)
        st.session_state.setdefault("profile_detected_interests", set())
        if not is_shortcut:
            # Slot-fill the persistent profile only on an explicit preference
            # statement ("I'm into nature", "I like history") - not just any
            # message that happens to mention the word in passing.
            st.session_state["profile_detected_interests"] |= engine.detect_explicit_interest_statements(user_query)

        # Carry the last mentioned place forward across turns (e.g. "i have a
        # car too" right after asking about Kladanj) - but only when this
        # message is otherwise ambiguous on its own, so a plain "thanks" or
        # "bye" never gets hijacked into a new recommendation. Nothing here
        # is hardcoded - the remembered place comes from the engine's own
        # earlier location resolution.
        last_reco = st.session_state.get("last_reco_context")
        effective_query = user_query
        if (
            not is_shortcut
            and intent == "unclear"
            and last_reco
            and last_reco.get("location") not in (None, "your query")
        ):
            combined_query = f"{last_reco['location']} {user_query}"
            if engine.classify_smalltalk(combined_query) is None:
                effective_query = combined_query
                intent = None
        elif (
            is_shortcut
            and last_reco
            and last_reco.get("location") not in (None, "your query")
        ):
            # "Suggest based on my profile" shouldn't forget a place the user
            # already mentioned in chat - keep it in the same corridor instead
            # of resetting to an unfiltered nationwide search.
            effective_query = f"{last_reco['location']} {user_query}"

        if intent == "meta_why":
            detected = ", ".join(sorted(engine.detect_interests_in_text(user_query))) or "nothing specific this message"
            context_note = (
                f"Factual context - use these exact values, don't invent others: sidebar Main interest='{interest}', "
                f"season='{season}', preferred region='{region_pref}', limited_time={limited_time}. "
                f"This message additionally mentioned these interests: {detected}. "
                "Briefly and accurately explain why recommendations lean that way."
            )
            answer = generate_llm_reply(
                get_llm_client(), intent, user_query, history=st.session_state["chat_history"], context_note=context_note
            )
        elif intent == "distance_pushback":
            last_reco = st.session_state.get("last_reco_context")
            if last_reco:
                sites_desc = "; ".join(
                    f"{s['name']} (~{s['distance_km']:.0f} km)" if s["distance_km"] is not None else s["name"]
                    for s in last_reco["sites"]
                )
                context_note = (
                    f"Factual context - use these exact values, don't invent others: the sites you just suggested "
                    f"near '{last_reco['location']}' were: {sites_desc}. Address the user's complaint honestly - "
                    "confirm the real distances, and offer to narrow to the single closest option or widen the "
                    "search area if they mention they're driving."
                )
            else:
                context_note = (
                    "There's no specific previous recommendation in memory to reference - ask what place they mean "
                    "so you can check the real distance."
                )
            answer = generate_llm_reply(
                get_llm_client(), intent, user_query, history=st.session_state["chat_history"], context_note=context_note
            )
        elif intent is not None:
            answer = generate_llm_reply(get_llm_client(), intent, user_query, history=st.session_state["chat_history"])
        else:
            results = engine.ask(
                query_text=effective_query,
                interest=interest,
                season=season,
                region_pref=region_pref,
                limited_time=limited_time,
                min_score=min_confidence,
                max_results=3,
                # Preferences mentioned earlier in this conversation keep biasing
                # the re-ranker even if this message doesn't repeat them.
                known_interests=st.session_state.get("profile_detected_interests", set()),
            )
            # engine.ask() can fall back to a single below-threshold result rather
            # than ever return nothing - but once a real chat message is involved,
            # the sidebar's confidence slider should be honored strictly, both in
            # what's shown in "Match details" and what's handed to the LLM.
            results = [r for r in results if r.score >= min_confidence]
            st.session_state["last_scores"] = [
                {
                    "Site": r.name,
                    "Confidence": round(r.score, 3),
                    "Nearby": ", ".join(engine.find_nearby(r.name, scale="micro", top_n=3)) or "-",
                }
                for r in results
            ]
            for r in results:
                # Only interests the user actually typed feed the profile tracker -
                # never assistant replies, system prompts, or the neutral
                # "evaluated every interest" scoring fallback.
                st.session_state["profile_detected_interests"] |= set(r.text_detected_interests)
            location_ref = engine.find_location_reference(effective_query)
            st.session_state["last_reco_context"] = {
                "location": location_ref["location"] if location_ref else "your query",
                "sites": [{"name": r.name, "distance_km": r.distance_km} for r in results],
            }

            answer = synthesize_recommendation_reply(
                get_llm_client(), user_query, results, history=st.session_state["chat_history"]
            )

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


# ---------------------------------------------------------------------------
# Page: Spatial Cluster Map (DBSCAN-powered corridor explorer)
# ---------------------------------------------------------------------------

def render_map_page() -> None:
    st.title("🗺️ Heritage Corridor Map")
    st.caption("Sites are grouped into travel corridors by proximity (DBSCAN clustering) - adjust the sensitivity below to see how the groupings change.")
    st.markdown("---")

    st.sidebar.header("Corridor Sensitivity")
    eps_km = st.sidebar.slider("Corridor radius (ε in km)", min_value=5.0, max_value=60.0, value=25.0, step=2.5, help="Sites within this distance of each other are grouped into the same corridor.")
    min_samples = st.sidebar.slider("Minimum sites per corridor", min_value=2, max_value=10, value=4, step=1, help="A corridor needs at least this many sites, or they're marked as isolated.")

    try:
        df = pd.read_csv(PROJECT_ROOT / "MLSTProject" / "BiH_Heritage_Final_Clean.csv")
        df.columns = df.columns.str.strip()
        df = df.sort_values(by="name").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading CSV data file. Details: {e}")
        st.stop()

    st.sidebar.header("Filters")
    category_options = sorted(df["category"].dropna().unique()) if "category" in df.columns else []
    era_options = sorted(df["era_group"].dropna().unique()) if "era_group" in df.columns else []
    selected_categories = st.sidebar.multiselect("Category", category_options, default=[], help="Leave empty to show every category.")
    selected_eras = st.sidebar.multiselect("Era", era_options, default=[], help="Leave empty to show every era.")
    if selected_categories:
        df = df[df["category"].isin(selected_categories)]
    if selected_eras:
        df = df[df["era_group"].isin(selected_eras)]
    df = df.reset_index(drop=True)
    if df.empty:
        st.warning("No sites match the selected filters - try widening your Category/Era selection.")
        st.stop()

    X_deg = df[["latitude", "longitude"]].values
    X_rad = np.radians(X_deg)
    EARTH_RADIUS_KM = 6371.009

    eps_rad = eps_km / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    df["cluster_id"] = db.fit_predict(X_rad)

    total_sites = len(df)
    labels = db.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sites Loaded", total_sites)
    col2.metric("Identified Spatial Corridors", num_clusters)
    col3.metric("Noise Points (Isolated Sites)", num_noise)

    st.markdown("---")
    st.subheader("Interactive Spatial Density Map")

    if not CARTO_API_KEY:
        st.warning("No CARTO_API_KEY configured (set it in .streamlit/secrets.toml or as an environment variable) - the map tiles below won't load.")

    m = folium.Map(
        location=[44.15, 17.80],
        zoom_start=8,
        tiles=f"https://basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    )

    colors = ["#00d2ff", "#70ff00", "#ff0076", "#ffb300", "#a000ff", "#00ffaa", "#ff5722", "#eccc68"]
    fun_fact_col = "fun_fact" if "fun_fact" in df.columns else ("fun_facts" if "fun_facts" in df.columns else None)

    if "last_selected" not in st.session_state:
        st.session_state["last_selected"] = "-- Select a Site to Trigger Map Popup --"
    if "map_render_key" not in st.session_state:
        st.session_state["map_render_key"] = 0

    for idx, row in df.iterrows():
        cid = row["cluster_id"]
        lat, lon = row["latitude"], row["longitude"]

        if cid == -1:
            color = "#ffffff"
            radius = 5
            fill_op = 0.2
            tag = "Noise / Spatial Outlier"
        else:
            color = colors[cid % len(colors)]
            radius = 7.5
            fill_op = 0.8
            tag = f"Corridor Group {cid + 1}"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 250px; background: #222; color: #fff; padding: 12px; border-radius: 6px; border-left: 5px solid {color}; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
            <span style="font-size: 9px; font-weight: bold; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">{tag}</span>
            <h4 style="margin: 4px 0 6px 0; font-size: 14px; color: #fff; border-bottom: 1px solid #444; padding-bottom: 4px;">{row['name']}</h4>
            <p style="margin: 0 0 6px 0; font-size: 11px; color: #aaa;"><b>Municipality:</b> {row['location']}</p>
            <p style="margin: 0 0 8px 0; font-size: 11px; line-height: 1.4; color: #eee; font-style: italic;">"{row['description']}"</p>
        """

        if fun_fact_col and pd.notna(row[fun_fact_col]):
            popup_html += f"""
            <div style="background: rgba(0, 210, 255, 0.15); border: 1px solid rgba(0, 210, 255, 0.3); padding: 6px; border-radius: 4px; font-size: 11px; color: #00d2ff; margin-top: 4px;">
                💡 <b>Fun Fact:</b> {row[fun_fact_col]}
            </div>
            """
        popup_html += "</div>"

        should_show_popup = st.session_state["last_selected"] == row["name"]

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=290, show=should_show_popup),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_op,
            weight=1.5,
            tooltip=row["name"],
        ).add_to(m)

    st_folium(m, width="100%", height=650, key=f"bih_map_tracker_{st.session_state['map_render_key']}", returned_objects=[])

    st.markdown("---")
    st.subheader("Heritage Site Navigator")

    selected_site = st.selectbox(
        "Choose any site from the dataset to instantly trigger its informational box on the map canvas above:",
        options=["-- Select a Site to Trigger Map Popup --"] + list(df["name"].unique()),
    )

    if selected_site != st.session_state["last_selected"]:
        st.session_state["last_selected"] = selected_site
        st.session_state["map_render_key"] += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Navigation - Chatbot is the default/first page
# ---------------------------------------------------------------------------

PAGES = {
    "💬 KulTur Chat": render_chatbot_page,
    "🗺️ Heritage Corridor Map": render_map_page,
}

st.sidebar.title("KulTur")
st.sidebar.caption("Smart AI Guide for BiH Heritage")
page = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)
st.sidebar.markdown("---")

PAGES[page]()
