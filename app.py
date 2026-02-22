import os
import base64
import requests
import queue
import time
import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode

import data
from backend.schemas import AuraUIPayload, DDxEntry, DiseaseQuestions
from backend.main_agent import AuraPipeline
from backend.audio_processing import AudioStreamingProcessor

load_dotenv()

# ── DDx merge helpers ─────────────────────────────────────────────────────────
_SUSPICION_RANK = {
    "High":   2,
    "Medium": 1,
    "Low":    0,
}
DDX_MAX = 5   # Maximum number of DDx entries to keep at any time


def _merge_ddx(
    existing: list,
    incoming: list,
    max_entries: int = DDX_MAX,
) -> list:
    """
    Merge two DDx lists without ever cancelling an existing diagnosis.

    Rules
    -----
    1. Existing entries are always kept (their suspicion level is updated if
       the new analysis raises or lowers confidence for that disease).
    2. Genuinely new diseases (not in the existing list) are appended.
    3. If the merged list exceeds *max_entries*, the entry with the lowest
       suspicion level that was added *earliest* (highest rank number = lowest
       priority) is removed.  Only removes one entry per call so the list
       converges gradually rather than jumping.
    4. Ranks are re-numbered 1..N after every merge.
    """
    # Index existing entries by normalised disease name
    merged: dict[str, DDxEntry] = {
        e.disease.lower().strip(): e for e in existing
    }

    for new_entry in incoming:
        key = new_entry.disease.lower().strip()
        if key in merged:
            # Always update probability_pct and supporting evidence;
            # update suspicion level if the new signal differs.
            updates = {
                "probability_pct": new_entry.probability_pct,
                "key_supporting": new_entry.key_supporting or merged[key].key_supporting,
            }
            old_rank = _SUSPICION_RANK.get(merged[key].suspicion.value, 0)
            new_rank = _SUSPICION_RANK.get(new_entry.suspicion.value, 0)
            if new_rank != old_rank:
                updates["suspicion"] = new_entry.suspicion
            merged[key] = merged[key].model_copy(update=updates)
        else:
            # Brand-new disease — append
            merged[key] = new_entry

    result = list(merged.values())

    # Enforce cap: drop oldest entry with lowest confidence when over limit
    while len(result) > max_entries:
        # Sort ascending by confidence so the weakest candidate is first;
        # among ties, the one with the highest rank number (added earliest /
        # ranked lowest) is chosen for removal.
        result.sort(
            key=lambda e: (
                e.confidence,
                -e.rank,
            )
        )
        result.pop(0)   # remove the lowest-confidence / oldest entry

    # Re-number ranks 1..N sorted by confidence descending
    result.sort(key=lambda e: e.confidence, reverse=True)
    for i, entry in enumerate(result, start=1):
        entry = entry.model_copy(update={"rank": i})
        result[i - 1] = entry

    return result

def transcribe_audio(audio_bytes):
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    payload = {"model_id": "scribe_v1"}
    try:
        resp = requests.post(url, headers=headers, files=files, data=payload)
        return resp.json().get("text", "") if resp.status_code == 200 else f"[Transcription Failed: {resp.text}]"
    except Exception as e:
        return f"[Transcription Error: {e}]"

st.set_page_config(
    page_title="Aura - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = AuraUIPayload()
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = None

active_patient = data.get_patient_by_id(st.session_state.current_patient_id)

if "patient_history" not in st.session_state:
    from backend.schemas import PatientHistory as _PH
    # Seed the PatientHistory object with the JSON data
    ph = _PH()
    if active_patient:
        ph.symptoms = []
        ph.risk_factors = active_patient.get("past_medical_history", [])
        ph.medications = active_patient.get("current_medications", [])
        ph.relevant_history = f"**{active_patient.get('name', 'Unknown')}** ({active_patient.get('age', 'N/A')} {active_patient.get('gender', 'N/A')}). "
        if active_patient.get("allergies") and active_patient["allergies"] != ["None"]:
            ph.relevant_history += f"Allergies: {', '.join(active_patient['allergies'])}. "
        past_visits = active_patient.get("past_visits", [])
        if past_visits:
            ph.relevant_history += "Previous visits: "
            for visit in past_visits:
                ph.relevant_history += (
                    f"[{visit.get('date', 'N/A')}] "
                    f"{visit.get('chief_complaint', '')} → "
                    f"Dx: {visit.get('diagnosis', '')}. "
                    f"Tx: {visit.get('treatment', '')}. "
                )
    st.session_state.patient_history = ph

if "pipeline" not in st.session_state:
    # Pass the seeded history into the pipeline so it doesn't start blank
    st.session_state.pipeline = AuraPipeline(initial_history=st.session_state.patient_history)
if "speaker_mode" not in st.session_state:
    st.session_state.speaker_mode = "Doctor"

if "was_playing" not in st.session_state:
    st.session_state.was_playing = False
if "last_pipeline_run" not in st.session_state:
    st.session_state.last_pipeline_run = 0.0
if "transcript_changed_since_llm" not in st.session_state:
    st.session_state.transcript_changed_since_llm = False


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Pixel-perfect match to the React/Tailwind source
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global Reset ──────────────────────────────────────────────────────────── */
html, body, * , [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Clinical light-blue background */
.stApp { background: linear-gradient(155deg, #f0f9ff 0%, #e0f2fe 45%, #f8fafc 100%) !important; min-height: 100vh; }

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
[data-testid="stSidebar"]      { display: none !important; }
[data-testid="stHeader"]       { display: none !important; }
#MainMenu, footer              { visibility: hidden; }
[data-testid="stIconMaterial"] { display: none !important; }
.stMarkdown h3                 { display: none; }

/* max-w-7xl mx-auto */
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1280px;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
.aura-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0369a1 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    margin: 0; padding: 0; line-height: 1.2;
}
.aura-subtitle {
    color: #7dd3fc;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 5px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

/* ── Section Titles ────────────────────────────────────────────────────────── */
.section-title {
    font-size: 0.68rem;
    font-weight: 700;
    color: #0369a1;
    margin-bottom: 0.85rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.section-title::before {
    content: \'\';
    display: inline-block;
    width: 3px; height: 14px;
    background: linear-gradient(180deg, #38bdf8, #0369a1);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ── Patient History Panel (frosted glass) ─────────────────────────────────── */
.ph-panel {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(186,230,253,0.8);
    border-radius: 1.1rem;
    padding: 1.5rem;
    min-height: 380px;
    overflow-y: auto;
    box-shadow: 0 1px 3px rgba(14,165,233,0.07), 0 8px 24px rgba(14,165,233,0.06);
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}

/* section label */
.ph-label {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}

/* symptom / risk chips */
.ph-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ph-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.7rem;
    border-radius: 9999px;
    font-size: 0.77rem;
    font-weight: 500;
    border: 1px solid;
    line-height: 1.4;
}
.ph-chip.symptom   { background: #e0f2fe; color: #0369a1; border-color: #7dd3fc; }
.ph-chip.negated   { background: #f1f5f9; color: #94a3b8; border-color: #e2e8f0; text-decoration: line-through; }
.ph-chip.risk      { background: #fff7ed; color: #c2410c; border-color: #fed7aa; }
.ph-chip.med       { background: #f0fdf4; color: #166534; border-color: #86efac; }

/* meta row (duration / severity) */
.ph-meta-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.ph-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 0.5rem;
    padding: 0.3rem 0.75rem;
    font-size: 0.8rem;
    color: #0369a1;
    font-weight: 500;
}
.ph-meta-pill strong { color: #0c4a6e; }

/* history prose box */
.ph-history-box {
    background: #f0f9ff;
    border-left: 3px solid #0ea5e9;
    border-radius: 0 0.5rem 0.5rem 0;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #0c4a6e;
    line-height: 1.65;
    font-style: italic;
}

/* listening pulse dot */
@keyframes ph-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.ph-live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #ef4444;
    display: inline-block;
    animation: ph-pulse 1.4s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 5rem 1.5rem;
    color: #7dd3fc;
    font-size: 0.88rem;
    line-height: 1.6;
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.empty-state .icon { font-size: 2.2rem; margin-bottom: 0.75rem; }

/* ── Form Input ────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid #bae6fd !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important;
    box-shadow: none !important;
    color: #0c4a6e !important;
    transition: all 0.15s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: transparent !important;
    box-shadow: 0 0 0 2px #0ea5e9 !important;
    outline: none !important;
}
.stTextInput > div > div,
.stTextInput > div { border: none !important; box-shadow: none !important; }

/* Form submit button */
.stFormSubmitButton > button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: #0ea5e9 !important;
    font-size: 1.2rem !important;
    padding: 0.5rem !important;
    min-height: 0 !important;
    transition: all 0.15s ease !important;
}
.stFormSubmitButton > button:hover {
    background: #e0f2fe !important;
    color: #0369a1 !important;
    transform: none !important;
    box-shadow: none !important;
    border-radius: 0.375rem !important;
}

/* General button */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    border: none; border-radius: 0.5rem;
    padding: 0.5rem 1.25rem; font-weight: 600; font-size: 0.85rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(14,165,233,0.3);
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(14,165,233,0.4);
    color: white; border: none;
}

/* ── DDx Cards ─────────────────────────────────────────────────────────────── */
.ddx-card {
    padding: 0.85rem 1rem 0.85rem 1.15rem;
    border-radius: 0.75rem;
    border-width: 1px;
    border-style: solid;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    transition: all 0.2s ease;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.ddx-card::before {
    content: \'\';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
}
.ddx-card:hover {
    transform: translateX(3px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
}

/* HIGH — rose/red */
.ddx-card.high { background: rgba(255,241,242,0.9); border-color: #fecdd3; }
.ddx-card.high::before { background: linear-gradient(180deg,#f43f5e,#be123c); }

/* MEDIUM — amber */
.ddx-card.medium { background: rgba(255,251,235,0.9); border-color: #fde68a; }
.ddx-card.medium::before { background: linear-gradient(180deg,#f59e0b,#d97706); }

/* LOW — sky blue */
.ddx-card.low { background: rgba(240,249,255,0.9); border-color: #bae6fd; }
.ddx-card.low::before { background: linear-gradient(180deg,#38bdf8,#0284c7); }

/* Condition name */
.ddx-card .condition { font-weight: 600; font-size: 0.9rem; }
.ddx-card.high .condition   { color: #881337; }
.ddx-card.medium .condition { color: #78350f; }
.ddx-card.low .condition    { color: #0c4a6e; }

/* Badge */
.ddx-badge {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 9999px;
    border-width: 1px;
    border-style: solid;
    white-space: nowrap;
    letter-spacing: 0.03em;
}
.ddx-badge.high   { background: #ffe4e6; color: #be123c; border-color: #fecdd3; }
.ddx-badge.medium { background: #fef3c7; color: #b45309; border-color: #fde68a; }
.ddx-badge.low    { background: #e0f2fe; color: #0369a1; border-color: #7dd3fc; }

/* ── Clinical Gap ──────────────────────────────────────────────────────────── */
.clinical-gap-header {
    font-size: 0.68rem;
    font-weight: 700;
    color: #0369a1;
    margin-top: 1.75rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.clinical-gap-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #7dd3fc;
    border-radius: 0.8rem;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 10px rgba(14,165,233,0.1);
}
.clinical-gap-card p,
.clinical-gap-card {
    color: #0c4a6e;
    font-weight: 500;
    line-height: 1.65;
    font-size: 0.9rem;
}

/* ── Mic Button ────────────────────────────────────────────────────────────── */
.mic-btn {
    width: 2.75rem; height: 2.75rem;
    border-radius: 9999px;
    border: 2px solid #bae6fd;
    background: #f0f9ff;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s ease;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.mic-btn:hover {
    border-color: #0ea5e9;
    background: #e0f2fe;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15);
}
.mic-btn.recording {
    border-color: #ef4444;
    background: #fef2f2;
    animation: mic-pulse 1.5s ease-in-out infinite;
}
@keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%      { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
}
.mic-status { font-size: 0.75rem; color: #ef4444; font-weight: 500; margin-top: 2px; text-align: center; }

/* ── Tabs (hide) ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0; }

/* ── Misc Streamlit overrides ──────────────────────────────────────────────── */
.stAlert { border-radius: 0.75rem; border: none; }
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background: rgba(255,255,255,0.82);
    border: 1px solid #bae6fd !important;
    border-radius: 1rem !important;
    box-shadow: 0 1px 3px rgba(14,165,233,0.08);
}
</style>
""", unsafe_allow_html=True)





# Splash: entirely JS-driven so Streamlit reruns never re-inject the element
st_html("""
<script>
(function() {
    var ss = window.parent.sessionStorage;
    if (ss && ss.getItem('aura_splash_done')) return;

    var doc = window.parent.document;

    // Inject CSS
    var style = doc.createElement('style');
    style.textContent = [
        '#aura-splash{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;',
        'background:linear-gradient(155deg,#f0f9ff 0%,#cae8fb 45%,#e0f2fe 100%);',
        'opacity:1;transition:opacity 0.7s ease;pointer-events:all;}',
        '.splash-bg{position:absolute;inset:0;background:',
        'radial-gradient(ellipse 60% 40% at 20% 30%,rgba(14,165,233,.13) 0%,transparent 70%),',
        'radial-gradient(ellipse 50% 50% at 80% 70%,rgba(56,189,248,.10) 0%,transparent 70%);',
        'animation:blob-drift 6s ease-in-out infinite alternate;}',
        '@keyframes blob-drift{from{transform:scale(1) translateY(0)}to{transform:scale(1.05) translateY(-10px)}}',
        '.splash-content{position:relative;display:flex;flex-direction:column;align-items:center;gap:.6rem;',
        'animation:splash-rise 0.9s cubic-bezier(0.16,1,0.3,1) 0.1s both;}',
        '@keyframes splash-rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}',
        '.splash-logo{font-family:Inter,sans-serif;font-size:clamp(4.5rem,12vw,8rem);font-weight:800;',
        'letter-spacing:-0.04em;line-height:1;',
        'background:linear-gradient(135deg,#0369a1 0%,#0ea5e9 50%,#38bdf8 100%);',
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;',
        'animation:logo-breathe 2s ease-in-out .1s infinite alternate;user-select:none;}',
        '@keyframes logo-breathe{from{opacity:.88;transform:scale(1)}to{opacity:1;transform:scale(1.025)}}',
        '.splash-sub{font-family:Inter,sans-serif;font-size:clamp(.75rem,2vw,.95rem);font-weight:600;',
        'letter-spacing:.22em;text-transform:uppercase;color:#0369a1;opacity:0;',
        'animation:sub-fadein .7s ease .8s forwards;user-select:none;}',
        '@keyframes sub-fadein{to{opacity:.75}}',
        '.splash-bar{width:120px;height:2px;background:rgba(14,165,233,.18);border-radius:9999px;',
        'overflow:hidden;margin-top:2rem;opacity:0;animation:sub-fadein .5s ease 1.1s forwards;}',
        '.splash-bar-inner{height:100%;width:0%;background:linear-gradient(90deg,#0ea5e9,#38bdf8);',
        'border-radius:9999px;animation:bar-fill 2s cubic-bezier(.4,0,.2,1) 1.15s forwards;}',
        '@keyframes bar-fill{from{width:0%}to{width:100%}}'
    ].join('');
    doc.head.appendChild(style);

    // Build DOM
    var el = doc.createElement('div');
    el.id = 'aura-splash';
    el.innerHTML = [
        '<div class="splash-bg"></div>',
        '<div class="splash-content">',
        '  <div class="splash-logo">Aura</div>',
        '  <div class="splash-sub">Think Faster. Diagnose Better.</div>',
        '  <div class="splash-bar"><div class="splash-bar-inner"></div></div>',
        '</div>'
    ].join('');
    doc.body.appendChild(el);

    // Fade out then remove
    setTimeout(function() {
        el.style.opacity = '0';
        setTimeout(function() {
            if (el.parentNode) el.parentNode.removeChild(el);
            if (ss) ss.setItem('aura_splash_done', '1');
        }, 750);
    }, 3100);
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
header_col1, header_col2 = st.columns([3, 2], gap="large")
with header_col1:
    st.markdown("""
    <div class="aura-header" style="margin-bottom:2rem">
        <h1>Aura</h1>
        <div class="aura-subtitle">Think Faster. Diagnose Better.</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.current_patient_id:
        active_patient_opt = data.get_patient_by_id(st.session_state.current_patient_id)
        p_name = active_patient_opt.get("name", "Unknown") if active_patient_opt else "Unknown"
        st.markdown(f'<div style="text-align:right; color:#0369a1; font-weight:700; font-size:0.95rem; padding-top:1rem; background:rgba(224,242,254,0.6); border:1px solid #bae6fd; border-radius:0.6rem; padding:0.5rem 0.9rem; display:inline-block; backdrop-filter:blur(8px);">🩺 {p_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right; color:#7dd3fc; font-style:italic; font-size:0.82rem; padding-top:1rem; letter-spacing:0.04em;">⟳ Listening for patient name…</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — grid-cols-5: 3/5 left, 2/5 right, gap-8
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ─── LEFT: Patient History (col-span-3) ─────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">Patient Clinical Profile</div>',
                unsafe_allow_html=True)

    ph_placeholder = st.empty()

    def _chips(items: list, cls: str, icon: str = "") -> str:
        if not items:
            return '<span style="color:#cbd5e1;font-size:0.82rem;">None recorded yet</span>'
        return "".join(
            f'<span class="ph-chip {cls}">{icon + " " if icon else ""}{item}</span>'
            for item in items
        )

    def render_patient_history(is_live: bool = False):
        ph = st.session_state.patient_history
        has_data = bool(
            ph.symptoms or ph.risk_factors or ph.medications
            or ph.duration or ph.severity or ph.relevant_history
        )

        live_str = '<span class="ph-live-dot"></span> Listening…' if is_live else ""

        if not has_data:
            content = '<div class="empty-state">' \
                      '<div class="icon">🩺</div>' \
                      '<div>Clinical data will appear here as the consultation progresses.<br>' \
                     f'{live_str}</div>' \
                      '</div>'
        else:
            # ── Symptoms ─────────────────────────────────────────────────────
            sym_html = _chips(ph.symptoms, "symptom", "●")
            neg_html = _chips(ph.negated_symptoms, "negated", "✕") if ph.negated_symptoms else ""

            # ── Risk factors & meds ──────────────────────────────────────────
            risk_html = _chips(ph.risk_factors, "risk", "⚠")
            med_html  = _chips(ph.medications,  "med",  "💊")

            # ── Meta (duration / severity) ────────────────────────────────────
            meta_parts = []
            if ph.duration:
                meta_parts.append(f'<span class="ph-meta-pill">⏱ Duration: <strong>{ph.duration}</strong></span>')
            if ph.severity:
                meta_parts.append(f'<span class="ph-meta-pill">📊 Severity: <strong>{ph.severity}</strong></span>')
            meta_html = "<div class='ph-meta-row'>" + "".join(meta_parts) + "</div>" if meta_parts else ""

            # ── Relevant history prose ────────────────────────────────────────
            hist_html = ""
            if ph.relevant_history:
                hist_html = f'<div class="ph-history-box">{ph.relevant_history}</div>'

            content = '<div>' \
                      '<div class="ph-label">Symptoms</div>' \
                     f'<div class="ph-chips">{sym_html}</div>' \
                      '</div>'
            if neg_html:
                content += f'<div><div class="ph-label">Ruled Out</div><div class="ph-chips">{neg_html}</div></div>'
            content += '<div>' \
                       '<div class="ph-label">Risk Factors</div>' \
                      f'<div class="ph-chips">{risk_html}</div>' \
                       '</div>' \
                       '<div>' \
                       '<div class="ph-label">Current Medications</div>' \
                      f'<div class="ph-chips">{med_html}</div>' \
                       '</div>'
            content += meta_html
            if hist_html:
                content += f'<div><div class="ph-label">Clinical Summary</div>{hist_html}</div>'
            if live_str:
                content += f'<div style="margin-top:0.25rem;font-size:0.75rem;color:#94a3b8;">{live_str}</div>'

        ph_placeholder.markdown(
            f'<div class="ph-panel">{content}</div>',
            unsafe_allow_html=True
        )

    render_patient_history()

    # ── Voice Recording Button (WebRTC) ───────────────────────────────────────
    # We call webrtc_streamer FIRST so we can get its fresh `state.playing` value
    webrtc_ctx = webrtc_streamer(
        key="speech_to_text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioStreamingProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    # Use the fresh state for rendering ALL UI dynamically
    is_recording = webrtc_ctx.state.playing

    rec_label = "🔴 Recording…" if is_recording else "Press to start consultation"
    rec_color = "#ef4444" if is_recording else "#94a3b8"
    st.markdown(
        f'<div style="text-align:center;font-size:0.82rem;font-weight:500;color:{rec_color};margin-top:1.5rem;">'
        f'{rec_label}</div>',
        unsafe_allow_html=True,
    )

    # Hide the WebRTC iframe dynamically but keep it in the DOM so it functions
    st.markdown("""
    <style>
    /* Our custom, native HTML recording button */
    .aura-record-btn {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: #ffffff;
        border: 5px solid #1e293b;
        cursor: pointer;
        padding: 0;
        margin: 8px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: none;
        transition: all 0.2s ease;
        position: relative;
    }
    .aura-record-btn::after {
        content: "";
        display: block;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #ef4444;
        transition: all 0.2s ease;
    }
    .aura-record-btn:hover {
        box-shadow: 0 0 0 6px rgba(239,68,68,0.15);
        transform: scale(1.05);
    }
    .btn-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render our custom native button and a script that explicitly forwards clicks
    # to the hidden WebRTC iframe's START button.
    
    # Conditional styles based on fresh recording state
    is_rec_str = str(is_recording).lower()
    button_inner_radius = "8px" if is_recording else "50%"
    button_inner_size = "24px" if is_recording else "32px"
    pulse_animation = "animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;" if is_recording else ""
    hover_scale = "1" if is_recording else "1.05"

    st_html(f"""
    <div class="btn-container">
        <button id="aura-trigger" class="aura-record-btn" aria-label="Start Recording">
            <div class="aura-record-inner"></div>
        </button>
    </div>
    
    <script>
    // Actively seek out the WebRTC iframe and hide its container completely
    function hideWebRTCNative() {{
        const iframes = window.parent.document.querySelectorAll('iframe');
        for (let iframe of iframes) {{
            if (iframe.title && iframe.title.includes('webrtc')) {{
                if (iframe.parentElement) {{
                    iframe.parentElement.style.position = 'absolute';
                    iframe.parentElement.style.opacity = '0';
                    iframe.parentElement.style.pointerEvents = 'none';
                    iframe.parentElement.style.height = '1px';
                    iframe.parentElement.style.width = '1px';
                    iframe.parentElement.style.overflow = 'hidden';
                    iframe.parentElement.style.left = '-9999px';
                }}
            }}
        }}
    }}
    hideWebRTCNative();
    setInterval(hideWebRTCNative, 500);

    document.getElementById('aura-trigger').addEventListener('click', function() {{
        const iframes = window.parent.document.querySelectorAll('iframe');
        for (let iframe of iframes) {{
            if (iframe.title && iframe.title.includes('webrtc')) {{
                try {{
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (!doc) continue;
                    const btns = doc.querySelectorAll('button');
                    if (btns.length > 0) {{
                        btns[0].click();
                        return;
                    }}
                }} catch (e) {{}}
            }}
        }}
    }});
    </script>
    
    <style>
    body {{ background: transparent !important; margin: 0; overflow: hidden; padding: 10px; }}
    
    @keyframes pulse-ring {{
        0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
        70% {{ box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
    }}

    .aura-record-btn {{
        width: 72px; 
        height: 72px; 
        border-radius: 50%; 
        background: #ffffff;
        border: 5px solid #1e293b; 
        cursor: pointer; 
        padding: 0; 
        margin: 10px auto;
        display: flex; 
        align-items: center; 
        justify-content: center; 
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        {pulse_animation}
    }}

    .aura-record-inner {{
        width: {button_inner_size}; 
        height: {button_inner_size}; 
        border-radius: {button_inner_radius};
        background: #ef4444; 
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .aura-record-btn:hover {{ 
        box-shadow: 0 0 0 6px rgba(239,68,68,0.15); 
        transform: scale({hover_scale}); 
    }}
    
    .btn-container {{ 
        display: flex; 
        justify-content: center; 
        width: 100%; 
        height: 100%; 
        align-items: center; 
    }}
    </style>
    """, height=120)

    if webrtc_ctx.state.playing:
        status_placeholder = st.empty()

# ─── RIGHT: DDx & Clinical Gaps (col-span-2) ────────────────────────────────
with col_right:
    payload: AuraUIPayload = st.session_state.ai_analysis

    # ── DDx Tracker ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Differential Diagnosis (DDx)</div>',
                unsafe_allow_html=True)

    if payload.ddx:
        html = ""
        for entry in payload.ddx:
            sev = entry.suspicion.value.lower()   # "high" / "medium" / "low"
            cls = sev if sev in ("high", "medium", "low") else "low"
            pct = f"{entry.probability_pct:.1f}%"
            html += f"""
            <div class="ddx-card {cls}">
                <div style="display:flex; flex-direction:column; gap:0.25rem;">
                    <span class="condition">{entry.disease}</span>
                    <span class="ddx-prob">📊 {pct} probability</span>
                </div>
                <span class="ddx-badge {cls}">{entry.confidence:.1f}.</span>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ddx-card low" style="opacity:.4; cursor:default">
            <span class="condition" style="color:#94a3b8; font-weight:400">
            Awaiting transcript data…</span>
        </div>""", unsafe_allow_html=True)

    # ── Clinical Gap / Follow-up Question ────────────────────────────────────
    st.markdown("""
    <div class="clinical-gap-header">
        <span>💡</span> Clinical Gap Identified
    </div>""", unsafe_allow_html=True)

    if payload.follow_up_question:
        st.markdown(
            f'<div class="clinical-gap-card"><p>{payload.follow_up_question}</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="clinical-gap-card" style="opacity:.5">
            <p>Clinical gaps will appear here as the consultation progresses.</p>
        </div>""", unsafe_allow_html=True)

    # ── Safety Issues ────────────────────────────────────────────────────────
    if payload.safety_issues:
        st.markdown("""
        <div class="clinical-gap-header">
            <span>⚠️</span> Safety Review
        </div>""", unsafe_allow_html=True)
        for issue in payload.safety_issues:
            st.markdown(f"""
            <div class="ddx-card high">
                <span class="condition">{issue}</span>
            </div>""", unsafe_allow_html=True)

    # ── Per-Disease Questions (QuestionGenie) ────────────────────────────────
    if payload.questions_by_disease:
        TARGET_COLORS = {
            "rule_in":      ("#dcfce7", "#166534", "#86efac"),
            "rule_out":     ("#fff1f2", "#9f1239", "#fecdd3"),
            "differentiate":("#e0f2fe", "#0369a1", "#7dd3fc"),
        }

        # Build one HTML block — hover reveals questions (no click needed)
        html = '<div class="clinical-gap-header"><span>❓</span> Targeted Questions</div>'
        html += '<style>'
        html += '.dq-card{position:relative;margin-bottom:0.55rem;border-radius:0.85rem;border:1px solid rgba(186,230,253,0.7);background:rgba(255,255,255,0.88);overflow:hidden;transition:box-shadow .25s,transform .25s;backdrop-filter:blur(10px);}'
        html += '.dq-card:hover{box-shadow:0 6px 20px rgba(14,165,233,0.12);transform:translateX(3px);}'
        html += '.dq-header{display:flex;align-items:center;gap:0.5rem;padding:0.7rem 1rem;font-size:0.85rem;font-weight:700;color:#0369a1;cursor:default;border-bottom:1px solid rgba(186,230,253,0);transition:border-color .25s;}'
        html += '.dq-card:hover .dq-header{border-bottom-color:rgba(186,230,253,0.5);}'
        html += '.dq-questions{max-height:0;overflow:hidden;transition:max-height .38s ease,padding .2s;}'
        html += '.dq-card:hover .dq-questions{max-height:600px;padding-bottom:0.8rem;}'
        html += '.dq-q{margin:0.35rem 0.75rem 0;padding:0.6rem 0.9rem;border-radius:0.6rem;border:1px solid;}'
        html += '.dq-q-text{font-weight:600;font-size:0.85rem;margin-bottom:0.25rem;}'
        html += '.dq-q-rationale{font-size:0.77rem;color:#475569;line-height:1.55;}'
        html += '.dq-q-tag{font-size:0.64rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-top:0.4rem;display:inline-block;padding:0.12rem 0.4rem;border-radius:4px;background:rgba(0,0,0,0.06);}'
        html += '</style>'

        for dq in payload.questions_by_disease:
            html += f'<div class="dq-card"><div class="dq-header">🔬 {dq.disease}</div><div class="dq-questions">'
            for q in dq.questions:
                bg, fg, border = TARGET_COLORS.get(q.target.value, ("#f1f5f9", "#334155", "#e2e8f0"))
                label = q.target.value.replace("_", " ")
                html += (
                    f'<div class="dq-q" style="background:{bg};border-color:{border};">'
                    f'<div class="dq-q-text" style="color:{fg};">{q.question}</div>'
                    f'<div class="dq-q-rationale">{q.clinical_rationale}</div>'
                    f'<span class="dq-q-tag" style="color:{fg};">{label}</span>'
                    f'</div>'
                )
            html += '</div></div>'

        st.markdown(html, unsafe_allow_html=True)

        # st.markdown strips <script> tags, so inject the scroll behaviour via
        # st_html (components.v1.html), which runs inside its own iframe and can
        # reach window.parent to query and scroll the main Streamlit frame.
        st_html("""
<script>
(function() {
    function attachScroll() {
        var doc = window.parent.document;
        var cards = doc.querySelectorAll('.dq-card');
        if (!cards.length) { setTimeout(attachScroll, 300); return; }
        cards.forEach(function(card) {
            // Guard: only attach once
            if (card._scrollAttached) return;
            card._scrollAttached = true;
            card.addEventListener('mouseenter', function() {
                setTimeout(function() {
                    var rect = card.getBoundingClientRect();
                    var cardMid = rect.top + rect.height / 2;
                    var vh = window.parent.innerHeight;
                    var delta = cardMid - vh / 2;
                    window.parent.scrollBy({ top: delta, behavior: 'smooth' });
                }, 60);
            });
        });
    }
    // Run once the parent DOM has settled
    if (document.readyState === 'complete') {
        attachScroll();
    } else {
        window.addEventListener('load', attachScroll);
    }
})();
</script>
""", height=0)
if webrtc_ctx.state.playing:
    st.session_state.was_playing = True
    status_placeholder.info("Listening... (streaming to Gradium API)")

    while webrtc_ctx.state.playing:
        if webrtc_ctx.audio_processor:
            collected_text = False
            try:
                while True:
                    text_item = webrtc_ctx.audio_processor.text_queue.get_nowait()
                    if text_item and text_item.strip():
                        st.session_state.transcript.append(text_item.strip() + " ")
                        collected_text = True
            except queue.Empty:
                pass

            if collected_text:
                st.session_state.transcript_changed_since_llm = True
                # Show live pulsing dot while accumulating speech
                render_patient_history(is_live=True)

            # Run the pipeline every 5 s if the transcript has new content
            if (
                st.session_state.transcript_changed_since_llm
                and time.time() - st.session_state.last_pipeline_run >= 5.0
            ):
                st.session_state.transcript_changed_since_llm = False
                st.session_state.last_pipeline_run = time.time()
                status_placeholder.info("AI is analysing consultation...")

                # use real transcript
                full_transcript = "".join(st.session_state.transcript)
                new_analysis = st.session_state.pipeline.run(full_transcript)
                print("new_analysis", new_analysis)

                # ── Always sync PatientHistory — even when updateUi is False ──
                new_ph = new_analysis.patient_history
                
                # ── Identity Extraction Logic ──
                if st.session_state.current_patient_id is None and new_ph.patient_name:
                    extracted_name = new_ph.patient_name.lower().strip()
                    all_patients = data.get_all_patients()
                    matched_id = None
                    for p in all_patients:
                        db_name = p.get("name", "").lower()
                        if extracted_name in db_name or db_name in extracted_name:
                            matched_id = p.get("id")
                            break
                    
                    if matched_id:
                        st.session_state.current_patient_id = matched_id
                        matched_patient_data = data.get_patient_by_id(matched_id)
                        
                        from backend.schemas import PatientHistory as _PH
                        # Carry forward ALL existing clinical data from the transcript
                        old_ph = st.session_state.patient_history
                        ph = _PH()
                        if matched_patient_data:
                            ph.patient_name = new_ph.patient_name
                            # Keep symptoms extracted from the conversation
                            ph.symptoms = list(set(new_ph.symptoms or old_ph.symptoms))
                            ph.negated_symptoms = list(set(new_ph.negated_symptoms or old_ph.negated_symptoms))
                            ph.duration = new_ph.duration or old_ph.duration
                            ph.severity = new_ph.severity or old_ph.severity
                            # Merge risk factors from JSON + any extracted from conversation
                            ph.risk_factors = list(set(
                                matched_patient_data.get("past_medical_history", [])
                                + (new_ph.risk_factors or [])
                            ))
                            ph.medications = matched_patient_data.get("current_medications", [])
                            ph.relevant_history = f"**{matched_patient_data.get('name', 'Unknown')}** ({matched_patient_data.get('age', 'N/A')} {matched_patient_data.get('gender', 'N/A')}). "
                            if matched_patient_data.get("allergies") and matched_patient_data["allergies"] != ["None"]:
                                ph.relevant_history += f"Allergies: {', '.join(matched_patient_data['allergies'])}. "
                            # Include past visit history for cross-visit intelligence
                            past_visits = matched_patient_data.get("past_visits", [])
                            if past_visits:
                                ph.relevant_history += "Previous visits: "
                                for visit in past_visits:
                                    ph.relevant_history += (
                                        f"[{visit.get('date', 'N/A')}] "
                                        f"{visit.get('chief_complaint', '')} → "
                                        f"Dx: {visit.get('diagnosis', '')}. "
                                        f"Tx: {visit.get('treatment', '')}. "
                                    )
                            # Append the AI-generated clinical summary if available
                            if new_ph.relevant_history:
                                ph.relevant_history += new_ph.relevant_history
                                
                        st.session_state.patient_history = ph
                        st.session_state.pipeline = AuraPipeline(initial_history=st.session_state.patient_history)
                        # Re-run pipeline with full transcript so DDx uses medication context
                        full_transcript = "".join(st.session_state.transcript)
                        if full_transcript.strip():
                            rerun_analysis = st.session_state.pipeline.run(full_transcript)
                            if rerun_analysis.updateUi:
                                st.session_state.ai_analysis = rerun_analysis
                        st.rerun()
                
                if new_ph != st.session_state.patient_history:
                    st.session_state.patient_history = new_ph
                    # Update the panel immediately so the doctor sees it
                    # without waiting for a full st.rerun()
                    render_patient_history(is_live=True)

                if new_analysis.updateUi:
                    # Merge DDx — never cancel existing diagnoses
                    merged_ddx = _merge_ddx(
                        existing=st.session_state.ai_analysis.ddx,
                        incoming=new_analysis.ddx,
                    )
                    new_analysis = new_analysis.model_copy(
                        update={"ddx": merged_ddx}
                    )
                    st.session_state.ai_analysis = new_analysis
                    payload = st.session_state.ai_analysis

                # Rerun to refresh full UI (DDx, questions, etc.)
                st.rerun()

        time.sleep(0.1)
else:
    st.session_state.was_playing = False
