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
DDX_MAX = 7   # Maximum number of DDx entries to keep at any time


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

    # Enforce cap: drop oldest entry with lowest suspicion when over limit
    while len(result) > max_entries:
        # Sort ascending by suspicion rank so the weakest candidate is first;
        # among ties, the one with the highest rank number (added earliest /
        # ranked lowest) is chosen for removal.
        result.sort(
            key=lambda e: (
                _SUSPICION_RANK.get(e.suspicion.value, 0),
                -e.rank,
            )
        )
        result.pop(0)   # remove the lowest-confidence / oldest entry

    # Re-number ranks 1..N sorted by suspicion descending
    result.sort(key=lambda e: _SUSPICION_RANK.get(e.suspicion.value, 0), reverse=True)
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

/* bg-slate-50 */
.stApp { background-color: #f8fafc; }

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
[data-testid="stSidebar"]      { display: none !important; }
[data-testid="stHeader"]       { display: none !important; }
#MainMenu, footer              { visibility: hidden; }
[data-testid="stIconMaterial"] { display: none !important; }
.stMarkdown h3                 { display: none; }

/* max-w-7xl mx-auto, p-4 md:p-8 */
.block-container {
    padding: 2rem 2rem !important;
    max-width: 1280px;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
/* text-3xl font-bold text-slate-800 tracking-tight */
.aura-header h1 {
    font-size: 1.875rem;
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.025em;
    margin: 0; padding: 0; line-height: 1.2;
}
/* text-slate-500 */
.aura-subtitle {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 4px;
}

/* ── Section Titles ────────────────────────────────────────────────────────── */
/* text-xl font-semibold text-slate-800 */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
}

/* ── Transcript Container ──────────────────────────────────────────────────── */
/* bg-white border border-slate-200 rounded-xl p-6 shadow-sm */
.transcript-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 1.5rem;
    min-height: 380px;
    max-height: 420px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
}


/* ── Chat Messages ─────────────────────────────────────────────────────────── */
/* flex gap-4 */
.msg-row {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
/* flex-row-reverse for doctor (user) */
.msg-row.doctor { flex-direction: row-reverse; }
/* flex-row for patient (assistant) */
.msg-row.patient { flex-direction: row; }

/* Avatar: w-10 h-10 rounded-full flex items-center justify-center */
.avatar-icon {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1rem;
}
/* bg-indigo-100 text-indigo-600 */
.avatar-icon.doctor {
    background: #e0e7ff;
    color: #4f46e5;
}
/* bg-emerald-100 text-emerald-600 */
.avatar-icon.patient {
    background: #d1fae5;
    color: #059669;
}

/* Bubble: max-w-[80%] rounded-2xl px-5 py-3 */
.msg-bubble {
    max-width: 80%;
    border-radius: 1rem;
    padding: 0.75rem 1.25rem;
    font-size: 0.9375rem;
    line-height: 1.625;
    word-wrap: break-word;
}
/* bg-indigo-600 text-white rounded-tr-none */
.msg-bubble.doctor {
    background: #4f46e5;
    color: #ffffff;
    border-top-right-radius: 0;
}
/* bg-slate-100 text-slate-800 rounded-tl-none */
.msg-bubble.patient {
    background: #f1f5f9;
    color: #1e293b;
    border-top-left-radius: 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 5rem 1.5rem;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.75rem; }

/* ── Patient History Panel ─────────────────────────────────────────────────── */
.ph-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 1.5rem;
    min-height: 380px;
    overflow-y: auto;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}
/* section label */
.ph-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
/* symptom / risk chips */
.ph-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ph-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid;
    line-height: 1.4;
}
.ph-chip.symptom   { background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; }
.ph-chip.negated   { background:#f8fafc; color:#94a3b8; border-color:#e2e8f0;
                     text-decoration: line-through; }
.ph-chip.risk      { background:#fff7ed; color:#c2410c; border-color:#fed7aa; }
.ph-chip.med       { background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
/* meta row (duration / severity) */
.ph-meta-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.ph-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 0.35rem 0.8rem;
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
}
.ph-meta-pill strong { color: #1e293b; }
/* history prose box */
.ph-history-box {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 0.5rem 0.5rem 0;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #334155;
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

/* ── Form Input ────────────────────────────────────────────────────────────── */
/* bg-slate-50 border border-slate-200 rounded-lg pl-4 pr-12 py-3 */
.stTextInput > div > div > input {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important;
    box-shadow: none !important;
    color: #0f172a !important;
    transition: all 0.15s ease !important;
}
/* focus:ring-2 focus:ring-indigo-500 focus:border-transparent */
.stTextInput > div > div > input:focus {
    border-color: transparent !important;
    box-shadow: 0 0 0 2px #6366f1 !important;
    outline: none !important;
}
.stTextInput > div > div,
.stTextInput > div { border: none !important; box-shadow: none !important; }

/* text-indigo-600 */
.stFormSubmitButton > button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: #4f46e5 !important;
    font-size: 1.2rem !important;
    padding: 0.5rem !important;
    min-height: 0 !important;
    transition: all 0.15s ease !important;
}
/* hover:bg-indigo-50 */
.stFormSubmitButton > button:hover {
    background: #eef2ff !important;
    color: #4f46e5 !important;
    transform: none !important;
    box-shadow: none !important;
    border-radius: 0.375rem !important;
}

/* General button */
.stButton > button {
    background: #4f46e5; color: white;
    border: none; border-radius: 0.5rem;
    padding: 0.5rem 1rem; font-weight: 500; font-size: 0.85rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #4338ca;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    color: white; border: none;
}

/* ── DDx Cards ─────────────────────────────────────────────────────────────── */
/* p-4 rounded-xl border transition-all cursor-pointer */
.ddx-card {
    padding: 1rem;
    border-radius: 0.75rem;
    border-width: 1px;
    border-style: solid;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    transition: all 0.2s ease;
    cursor: pointer;
}
/* hover:shadow-md hover:-translate-y-0.5 hover:border-slate-300 */
.ddx-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
    transform: translateY(-2px);
    border-color: #cbd5e1;
}

/* High: bg-red-50 border-red-200 */
.ddx-card.high {
    background: #fef2f2;
    border-color: #fecaca;
}
/* Medium: bg-amber-50 border-amber-200 */
.ddx-card.medium {
    background: #fffbeb;
    border-color: #fde68a;
}
/* Low: bg-blue-50 border-blue-200 */
.ddx-card.low {
    background: #eff6ff;
    border-color: #bfdbfe;
}

/* font-medium + text color per severity */
.ddx-card .condition {
    font-weight: 500;
    font-size: 0.9375rem;
}
.ddx-card.high .condition  { color: #7f1d1d; }  /* text-red-900 */
.ddx-card.medium .condition { color: #78350f; } /* text-amber-900 */
.ddx-card.low .condition   { color: #1e3a5f; }  /* text-blue-900 */

/* Badge: text-xs font-semibold px-2.5 py-1 rounded-full border */
.ddx-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.625rem;
    border-radius: 9999px;
    border-width: 1px;
    border-style: solid;
    white-space: nowrap;
}
/* High badge: bg-red-100 text-red-700 border-red-200 */
.ddx-badge.high {
    background: #fee2e2;
    color: #b91c1c;
    border-color: #fecaca;
}
/* Medium badge: bg-amber-100 text-amber-700 border-amber-200 */
.ddx-badge.medium {
    background: #fef3c7;
    color: #b45309;
    border-color: #fde68a;
}
/* Low badge: bg-blue-100 text-blue-700 border-blue-200 */
.ddx-badge.low {
    background: #dbeafe;
    color: #1d4ed8;
    border-color: #bfdbfe;
}

/* ── Clinical Gap ──────────────────────────────────────────────────────────── */
/* text-xl font-semibold text-slate-800 flex items-center gap-2 */
.clinical-gap-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-top: 2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
/* bg-emerald-50 border border-emerald-200 rounded-xl p-5 shadow-sm */
.clinical-gap-card {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 0.75rem;
    padding: 1.25rem;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
}
/* text-emerald-900 font-medium leading-relaxed */
.clinical-gap-card p,
.clinical-gap-card {
    color: #064e3b;
    font-weight: 500;
    line-height: 1.625;
    font-size: 0.9375rem;
}

/* ── Mic Button ────────────────────────────────────────────────────────────── */
.mic-btn {
    width: 2.75rem; height: 2.75rem;
    border-radius: 9999px;
    border: 2px solid #e2e8f0;
    background: #ffffff;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s ease;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.mic-btn:hover {
    border-color: #4f46e5;
    background: #eef2ff;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
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
.mic-status {
    font-size: 0.75rem;
    color: #ef4444;
    font-weight: 500;
    margin-top: 2px;
    text-align: center;
}

/* ── Tabs (hide) ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0; }

/* ── Misc Streamlit overrides ──────────────────────────────────────────────── */
.stAlert {
    border-radius: 0.75rem;
    border: none;
}
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background: white;
    border: 1px solid #e2e8f0 !important;
    border-radius: 0.75rem !important;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
header_col1, header_col2 = st.columns([3, 2], gap="large")
with header_col1:
    st.markdown("""
    <div class="aura-header" style="margin-bottom:2rem">
        <h1>Aura</h1>
        <div class="aura-subtitle">Real-time Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.current_patient_id:
        active_patient_opt = data.get_patient_by_id(st.session_state.current_patient_id)
        p_name = active_patient_opt.get("name", "Unknown") if active_patient_opt else "Unknown"
        st.markdown(f'<div style="text-align:right; color:#0f172a; font-weight:600; font-size:1.1rem; padding-top:1rem;">Patient: {p_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right; color:#64748b; font-style:italic; padding-top:1rem;">Listening for patient name...</div>', unsafe_allow_html=True)

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
                <span class="condition">{entry.disease}</span>
                <span class="ddx-badge {cls}">{pct}</span>
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
            "rule_in":      ("#dcfce7", "#166534", "#bbf7d0"),
            "rule_out":     ("#fee2e2", "#991b1b", "#fecaca"),
            "differentiate":("#dbeafe", "#1e40af", "#bfdbfe"),
        }

        # Build one HTML block — hover reveals questions (no click needed)
        html = '<div class="clinical-gap-header"><span>❓</span> Targeted Questions</div>'
        html += '<style>'
        html += '.dq-card{position:relative;margin-bottom:0.5rem;border-radius:0.75rem;border:1px solid #e2e8f0;background:#fff;overflow:hidden;transition:box-shadow .2s;}'
        html += '.dq-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.1);}'
        html += '.dq-header{display:flex;align-items:center;gap:0.5rem;padding:0.65rem 1rem;font-size:0.875rem;font-weight:600;color:#1e293b;cursor:default;}'
        html += '.dq-questions{max-height:0;overflow:hidden;transition:max-height .35s ease,padding .2s;}'
        html += '.dq-card:hover .dq-questions{max-height:600px;padding-bottom:0.75rem;}'
        html += '.dq-q{margin:0 0.75rem 0.5rem;padding:0.65rem 0.9rem;border-radius:0.5rem;border:1px solid;}'
        html += '.dq-q-text{font-weight:500;font-size:0.875rem;margin-bottom:0.2rem;}'
        html += '.dq-q-rationale{font-size:0.78rem;color:#64748b;line-height:1.5;}'
        html += '.dq-q-tag{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-top:0.3rem;display:inline-block;}'
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
                        ph = _PH()
                        if matched_patient_data:
                            ph.patient_name = new_ph.patient_name
                            ph.symptoms = new_ph.symptoms
                            ph.risk_factors = matched_patient_data.get("past_medical_history", [])
                            ph.medications = matched_patient_data.get("current_medications", [])
                            ph.relevant_history = f"**{matched_patient_data.get('name', 'Unknown')}** ({matched_patient_data.get('age', 'N/A')} {matched_patient_data.get('gender', 'N/A')}). "
                            if matched_patient_data.get("allergies") and matched_patient_data["allergies"] != ["None"]:
                                ph.relevant_history += f"Allergies: {', '.join(matched_patient_data['allergies'])}. "
                                
                        st.session_state.patient_history = ph
                        st.session_state.pipeline = AuraPipeline(initial_history=st.session_state.patient_history)
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
