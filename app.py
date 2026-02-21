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
from backend.agents import AuraPipeline
from backend.audio_processing import AudioStreamingProcessor

load_dotenv()

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
if "pipeline" not in st.session_state:
    st.session_state.pipeline = AuraPipeline()
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = "P001"
if "speaker_mode" not in st.session_state:
    st.session_state.speaker_mode = "Doctor"

if "was_playing" not in st.session_state:
    st.session_state.was_playing = False
if "last_speech_time" not in st.session_state:
    st.session_state.last_speech_time = time.time()
if "transcript_changed_since_llm" not in st.session_state:
    st.session_state.transcript_changed_since_llm = False

active_patient = data.get_patient_by_id(st.session_state.current_patient_id)

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
st.markdown("""
<div class="aura-header" style="margin-bottom:2rem">
    <h1>Aura</h1>
    <div class="aura-subtitle">Real-time Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — grid-cols-5: 3/5 left, 2/5 right, gap-8
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ─── LEFT: Live Transcript (col-span-3) ──────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">Consultation Transcript</div>',
                unsafe_allow_html=True)

    transcript_placeholder = st.empty()
    
    def render_transcript():
        transcript_html = ""
        if not st.session_state.transcript:
            transcript_html = """
            <div class="empty-state">
                <div class="icon">💬</div>
                <div>No dialogue recorded yet.<br>Start the consultation below.</div>
            </div>"""
        else:
            for speaker, text in st.session_state.transcript:
                if speaker == "Doctor":
                    # Doctor = user role → flex-row-reverse, indigo avatar, indigo bubble
                    transcript_html += f"""
                    <div class="msg-row doctor">
                        <div class="avatar-icon doctor">🩺</div>
                        <div class="msg-bubble doctor">{text}</div>
                    </div>"""
                else:
                    # Patient = assistant role → flex-row, emerald avatar, slate bubble
                    transcript_html += f"""
                    <div class="msg-row patient">
                        <div class="avatar-icon patient">👤</div>
                        <div class="msg-bubble patient">{text}</div>
                    </div>"""
        transcript_placeholder.markdown(f'<div class="transcript-panel" id="transcript-auto-scroll">{transcript_html}</div>', unsafe_allow_html=True)

    render_transcript()

    # ── Voice Recording Button (WebRTC) ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🔴 Live Audio Stream (Transcribes with Gradium - analyzes after 5s silence)")
    webrtc_ctx = webrtc_streamer(
        key="speech_to_text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioStreamingProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True}
    )
    
    if webrtc_ctx.state.playing:
        st.session_state.was_playing = True
        status_placeholder = st.empty()
        status_placeholder.info("Listening... (streaming to Gradium API)")
        
        while webrtc_ctx.state.playing:
            if webrtc_ctx.audio_processor:
                collected_text = False
                try:
                    # Drain the queue to get real-time words
                    while True:
                        text_item = webrtc_ctx.audio_processor.text_queue.get_nowait()
                        if text_item and text_item.strip():
                            # If we have no transcript, start a new tuple
                            if not st.session_state.transcript:
                                st.session_state.transcript.append(("Doctor", text_item.strip() + " "))
                            else:
                                # Append to the last tuple to avoid spawning 1000s of chat bubbles
                                speaker, existing_text = st.session_state.transcript[-1]
                                st.session_state.transcript[-1] = (speaker, existing_text + text_item.strip() + " ")
                                
                            collected_text = True
                except queue.Empty:
                    pass
                    
                if collected_text:
                    st.session_state.last_speech_time = time.time()
                    st.session_state.transcript_changed_since_llm = True
                    # Instantly update the UI placeholder without blocking the thread
                    render_transcript()
                    
                # If 5 seconds of silence happened, auto-analyze
                if st.session_state.transcript_changed_since_llm and (time.time() - st.session_state.last_speech_time > 5.0):
                    st.session_state.transcript_changed_since_llm = False
                    status_placeholder.info("Silence detected. AI is analyzing consultation...")
                    
                    # Construct the full_transcript exactly like before
                    full_transcript = "\\n".join(
                        [f"{s}: {t}" for s, t in st.session_state.transcript]
                    )
                    
                    
                    new_analysis = st.session_state.pipeline.run(full_transcript)
                    if new_analysis.updateUi:
                        st.session_state.ai_analysis = new_analysis
                    st.rerun()
                    
            time.sleep(0.1)
    else:
        st.session_state.was_playing = False


# ─── RIGHT: DDx & Clinical Gaps (col-span-2) ────────────────────────────────
with col_right:
    payload: AuraUIPayload = st.session_state.ai_analysis

    # ── DDx Tracker ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Differential Diagnosis (DDx)</div>',
                unsafe_allow_html=True)

    if payload.updateUi and payload.ddx:
        html = ""
        for entry in payload.ddx:
            sev = entry.suspicion.value.lower()   # "high" / "medium" / "low"
            cls = sev if sev in ("high", "medium", "low") else "low"
            html += f"""
            <div class="ddx-card {cls}">
                <span class="condition">{entry.disease}</span>
                <span class="ddx-badge {cls}">{sev.capitalize()}</span>
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

    if payload.updateUi and payload.follow_up_question:
        st.markdown(
            f'<div class="clinical-gap-card"><p>{payload.follow_up_question}</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="clinical-gap-card" style="opacity:.5">
            <p>Clinical gaps will appear here as the consultation progresses.</p>
        </div>""", unsafe_allow_html=True)

    # ── Safety Issues ────────────────────────────────────────────────────────
    if payload.updateUi and payload.safety_issues:
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
    if payload.updateUi and payload.questions_by_disease:
        st.markdown("""
        <div class="clinical-gap-header">
            <span>❓</span> Targeted Questions
        </div>""", unsafe_allow_html=True)

        for dq in payload.questions_by_disease:
            with st.expander(f"🔬 {dq.disease}", expanded=False):
                for q in dq.questions:
                    target_colors = {
                        "rule_in": ("#dcfce7", "#166534", "#bbf7d0"),
                        "rule_out": ("#fee2e2", "#991b1b", "#fecaca"),
                        "differentiate": ("#dbeafe", "#1e40af", "#bfdbfe"),
                    }
                    bg, fg, border = target_colors.get(
                        q.target.value, ("#f1f5f9", "#334155", "#e2e8f0"))
                    st.markdown(f"""
                    <div style="background:{bg}; border:1px solid {border};
                                border-radius:0.5rem; padding:0.75rem 1rem;
                                margin-bottom:0.5rem;">
                        <div style="font-weight:500; color:{fg}; font-size:0.9rem;
                                    margin-bottom:0.25rem;">
                            {q.question}
                        </div>
                        <div style="font-size:0.8rem; color:#64748b; line-height:1.5;">
                            {q.clinical_rationale}
                        </div>
                        <span style="font-size:0.7rem; font-weight:600;
                                     color:{fg}; text-transform:uppercase;
                                     letter-spacing:0.05em;">
                            {q.target.value.replace('_', ' ')}
                        </span>
                    </div>""", unsafe_allow_html=True)
