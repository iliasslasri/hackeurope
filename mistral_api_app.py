import os
import time
import queue
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from backend.audio_processing import AudioStreamingProcessor

load_dotenv()

st.set_page_config(page_title="Mistral Voice Test", page_icon="🎙️", layout="centered")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, * { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(155deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important; }
[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

.title { text-align: center; margin-top: 2rem; }
.title h1 {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.subtitle { text-align: center; color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }

.transcript-box {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 1rem;
    padding: 1.5rem 2rem;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.8;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.transcript-box .placeholder { color: #475569; font-style: italic; }
.transcript-box .word { color: #f1f5f9; }

.status-bar {
    text-align: center; margin-top: 1rem;
    font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.status-live { color: #34d399; }
.status-idle { color: #64748b; }

.clear-btn { text-align: center; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="title"><h1>🎙️ Mistral Voice Test</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Voxtral Mini Transcribe Realtime</div>', unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""

# ── WebRTC Streamer ──────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="mistral_test",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioStreamingProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
)

is_recording = webrtc_ctx.state.playing

# ── Status ───────────────────────────────────────────────────────────────────
if is_recording:
    st.markdown('<div class="status-bar status-live">● Live — Speak now</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-bar status-idle">○ Click START to begin</div>', unsafe_allow_html=True)

# ── Drain text from the processor queue ──────────────────────────────────────
if is_recording and webrtc_ctx.audio_processor:
    processor = webrtc_ctx.audio_processor
    try:
        while True:
            text = processor.text_queue.get_nowait()
            if text:
                st.session_state.full_transcript += text + " "
    except queue.Empty:
        pass

# ── Transcript Display ──────────────────────────────────────────────────────
transcript = st.session_state.full_transcript.strip()
if transcript:
    st.markdown(f'<div class="transcript-box"><span class="word">{transcript}</span></div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="transcript-box"><span class="placeholder">'
        'Your transcription will appear here as you speak...</span></div>',
        unsafe_allow_html=True,
    )

# ── Clear Button ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Transcript", use_container_width=True):
        st.session_state.full_transcript = ""
        st.rerun()

# ── Auto-refresh while recording ────────────────────────────────────────────
if is_recording:
    time.sleep(0.5)
    st.rerun()
