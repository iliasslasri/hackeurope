import os
import base64
import requests
import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
import data
from assistant import analyze_consultation

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
    st.session_state.ai_analysis = {
        "suggested_medications": [], "reminders": [],
        "summary_notes": "Awaiting conversation start...",
        "differential_diagnosis": [], "clinical_gaps": []
    }
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = "P001"
if "speaker_mode" not in st.session_state:
    st.session_state.speaker_mode = "Doctor"
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "audio_b64" not in st.session_state:
    st.session_state.audio_b64 = None

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

    # Build transcript HTML
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

    st.markdown(f'<div class="transcript-panel">{transcript_html}</div>',
                unsafe_allow_html=True)


    # ── Voice Recording Button ────────────────────────────────────────────────
    # Custom HTML/JS mic button: click to record, auto-stops after 5s silence.
    # Audio is sent back to Streamlit via query params.
    mic_js = """
    <div id="mic-container" style="display:flex;align-items:center;gap:0.75rem;margin-top:0.5rem;">
        <button id="micBtn" class="mic-btn" onclick="toggleRecording()" title="Click to record">
            🎙️
        </button>
        <span id="micStatus" style="font-size:0.8rem;color:#64748b;">Click mic to start recording</span>
    </div>
    <script>
    let mediaRecorder = null;
    let audioChunks = [];
    let silenceTimer = null;
    let analyserNode = null;
    let audioCtx = null;
    let animFrame = null;
    let isRecording = false;
    const SILENCE_THRESHOLD = 0.015;
    const SILENCE_TIMEOUT = 5000;

    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioCtx = new AudioContext();
            const source = audioCtx.createMediaStreamSource(stream);
            analyserNode = audioCtx.createAnalyser();
            analyserNode.fftSize = 2048;
            source.connect(analyserNode);

            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                stream.getTracks().forEach(t => t.stop());
                if (audioCtx) { audioCtx.close(); audioCtx = null; }
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                sendAudio(blob);
            };

            mediaRecorder.start(250);
            isRecording = true;
            document.getElementById('micBtn').classList.add('recording');
            document.getElementById('micStatus').textContent = '🔴 Recording… (auto-stops after 5s silence)';
            document.getElementById('micStatus').style.color = '#ef4444';
            monitorSilence();
        } catch (err) {
            document.getElementById('micStatus').textContent = '⚠️ Mic access denied';
            document.getElementById('micStatus').style.color = '#f59e0b';
        }
    }

    function stopRecording() {
        isRecording = false;
        if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
        if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
        if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); }
        document.getElementById('micBtn').classList.remove('recording');
        document.getElementById('micStatus').textContent = '⏳ Processing audio…';
        document.getElementById('micStatus').style.color = '#4f46e5';
    }

    function monitorSilence() {
        const bufLen = analyserNode.frequencyBinCount;
        const data = new Float32Array(bufLen);
        let lastSoundTime = Date.now();

        function check() {
            if (!isRecording) return;
            analyserNode.getFloatTimeDomainData(data);
            let rms = 0;
            for (let i = 0; i < bufLen; i++) rms += data[i] * data[i];
            rms = Math.sqrt(rms / bufLen);

            if (rms > SILENCE_THRESHOLD) {
                lastSoundTime = Date.now();
            } else if (Date.now() - lastSoundTime > SILENCE_TIMEOUT) {
                stopRecording();
                return;
            }
            animFrame = requestAnimationFrame(check);
        }
        check();
    }

    function sendAudio(blob) {
        const reader = new FileReader();
        reader.onloadend = () => {
            const b64 = reader.result.split(',')[1];
            // Send to Streamlit via query param update
            const url = new URL(window.parent.location);
            url.searchParams.set('audio_b64', b64);
            url.searchParams.set('audio_ts', Date.now().toString());
            window.parent.history.replaceState({}, '', url);
            // Trigger Streamlit rerun
            window.parent.document.querySelectorAll('iframe').forEach(f => {
                try { f.contentWindow.postMessage({type:'streamlit:setComponentValue', value: b64}, '*'); } catch(e){}
            });
            // Force rerun via a small hack — click a hidden button if it exists
            document.getElementById('micStatus').textContent = '✅ Audio captured — reloading…';
            document.getElementById('micStatus').style.color = '#059669';
            // Use Streamlit's setComponentValue
            window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: b64}, '*');
            // Fallback: set in sessionStorage and trigger via fragment
            window.parent.sessionStorage.setItem('aura_audio_b64', b64);
            window.parent.location.hash = 'audio_ready_' + Date.now();
        };
        reader.readAsDataURL(blob);
    }
    </script>
    """
    st_html(mic_js, height=55)

    # Check for audio data from the JS component via query params
    query_params = st.query_params
    audio_b64 = query_params.get("audio_b64", None)
    audio_ts = query_params.get("audio_ts", None)

    if audio_b64 and audio_ts:
        # Avoid re-processing the same audio
        if st.session_state.get("last_audio_ts") != audio_ts:
            st.session_state.last_audio_ts = audio_ts
            # Clear query params
            st.query_params.clear()

            # Decode and transcribe
            audio_bytes = base64.b64decode(audio_b64)
            with st.spinner("🎙️ Transcribing audio..."):
                txt = transcribe_audio(audio_bytes)

            if txt and not txt.startswith("[Transcription"):
                st.session_state.transcript.append(
                    (st.session_state.speaker_mode, txt))
                st.session_state.speaker_mode = (
                    "Patient" if st.session_state.speaker_mode == "Doctor" else "Doctor")

                ts = "\n".join([f"{s}: {t}" for s, t in st.session_state.transcript])
                ps = data.format_patient_summary(active_patient)
                with st.spinner("Analyzing..."):
                    st.session_state.ai_analysis = analyze_consultation(ps, ts)
                st.rerun()
            else:
                st.error(txt or "No speech detected.")
        else:
            # Already processed, just clear params
            st.query_params.clear()


# ─── RIGHT: DDx & Clinical Gaps (col-span-2) ────────────────────────────────
with col_right:
    analysis = st.session_state.ai_analysis

    # ── DDx Tracker ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Differential Diagnosis (DDx)</div>',
                unsafe_allow_html=True)

    ddx_items = analysis.get("differential_diagnosis", [])
    if ddx_items:
        html = ""
        for item in ddx_items:
            cond = item.get("condition", "Unknown")
            sev = item.get("severity", "Low").lower()
            cls = sev if sev in ("high", "medium", "low") else "low"
            html += f"""
            <div class="ddx-card {cls}">
                <span class="condition">{cond}</span>
                <span class="ddx-badge {cls}">{sev.capitalize()}</span>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ddx-card low" style="opacity:.4; cursor:default">
            <span class="condition" style="color:#94a3b8; font-weight:400">
            Awaiting transcript data…</span>
        </div>""", unsafe_allow_html=True)

    # ── Clinical Gap ─────────────────────────────────────────────────────────
    gaps = analysis.get("clinical_gaps", []) or analysis.get("reminders", [])

    st.markdown("""
    <div class="clinical-gap-header">
        <span>💡</span> Clinical Gap Identified
    </div>""", unsafe_allow_html=True)

    if gaps:
        for g in gaps:
            st.markdown(f'<div class="clinical-gap-card"><p>{g}</p></div>',
                        unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="clinical-gap-card" style="opacity:.5">
            <p>Clinical gaps will appear here as the consultation progresses.</p>
        </div>""", unsafe_allow_html=True)

    # ── Suggested Medications ────────────────────────────────────────────────
    meds = analysis.get("suggested_medications", [])
    if meds:
        st.markdown("""
        <div class="clinical-gap-header">
            <span>💊</span> Suggested Medications
        </div>""", unsafe_allow_html=True)
        for m in meds:
            st.markdown(f"""
            <div class="ddx-card low">
                <span class="condition">{m}</span>
                <span class="ddx-badge low">Rx</span>
            </div>""", unsafe_allow_html=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = analysis.get("summary_notes", "")
    if summary and summary != "Awaiting conversation start...":
        st.markdown("""
        <div class="clinical-gap-header">
            <span>📝</span> Encounter Summary
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="clinical-gap-card"><p>{summary}</p></div>',
                    unsafe_allow_html=True)
