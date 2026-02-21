import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import data
from assistant import analyze_consultation
from audio_processing import AudioStreamingProcessor, transcribe_audio_chunk, get_speaker_role
import queue

# Load environment variables from .env
load_dotenv()

# Removed old transcribe_audio_diarized as it is now in audio_processing.py

# Set page config for wide layout and custom title
st.set_page_config(
    page_title="MedAssist - AI Medical Co-Pilot",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State Variables
if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = {
        "suggested_medications": [],
        "reminders": [],
        "summary_notes": "Awaiting conversation start..."
    }
    
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = None

# Custom CSS for Premium Medical Styling
def load_css():
    st.markdown("""
        <style>
        /* Global typography & background */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
        
        .stApp {
            background-color: #FAFCFF; /* Light clinical blue tint */
        }
        
        * {
            font-family: 'Inter', sans-serif !important;
        }
        
        h1, h2, h3, h4, .stMarkdown p strong {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Headers and distinct colors */
        h1 {
            color: #0F172A;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #2563EB, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
            padding-top: 2rem;
        }
        
        /* Allergy Highlight */
        .allergy-warning {
            color: #B91C1C;
            font-weight: 600;
            background-color: #FEE2E2;
            padding: 4px 10px;
            border-radius: 99px;
            font-size: 0.9em;
            display: inline-block;
            margin-top: 4px;
        }

        /* Medical Alert Badge */
        .medical-badge {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #3B82F6;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
            color: #334155;
            font-size: 0.95em;
        }

        /* AI Assistant Section */
        .ai-panel-container {
            background: linear-gradient(145deg, #ffffff, #f0f4f8);
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.05), inset 0 2px 0 rgba(255, 255, 255, 0.7);
            border: 1px solid #E2E8F0;
            height: 100%;
            transition: all 0.3s ease;
        }
        
        .ai-panel-container:hover {
            box-shadow: 0 15px 35px rgba(37, 99, 235, 0.08);
        }

        .ai-header {
            color: #1E40AF;
            border-bottom: 2px solid #BFDBFE;
            padding-bottom: 12px;
            margin-bottom: 20px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Refined elements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500;
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.2s;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            color: white;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Sidebar: Patient Selection
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387560.png", width=70)
    st.markdown("## MedAssist AI")
    st.caption("Intelligent Clinical Co-Pilot")
    st.markdown("<br>", unsafe_allow_html=True)
    
    patients = data.get_all_patients()
    patient_options = {p["id"]: f"{p['name']} (ID: {p['id']})" for p in patients}
    
    selected_pid = st.selectbox(
        "Current Patient File",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options[x]
    )
    
    # If a new patient is selected, clear the transcript
    if selected_pid != st.session_state.current_patient_id:
        st.session_state.current_patient_id = selected_pid
        st.session_state.transcript = []
        st.session_state.ai_analysis = {
            "suggested_medications": [],
            "reminders": [],
            "summary_notes": "Awaiting conversation start..."
        }

    active_patient = data.get_patient_by_id(selected_pid)
    
    # Patient Profile Sidebar Details
    if active_patient:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {active_patient['name']}")
        
        col_age, col_gen = st.columns(2)
        col_age.metric("Age", active_patient['age'])
        col_gen.metric("Gender", active_patient['gender'])
        
        # Display Allergies with Highlight
        allergies_str = ", ".join(active_patient["allergies"])
        if "None" not in active_patient["allergies"]:
            st.markdown(f"**Critical Allergies**<br><span class='allergy-warning'>⚠️ {allergies_str}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Allergies:** {allergies_str}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='medical-badge'><b>Past History:</b><br/>{', '.join(active_patient['past_medical_history'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='medical-badge'><b>Current Meds:</b><br/>{', '.join(active_patient['current_medications'])}</div>", unsafe_allow_html=True)


# Main Interface Layout
st.markdown(f"<h1>Consultation: {active_patient['name']}</h1>", unsafe_allow_html=True)
st.markdown("Real-time encounter tracking powered by Crusoe Qwen3 and ElevenLabs.")
st.markdown("<br>", unsafe_allow_html=True)

col1, spacer, col2 = st.columns([1.6, 0.1, 1.2])

# Left Column: Conversation Tracking
with col1:
    st.markdown("### 🗣️ Live Transcript")
    
    # Container for Chat History using native st.chat_message
    chat_container = st.container(height=450, border=True)
    with chat_container:
        if not st.session_state.transcript:
            st.info("No dialogue recorded yet. Start the consultation below.")
        else:
            for speaker, text in st.session_state.transcript:
                avatar = "👨‍⚕️" if speaker == "Doctor" else "🤕"
                with st.chat_message(speaker.lower(), avatar=avatar):
                    st.write(f"**{speaker}:** {text}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Area - Text & Audio
    input_container = st.container()
    with input_container:
        text_tab, audio_tab = st.tabs(["⌨️ Type Note", "🎙️ Record Voice"])
        
        with text_tab:
            with st.form("dialogue_form", clear_on_submit=True, border=False):
                dialogue_input = st.text_input("Enter speech text...", placeholder="e.g. Doctor: How long have you had this cough?", label_visibility="collapsed")
                submitted = st.form_submit_button("Submit & Analyze")
                
                if submitted and dialogue_input.strip():
                    # Parse speaker from text if provided (e.g. "Doctor: Hello")
                    input_text = dialogue_input.strip()
                    speaker = "Doctor"
                    if ":" in input_text:
                        parts = input_text.split(":", 1)
                        if parts[0].strip().lower() in ["doctor", "patient"]:
                            speaker = parts[0].strip().capitalize()
                            input_text = parts[1].strip()
                            
                    st.session_state.transcript.append((speaker, input_text))
                    
                    full_transcript_str = "\\n".join([f"{s}: {t}" for s, t in st.session_state.transcript])
                    patient_context_str = data.format_patient_summary(active_patient)
                    
                    with st.spinner("Qwen3 is analyzing..."):
                        st.session_state.ai_analysis = analyze_consultation(patient_context_str, full_transcript_str)
                    st.rerun()
                        
            with audio_tab:
                st.caption("Live streaming enabled. Speak and your words will appear automatically.")
                webrtc_ctx = webrtc_streamer(
                    key="speech_to_text",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=AudioStreamingProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
                
                if webrtc_ctx.state.playing:
                    status_placeholder = st.empty()
                    status_placeholder.info("Listening... (processing every 3.5 seconds)")
                    
                    # Prevent strict blocking, allow streamlit to breathe
                    import time
                    while webrtc_ctx.state.playing:
                        if webrtc_ctx.audio_processor:
                            try:
                                audio_chunk = webrtc_ctx.audio_processor.audio_queue.get(timeout=0.5)
                                with st.spinner("Processing chunk..."):
                                    # 1. Identify speaker via Pyannote embedding
                                    role = get_speaker_role(audio_chunk["waveform"], audio_chunk["sample_rate"])
                                    
                                    # 2. Transcribe plain text via ElevenLabs STT
                                    txt = transcribe_audio_chunk(audio_chunk["wav_bytes"])
                                    
                                if txt and txt.strip():
                                    st.session_state.transcript.append((role, txt.strip()))
                                    
                                    # Re-analyze with new context
                                    full_transcript_str = "\\n".join([f"{s}: {t}" for s, t in st.session_state.transcript])
                                    patient_context_str = data.format_patient_summary(active_patient)
                                    st.session_state.ai_analysis = analyze_consultation(patient_context_str, full_transcript_str)
                                    st.rerun()
                                                
                            except queue.Empty:
                                pass
                        time.sleep(0.1)

# Right Column: AI Medical Assistant
with col2:
    st.markdown("<div class='ai-panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3 class='ai-header'>✨ Clinical Co-Pilot</h3>", unsafe_allow_html=True)
    
    analysis = st.session_state.ai_analysis
    
    # Use tabs for a cleaner, premium look
    tab1, tab2, tab3 = st.tabs(["💊 Meds", "🔔 Actions", "📝 Notes"])
    
    with tab1:
        st.markdown("#### Suggested Medications")
        if analysis.get("suggested_medications"):
            for med in analysis["suggested_medications"]:
                st.success(f"**{med}**", icon="✅")
        else:
            st.caption("No suggestions yet. Provide more context in the transcript.")
            
    with tab2:
        st.markdown("#### Clinical Reminders")
        if analysis.get("reminders"):
            for reminder in analysis["reminders"]:
                st.warning(f"**{reminder}**", icon="⚠️")
        else:
            st.caption("No active reminders.")
            
    with tab3:
        st.markdown("#### Encounter Summary")
        st.info(analysis.get("summary_notes", ""))
    
    st.markdown("</div>", unsafe_allow_html=True)

