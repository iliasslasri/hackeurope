# Aura - Real-time Clinical Decision Support

> *"For every hour a doctor spends looking at a patient, they spend two hours looking at a computer. We are bringing the physician's eyes back to the patient."*

Aura is an invisible, real-time Clinical Co-Pilot. It is an autonomous diagnostic engine that structures clinical data and surfaces physician-grade insights **with zero clicks**. Built for clinical hackathons and rapid prototyping, Aura listens to the live doctor-patient dialogue and acts as an intelligent sounding board.

## The "Wow" Factor

1. **Ambient Identity Detection**: No dropdowns, no searching. Say, *"Hi John, how are you today?"* and our **TriageGenie** LLM parses the audio, identifies the patient, and instantly summons their medical history and risk factors.
2. **Zero-Click Phenotyping**: As the conversation flows, Aura actively extracts and structures the clinical array, distinguishing between positive symptoms and negated symptoms.
3. **Live Bayesian DDx Engine**: The moment Aura detects enough clinical signal, the locally integrated **Medical Diagnosis Agent** maps symptoms against pre-existing risk factors to render a continuously re-ranking Differential Diagnosis (DDx) leaderboard.
4. **Closing the Diagnostic Gap**: Our **QuestionGenie** (powered by Crusoe Cloud's Qwen3) analyzes top disease candidates and dynamically generates highly targeted questions for the doctor to ask.
5. **Autonomic Feedback Loop**: When the doctor asks a suggested question, the **QAGenie** captures the patient's answer, recalculates the probabilistic weight, and updates the DDx leaderboard instantly.

---

## Tech Stack

*   **Frontend & Orchestration:** Streamlit, `streamlit-webrtc`
*   **Audio Transcription:** ElevenLabs (Scribe v1) via WebRTC
*   **LLM Provider:** Crusoe Cloud Compute (Qwen3-72B Instruct)
*   **Diagnostic Scoring:** Custom local `MedicalDiagnosisAgent` utilizing semantic similarity
*   **Data Models:** Pydantic (Strict JSON enforcement)

## Project Structure

*   `app.py`: The main Streamlit frontend application, housing the WebRTC connection, UI rendering layout, and the asynchronous event loop driving the AI analysis.
*   `audio_processing.py`: Handles audio byte streams, interactions with the ElevenLabs STT API, and Pyannote diarization setup.
*   `data.py`: A lightweight mock database utility. Actively scans the `patients/` directory to load any JSON profiles.
*   **`backend/`**
    *   `main_agent.py`: The `AuraPipeline` orchestrator unifying triage, diagnosis, and questioning.
    *   `triageGenie.py`: Extracts patient identity and parses the running transcript into structured clinical arrays.
    *   `questionGenie.py`: Computes the highest-yield follow-up questions to bridge the clinical gap.
    *   `qaGenie.py`: Scans live text for explicit Question & Answer sequences successfully executed by the physician and patient.
    *   `schemas.py`: Rigid Pydantic classes defining the API contract.
    *   `prompts.py`: Centralized LLM system prompts for the various AI personas.

---

## Setup & Running

1. **Clone & Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory.
   ```ini
   ELEVENLABS_API_KEY="..."
   CRUSOE_API_KEY="..."
   CRUSOE_URL="https://api.crusoecloud.com/v1/projects/.../chat/completions"
   HF_TOKEN="..." # For Pyannote Speaker Diarization
   ```

3. **Generate Local SSL Certificates**
   WebRTC requires HTTPS to access the microphone securely in local browsers.
   ```bash
   openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem
   ```

4. **Add Patient Profiles**
   Create JSON files inside the `patients/` folder following the `john_doe.json` signature.

5. **Start the App**
   ```bash
   streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
   ```

---

## Live Demo Guide

**Pro-Tip:** Start the demo with your hands completely off the keyboard.

1. Launch the app in your browser and accept microphone permissions.
2. Look at the judges and say: *"Let me show you how many clicks it takes to pull up a chart, document symptoms, and formulate a differential diagnosis..."*
3. Introduce the patient naturally: *"Hello John Doe, how are we doing today?"* - Note that the top right identity corner will snap from "Listening..." to "Patient: John Doe" and pull his allergies and medications.
4. State >2 distinct symptoms: *"So I hear you've been feeling deeply fatigued and constantly cold?"*
5. Watch the DDx leaderboard calculate and wait for the AI to spawn targeted follow up questions.
6. Ask one of the suggested questions: *"Have you noticed any changes in your bowel movements lately?"* and reply with a definitive answer to watch the `QAGenie` recalibrate the diagnosis automatically!
