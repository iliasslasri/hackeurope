# MedAssist - AI Medical Co-Pilot

MedAssist is a real-time medical consultation assistant designed to support doctors during patient encounters. It leverages AI to transcribe dialogue, provide clinical reminders, suggest medications while checking for allergies, and summarize the consultation.

## Project Overview

- **Purpose:** Intelligent clinical co-pilot for real-time consultation tracking and analysis.
- **Technologies:**
    - **Frontend:** [Streamlit](https://streamlit.io/)
    - **Transcription:** [ElevenLabs Speech-to-Text](https://elevenlabs.io/) (`scribe_v1`)
    - **LLM Analysis:** [Crusoe Cloud](https://crusoecloud.com/) using the `Qwen/Qwen3` model (via an OpenAI-compatible API).
    - **Data Management:** Python/Pandas with mock patient data.

## Project Structure

- `app.py`: Main entry point for the Streamlit application. Handles the UI, session state, and transcription integration.
- `assistant.py`: Contains the logic for interacting with the LLM (Crusoe Cloud) to perform clinical analysis of the consultation transcript.
- `data.py`: Manages the mock patient database and provides utility functions for retrieving and formatting patient information.
- `requirements.txt`: Lists all Python dependencies.

## Setup and Running

### Prerequisites
- Python 3.8+
- API keys for ElevenLabs and Crusoe Cloud.

### Installation
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory and add your API keys:
   ```env
   ELEVENLABS_API_KEY=your_elevenlabs_key
   CRUSOE_API_KEY=your_crusoe_key
   ```

### Running the Application
To start the MedAssist assistant, run:
```bash
streamlit run app.py
```

## Development Conventions

- **State Management:** Uses Streamlit's `st.session_state` to maintain the consultation transcript and AI analysis results across reruns.
- **Styling:** Custom CSS is embedded in `app.py` to provide a "Premium Medical" aesthetic using the Outfit and Inter fonts.
- **Mock Data:** Patient data is currently hardcoded in `data.py`. For production, this would be replaced with a real EMR/EHR integration.
- **LLM Prompting:** The prompt in `assistant.py` strictly enforces a JSON response format for easy parsing and UI integration.

## Key Features

- **Live Transcript:** Supports both manual text input and audio recording (transcribed via ElevenLabs).
- **Patient Context:** Automatically pulls in patient history, including critical allergies and current medications.
- **AI Co-Pilot Panel:**
    - **Meds:** Real-time medication suggestions filtered against known patient allergies.
    - **Actions:** Clinical reminders for the doctor (e.g., suggested tests or follow-up questions).
    - **Notes:** Automated summary of the encounter.
