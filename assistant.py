import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = 'NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4'

_client = OpenAI(
    base_url='https://hackeurope.crusoecloud.com/v1/',
    api_key=os.getenv("CRUSOE_API_KEY"),
)

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("CRUSOE_API_KEY")
        if not api_key:
            return None
        _client = OpenAI(
            base_url='https://hackeurope.crusoecloud.com/v1/',
            api_key=api_key,
        )
    return _client

def analyze_consultation(patient_summary, transcript):
    """
    Analyzes the doctor-patient consultation transcript and patient history.
    Returns a structured dictionary with suggested medications, reminders, and summary notes.
    """
    if not transcript.strip():
        return {
            "suggested_medications": [],
            "reminders": [],
            "summary_notes": "Awaiting conversation..."
        }

    prompt = f"""
    You are an expert AI medical assistant designed to help doctors during patient consultations.
    
    PATIENT MEDICAL HISTORY:
    {patient_summary}
    
    CURRENT CONSULTATION TRANSCRIPT:
    {transcript}
    
    Based on the patient's medical history (especially allergies and current medications) and the 
    ongoing conversation, please provide real-time assistance to the doctor.
    
    Respond STRICTLY in JSON format with the following keys:
    1. "differential_diagnosis": A list of objects, each with "condition" (string) and "severity" 
       (string: "High", "Medium", or "Low"). List 2-4 possible diagnoses ranked by likelihood.
    2. "clinical_gaps": A list of strings. Identify gaps in the clinical assessment — suggest 
       questions the doctor should ask or tests to consider. Keep each item concise (1-2 sentences).
    3. "suggested_medications": A list of strings. Suggest appropriate medications for the conditions 
       discussed. CRITICAL: Do NOT suggest medications the patient is allergic to or that negatively 
       interact with their current medications.
    4. "reminders": A list of strings. Provide clinical reminders for the doctor (e.g., ask about 
       duration of symptoms, recommend specific lab tests, schedule follow-ups).
    5. "summary_notes": A short string summarizing the key points of the consultation so far.
    
    Output nothing but valid JSON.
    """
    
    try:
        client = _get_client()
        if client is None:
            return {
                "differential_diagnosis": [],
                "clinical_gaps": ["API key not configured. Set CRUSOE_API_KEY in your .env file."],
                "suggested_medications": [],
                "reminders": [],
                "summary_notes": "API key not configured."
            }
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        # Extract and parse the JSON response
        text = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if present
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        result = json.loads(text.strip())
        return result
    except Exception as e:
        # Fallback if there's an API error or parsing issue
        print(f"Error accessing LLM: {e}")
        return {
            "error": str(e),
            "suggested_medications": [],
            "reminders": ["An error occurred while connecting to the AI assistant."],
            "summary_notes": "Unable to generate summary due to API error."
        }

