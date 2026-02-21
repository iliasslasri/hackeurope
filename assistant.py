import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the OpenAI API client for Crusoe
client = OpenAI(
    base_url='https://hackeurope.crusoecloud.com/v1/',
    api_key=os.getenv("CRUSOE_API_KEY"),
)

MODEL_NAME = 'NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4'

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
    1. "suggested_medications": A list of strings. Suggest appropriate medications for the conditions 
       discussed. CRITICAL: Do NOT suggest medications the patient is allergic to or that negatively 
       interact with their current medications.
    2. "reminders": A list of strings. Provide clinical reminders for the doctor (e.g., ask about 
       duration of symptoms, recommend specific lab tests, schedule follow-ups).
    3. "summary_notes": A short string summarizing the key points of the consultation so far.
    
    Output nothing but valid JSON.
    """
    
    try:
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
