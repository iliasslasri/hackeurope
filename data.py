import json
import os
import glob

PATIENTS_DIR = "patients"

def get_all_patients():
    """Returns a list of all mock patients loaded from JSON files."""
    patients = []
    if not os.path.exists(PATIENTS_DIR):
        return patients
        
    for filepath in glob.glob(f"{PATIENTS_DIR}/*.json"):
        with open(filepath, 'r') as f:
            try:
                patients.append(json.load(f))
            except json.JSONDecodeError:
                pass
    return patients

def get_patient_by_id(patient_id):
    """Retrieves a specific patient by loading their JSON file equivalent."""
    filepath = os.path.join(PATIENTS_DIR, f"{patient_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def format_patient_summary(patient):
    """Creates a formatted markdown string of the patient's medical summary."""
    if not patient:
        return "Patient not found."
    
    summary = f"**Age/Gender**: {patient.get('age', 'N/A')} / {patient.get('gender', 'N/A')}\n\n"
    summary += f"**Allergies**: {(', ').join(patient.get('allergies', []))}\n\n"
    summary += f"**Past Medical History**: {(', ').join(patient.get('past_medical_history', []))}\n\n"
    summary += f"**Current Medications**: {(', ').join(patient.get('current_medications', []))}\n"
    return summary
