import pandas as pd

# Mock patient database for the Streamlit hackathon app

MOCK_PATIENTS = [
    {
        "id": "P001",
        "name": "John Doe",
        "age": 45,
        "gender": "Male",
        "allergies": ["Penicillin", "Peanuts"],
        "past_medical_history": ["Hypertension", "Type 2 Diabetes"],
        "current_medications": ["Lisinopril 10mg", "Metformin 500mg"],
    },
    {
        "id": "P002",
        "name": "Jane Smith",
        "age": 32,
        "gender": "Female",
        "allergies": ["None"],
        "past_medical_history": ["Asthma"],
        "current_medications": ["Albuterol Inhaler PRN"],
    },
    {
        "id": "P003",
        "name": "Robert Johnson",
        "age": 68,
        "gender": "Male",
        "allergies": ["Sulfa drugs"],
        "past_medical_history": ["Coronary Artery Disease", "Hyperlipidemia"],
        "current_medications": ["Aspirin 81mg", "Atorvastatin 40mg"],
    },
    {
        "id": "P004",
        "name": "Emily Davis",
        "age": 28,
        "gender": "Female",
        "allergies": ["Latex"],
        "past_medical_history": ["Migraines"],
        "current_medications": ["Sumatriptan 50mg PRN"],
    },
]

def get_all_patients():
    """Returns a list of all mock patients."""
    return MOCK_PATIENTS

def get_patient_by_id(patient_id):
    """Retrieves a specific patient by their ID."""
    for patient in MOCK_PATIENTS:
        if patient["id"] == patient_id:
            return patient
    return None

def format_patient_summary(patient):
    """Creates a formatted markdown string of the patient's medical summary."""
    if not patient:
        return "Patient not found."
    
    summary = f"**Age/Gender**: {patient['age']} / {patient['gender']}\n\n"
    summary += f"**Allergies**: {(', ').join(patient['allergies'])}\n\n"
    summary += f"**Past Medical History**: {(', ').join(patient['past_medical_history'])}\n\n"
    summary += f"**Current Medications**: {(', ').join(patient['current_medications'])}\n"
    return summary
