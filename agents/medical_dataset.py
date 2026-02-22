"""
medical_dataset.py
------------------
Loads a symptom-disease dataset.

Priority order:
1. HuggingFace `datasets` library  (requires network + pip install datasets)
2. Built-in bundled dataset  (always available, no internet needed)

The bundled dataset is derived from the public "Disease-Symptom Knowledge Database"
and augmented with common risk-factor / anamnesis fields.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Built-in dataset
# ---------------------------------------------------------------------------

BUNDLED_DATA = [
    # Each row: disease, symptoms (list), risk_factors (list), base_prevalence
    ("Common Cold", 
     ["runny nose", "sneezing", "sore throat", "cough", "mild fever", "congestion", "fatigue"],
     ["contact with infected person", "winter season", "immunodeficiency"],
     0.30),

    ("Influenza",
     ["high fever", "chills", "muscle aches", "headache", "cough", "fatigue", "sore throat",
      "loss of appetite", "runny nose"],
     ["no flu vaccine", "elderly", "immunodeficiency", "winter season", "contact with infected person"],
     0.10),

    ("COVID-19",
     ["fever", "cough", "shortness of breath", "fatigue", "loss of taste", "loss of smell",
      "headache", "sore throat", "muscle aches", "diarrhea", "chest pain"],
     ["no covid vaccine", "elderly", "obesity", "diabetes", "hypertension", "immunodeficiency",
      "contact with infected person"],
     0.08),

    ("Pneumonia",
     ["high fever", "chills", "cough", "chest pain", "shortness of breath", "fatigue",
      "rapid breathing", "sweating", "nausea"],
     ["elderly", "smoking", "immunodeficiency", "chronic lung disease", "diabetes",
      "recent respiratory infection"],
     0.04),

    ("Bronchitis",
     ["cough", "mucus production", "fatigue", "shortness of breath", "chest discomfort",
      "mild fever", "chills"],
     ["smoking", "air pollution exposure", "repeated respiratory infections"],
     0.08),

    ("Asthma",
     ["shortness of breath", "wheezing", "chest tightness", "cough", "nocturnal symptoms"],
     ["family history of asthma", "allergies", "eczema", "smoking", "air pollution exposure"],
     0.06),

    ("Allergic Rhinitis",
     ["sneezing", "runny nose", "nasal congestion", "itchy eyes", "watery eyes", "cough"],
     ["family history of allergies", "eczema", "asthma", "spring season", "pet exposure"],
     0.15),

    ("Gastroenteritis",
     ["nausea", "vomiting", "diarrhea", "abdominal cramps", "mild fever", "headache",
      "muscle aches", "loss of appetite"],
     ["contaminated food", "contaminated water", "contact with infected person"],
     0.12),

    ("Urinary Tract Infection",
     ["burning urination", "frequent urination", "urgency to urinate", "cloudy urine",
      "pelvic pain", "mild fever", "back pain", "blood in urine"],
     ["female sex", "sexual activity", "diabetes", "urinary catheter", "kidney stones"],
     0.07),

    ("Hypertension",
     ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath",
      "nosebleed", "fatigue"],
     ["obesity", "family history of hypertension", "high salt diet", "sedentary lifestyle",
      "smoking", "diabetes", "elderly"],
     0.20),

    ("Type 2 Diabetes",
     ["frequent urination", "excessive thirst", "unexplained weight loss", "fatigue",
      "blurred vision", "slow healing wounds", "frequent infections"],
     ["obesity", "sedentary lifestyle", "family history of diabetes", "elderly",
      "hypertension", "high sugar diet"],
     0.10),

    ("Migraine",
     ["severe headache", "nausea", "vomiting", "sensitivity to light", "sensitivity to sound",
      "visual aura", "throbbing pain", "dizziness"],
     ["family history of migraine", "female sex", "hormonal changes", "stress",
      "sleep deprivation", "alcohol consumption"],
     0.12),

    ("Anxiety Disorder",
     ["excessive worry", "restlessness", "fatigue", "difficulty concentrating", "irritability",
      "muscle tension", "sleep disturbance", "palpitations", "shortness of breath"],
     ["family history of anxiety", "stress", "trauma", "substance abuse", "chronic illness"],
     0.15),

    ("Depression",
     ["persistent sadness", "loss of interest", "fatigue", "sleep disturbance",
      "appetite changes", "difficulty concentrating", "feelings of worthlessness",
      "psychomotor changes"],
     ["family history of depression", "trauma", "chronic illness", "stress",
      "substance abuse", "social isolation"],
     0.12),

    ("Appendicitis",
     ["right lower abdominal pain", "nausea", "vomiting", "fever", "loss of appetite",
      "rebound tenderness", "abdominal rigidity"],
     ["young age", "male sex", "family history of appendicitis"],
     0.02),

    ("Gastroesophageal Reflux Disease",
     ["heartburn", "acid regurgitation", "chest pain", "dysphagia", "chronic cough",
      "hoarseness", "nausea"],
     ["obesity", "smoking", "alcohol consumption", "pregnancy", "hiatal hernia",
      "high fat diet"],
     0.15),

    ("Hypothyroidism",
     ["fatigue", "weight gain", "cold intolerance", "constipation", "dry skin",
      "hair loss", "slow heart rate", "depression", "muscle weakness"],
     ["female sex", "family history of thyroid disease", "autoimmune disease", "elderly"],
     0.05),

    ("Anemia",
     ["fatigue", "weakness", "pale skin", "shortness of breath", "dizziness",
      "rapid heart rate", "cold hands and feet", "headache"],
     ["female sex", "poor diet", "chronic disease", "pregnancy", "vegetarian diet",
      "blood loss"],
     0.08),

    ("Coronary Artery Disease",
     ["chest pain", "shortness of breath", "fatigue", "palpitations", "chest tightness",
      "radiating arm pain", "sweating", "nausea"],
     ["smoking", "diabetes", "hypertension", "high cholesterol", "obesity",
      "family history of heart disease", "sedentary lifestyle", "elderly", "male sex"],
     0.06),

    ("Pulmonary Embolism",
     ["sudden shortness of breath", "chest pain", "rapid heart rate", "cough",
      "blood in sputum", "leg swelling", "dizziness", "fainting"],
     ["deep vein thrombosis", "prolonged immobility", "surgery", "cancer",
      "oral contraceptives", "pregnancy", "obesity"],
     0.01),
     ("Hyperthyroidism",
     ["unintentional weight loss", "rapid heartbeat", "increased appetite", "nervousness", "anxiety", "tremor", "sweating", "heat intolerance", "fatigue"],
     ["female sex", "family history of thyroid disease", "excessive iodine intake", "Graves' disease"],
     0.01),

    ("Peptic Ulcer Disease",
     ["burning stomach pain", "bloating", "heartburn", "nausea", "intolerance to fatty foods", "dark or bloody stools", "unexplained weight loss"],
     ["H. pylori infection", "long-term NSAID use", "smoking", "excessive alcohol consumption", "stress"],
     0.05),

    ("Osteoarthritis",
     ["joint pain", "stiffness", "tenderness", "loss of flexibility", "grating sensation", "bone spurs", "swelling"],
     ["elderly", "obesity", "joint injuries", "repeated stress on the joint", "genetics", "female sex"],
     0.12),

    ("Rheumatoid Arthritis",
     ["tender joints", "swollen joints", "joint stiffness", "fatigue", "fever", "loss of appetite"],
     ["female sex", "family history of RA", "smoking", "obesity", "environmental exposures"],
     0.01),

    ("Chronic Obstructive Pulmonary Disease",
     ["shortness of breath", "wheezing", "chest tightness", "chronic cough", "cyanosis", "frequent respiratory infections", "lack of energy"],
     ["smoking", "long-term exposure to chemical fumes", "air pollution exposure", "genetics"],
     0.06),

    ("Chronic Kidney Disease",
     ["nausea", "vomiting", "loss of appetite", "fatigue", "sleep problems", "changes in urination", "decreased mental sharpness", "muscle cramps", "swelling of feet"],
     ["diabetes", "hypertension", "heart disease", "smoking", "obesity", "family history of kidney disease", "elderly"],
     0.14),

    ("Gout",
     ["intense joint pain", "lingering discomfort", "inflammation", "redness", "limited range of motion"],
     ["male sex", "obesity", "high purine diet", "alcohol consumption", "hypertension", "diabetes", "kidney disease"],
     0.04),

    ("Irritable Bowel Syndrome",
     ["abdominal pain", "bloating", "gas", "diarrhea", "constipation", "mucus in stool"],
     ["female sex", "young age", "family history of IBS", "anxiety", "depression", "food sensitivities"],
     0.11),

    ("Celiac Disease",
     ["diarrhea", "fatigue", "weight loss", "bloating", "gas", "abdominal pain", "nausea", "constipation"],
     ["family history of celiac disease", "type 1 diabetes", "down syndrome", "autoimmune thyroid disease"],
     0.01),

    ("Eczema",
     ["dry skin", "itching", "red to brownish-gray patches", "small raised bumps", "thickened cracked skin", "raw sensitive skin"],
     ["family history of eczema", "allergies", "asthma", "environmental irritants"],
     0.10),

    ("Psoriasis",
     ["red patches of skin", "silvery scales", "dry cracked skin", "itching", "burning", "soreness", "thickened pitted nails", "swollen joints"],
     ["family history of psoriasis", "stress", "smoking", "obesity", "heavy alcohol consumption"],
     0.03),

    ("Multiple Sclerosis",
     ["numbness", "tingling", "electric-shock sensations", "tremor", "lack of coordination", "unsteady gait", "vision loss", "fatigue", "slurred speech"],
     ["female sex", "young adulthood", "family history of MS", "vitamin D deficiency", "smoking"],
     0.001),

    ("Parkinson's Disease",
     ["tremor", "slowed movement", "rigid muscles", "impaired posture", "loss of automatic movements", "speech changes", "writing changes"],
     ["elderly", "male sex", "family history of Parkinson's", "exposure to toxins"],
     0.01),

    ("Alzheimer's Disease",
     ["memory loss", "confusion with time or place", "difficulty completing familiar tasks", "misplacing things", "poor judgment", "withdrawal from social activities"],
     ["elderly", "family history of Alzheimer's", "genetics", "head trauma", "sedentary lifestyle", "hypertension"],
     0.05),

    ("Atrial Fibrillation",
     ["palpitations", "weakness", "reduced ability to exercise", "fatigue", "lightheadedness", "dizziness", "shortness of breath", "chest pain"],
     ["elderly", "hypertension", "obesity", "heart disease", "alcohol consumption", "sleep apnea", "family history"],
     0.02),

    ("Sleep Apnea",
     ["loud snoring", "episodes in which you stop breathing", "gasping for air", "morning headache", "insomnia", "excessive daytime sleepiness", "irritability"],
     ["obesity", "male sex", "elderly", "smoking", "nasal congestion", "family history"],
     0.09),

    ("Fibromyalgia",
     ["widespread pain", "fatigue", "cognitive difficulties", "sleep disturbances", "headache", "depression", "anxiety"],
     ["female sex", "family history of fibromyalgia", "osteoarthritis", "rheumatoid arthritis", "lupus"],
     0.04),

    ("Lyme Disease",
     ["fever", "chills", "headache", "fatigue", "muscle and joint aches", "swollen lymph nodes", "erythema migrans rash"],
     ["outdoor activities in wooded areas", "exposure to deer ticks", "summer season"],
     0.005),

    ("Polycystic Ovary Syndrome",
     ["irregular periods", "excess androgen", "polycystic ovaries", "weight gain", "thinning hair", "acne"],
     ["obesity", "family history of PCOS", "insulin resistance"],
     0.08),

    ("Pleurisy",
     ["sharp chest pain", "shortness of breath", "cough", "fever"],
     ["pneumonia", "viral infection", "pulmonary embolism", "rib fracture", "lung cancer"],
     0.01),
     ("Bacterial Meningitis",
     ["high fever", "stiff neck", "severe headache", "nausea", "vomiting", "confusion", "sensitivity to light", "seizures", "skin rash"],
     ["young age", "community living", "skipping vaccinations", "immunodeficiency"],
     0.001),

    ("Acute Myocardial Infarction",
     ["crushing chest pain", "shortness of breath", "nausea", "sweating", "pain radiating to jaw", "pain radiating to left arm", "anxiety", "lightheadedness"],
     ["hypertension", "hyperlipidemia", "smoking", "diabetes", "obesity", "family history of heart disease", "male sex"],
     0.03),

    ("Diverticulitis",
     ["left lower abdominal pain", "fever", "nausea", "vomiting", "constipation", "abdominal tenderness", "bloating"],
     ["elderly", "low fiber diet", "obesity", "smoking", "sedentary lifestyle"],
     0.04),

    ("Cholecystitis",
     ["right upper abdominal pain", "pain radiating to right shoulder", "nausea", "vomiting", "fever", "tenderness when touched", "bloating"],
     ["female sex", "pregnancy", "obesity", "rapid weight loss", "elderly", "gallstones"],
     0.05),

    ("Crohn's Disease",
     ["diarrhea", "abdominal cramping", "blood in stool", "fatigue", "unintentional weight loss", "mouth sores", "fistulas"],
     ["family history of IBD", "smoking", "young age", "high fat diet"],
     0.005),

    ("Pulmonary Tuberculosis",
     ["persistent cough", "coughing up blood", "chest pain", "unintentional weight loss", "fatigue", "fever", "night sweats", "chills"],
     ["immunodeficiency", "travel to high-risk areas", "substance abuse", "healthcare work", "homelessness"],
     0.002),

    ("Deep Vein Thrombosis",
     ["leg pain", "leg swelling", "redness of the skin", "warmth in the affected leg"],
     ["prolonged immobility", "surgery", "oral contraceptives", "smoking", "cancer", "pregnancy", "obesity"],
     0.02),

    ("Pancreatitis",
     ["upper abdominal pain radiating to back", "abdominal tenderness", "fever", "rapid pulse", "nausea", "vomiting"],
     ["alcohol consumption", "gallstones", "high triglycerides", "obesity", "abdominal surgery"],
     0.01),

    ("Hypoglycemia",
     ["shakiness", "sweating", "dizziness", "fast heartbeat", "hunger", "confusion", "irritability", "blurred vision"],
     ["insulin use", "skipping meals", "excessive exercise", "alcohol consumption", "liver disease"],
     0.06),

    ("Hyperglycemic Hyperosmolar State",
     ["extreme thirst", "frequent urination", "dry mouth", "fever", "blurred vision", "confusion", "hallucinations", "weakness"],
     ["type 2 diabetes", "infection", "non-adherence to diabetes medication", "diuretics", "elderly"],
     0.005),

    ("Systemic Lupus Erythematosus",
     ["butterfly-shaped rash", "joint pain", "fatigue", "fever", "fingers turning blue in cold", "shortness of breath", "chest pain", "dry eyes"],
     ["female sex", "age 15-45", "family history of autoimmune disease", "sun exposure"],
     0.01),

    ("Multiple Myeloma",
     ["bone pain", "frequent infections", "weakness in legs", "excessive thirst", "constipation", "confusion", "weight loss"],
     ["elderly", "male sex", "family history of myeloma", "obesity", "exposure to radiation"],
     0.007),

    ("Sepsis",
     ["high heart rate", "low blood pressure", "confusion", "shivering", "fever", "extreme pain", "shortness of breath", "clammy skin"],
     ["recent surgery", "recent infection", "immunodeficiency", "elderly", "infant age", "chronic illness"],
     0.02),

    ("Congestive Heart Failure",
     ["shortness of breath while lying down", "fatigue", "swelling in legs", "rapid heartbeat", "persistent cough", "abdominal swelling", "sudden weight gain"],
     ["hypertension", "coronary artery disease", "diabetes", "obesity", "smoking", "sleep apnea"],
     0.05),

    ("Cushing's Syndrome",
     ["weight gain in midsection", "buffalo hump", "moon face", "purple stretch marks", "thin skin", "slow healing", "acne", "muscle weakness"],
     ["long-term corticosteroid use", "pituitary tumor", "adrenal tumor"],
     0.001),

    ("Endocarditis",
     ["fever", "chills", "new heart murmur", "fatigue", "aching joints", "night sweats", "shortness of breath", "small red spots on skin"],
     ["artificial heart valve", "damaged heart valves", "intravenous drug use", "dental procedures", "indwelling catheters"],
     0.002),

    ("Glaucoma",
     ["gradual loss of peripheral vision", "tunnel vision", "severe eye pain", "nausea", "blurred vision", "halos around lights", "eye redness"],
     ["elderly", "family history of glaucoma", "diabetes", "hypertension", "high intraocular pressure"],
     0.04),

    ("Chronic Venous Insufficiency",
     ["leg heaviness", "varicose veins", "skin color changes near ankles", "leg ulcers", "swelling", "itching"],
     ["obesity", "pregnancy", "female sex", "prolonged standing", "family history", "history of DVT"],
     0.10),

    ("Meniere's Disease",
     ["vertigo", "tinnitus", "hearing loss", "feeling of fullness in the ear", "nausea", "vomiting"],
     ["age 40-60", "family history", "autoimmune disorders", "allergies", "viral infections"],
     0.003),

    ("Sarcoidosis",
     ["persistent dry cough", "shortness of breath", "fatigue", "swollen lymph nodes", "tender red bumps on shins", "blurred vision"],
     ["young adulthood", "family history", "occupational exposure to dust/mold"],
     0.002),
]


def load_bundled_dataset() -> pd.DataFrame:
    rows = []
    for disease, symptoms, risk_factors, prevalence in BUNDLED_DATA:
        rows.append({
            "disease": disease,
            "symptoms": symptoms,
            "risk_factors": risk_factors,
            "base_prevalence": prevalence,
        })
    return pd.DataFrame(rows)


def try_load_huggingface_dataset() -> Optional[pd.DataFrame]:
    """
    Attempts to load the 'QuyenAnhDE/Diseases_Symptoms' dataset from HuggingFace.
    Returns None if the library or network is unavailable.
    """
    try:
        from datasets import Dataset  # noqa: F401
        from huggingface_hub import hf_hub_download, list_repo_files

        repo = os.getenv("REPO", "QuyenAnhDE/Diseases_Symptoms")
        files = list_repo_files(repo, repo_type="dataset")
        csv_name = [f for f in files if f.endswith(".csv")][0]

        csv_path = hf_hub_download(repo_id=repo, filename=csv_name, repo_type="dataset")
        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)
        df = df.rename(columns={"Name": "disease", "Symptoms": "symptoms_raw"})
        df["symptoms"] = df["symptoms_raw"].apply(
            lambda x: [s.strip().lower() for s in str(x).split(",") if s.strip()])
        df["risk_factors"] = [[] for _ in range(len(df))]
        df["base_prevalence"] = 0.05  # unknown, use uniform prior
        return df[["disease", "symptoms", "risk_factors", "base_prevalence"]]
    except Exception as e:
        print(f"Warning: Could not load HuggingFace dataset. Exception: {e}")
        print("Falling back to bundled dataset...")
        return None


def load_datasets() -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_name)."""
    hf = try_load_huggingface_dataset()
    if hf is not None:
        return hf, "HuggingFace (QuyenAnhDE/Diseases_Symptoms)"
    return load_bundled_dataset(), "Built-in bundled dataset"
