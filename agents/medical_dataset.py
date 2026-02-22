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
