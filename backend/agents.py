
from backend.schemas import PatientHistory
from backend.llm_client import call_llm
from backend.prompts import INFORMATION_EXTRACTOR_SYSTEM


class AuraPipeline:
    def __init__(self):
        self.patient_history = PatientHistory()

    def run(self, full_transcript: str) -> AuraUIPayload:
        json_object = call_llm(INFORMATION_EXTRACTOR_SYSTEM, full_transcript)
        self.patient_history.update(json_object)
        
        ai_analysis = AuraUIPayload(
            patient_history=self.patient_history,
            ai_analysis=json_object
        )     
        return ai_analysis