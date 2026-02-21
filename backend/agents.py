
from backend.schemas import PatientHistory
from backend.llm_client import call_llm
from backend.prompts import INFORMATION_EXTRACTOR_SYSTEM, QUESTION_GENIE_SYSTEM, QUESTION_GENIE_USER_TEMPLATE
from backend.schemas import AuraUIPayload

class AuraPipeline:
    def __init__(self):
        self.patient_history = PatientHistory()

    def run(self, full_transcript: str) -> AuraUIPayload:
        # first call to extract info from the conversation
        info_json = call_llm(INFORMATION_EXTRACTOR_SYSTEM, full_transcript)
        self.patient_history.update(info_json)
        
        # second call to generate the ai analysis
        prompt = QUESTION_GENIE_SYSTEM + QUESTION_GENIE_USER_TEMPLATE.format(patient_history=self.patient_history)
        diseases_question_json = call_llm(prompt, self.patient_history)
        
        ui_payload = AuraUIPayload(
            patient_history=self.patient_history,
            ai_analysis=diseases_question_json
        )     
        return ui_payload