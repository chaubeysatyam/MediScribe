from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import uuid
class ClinicalEntity(BaseModel):
    chief_complaint: str = ""
    symptoms: List[str] = Field(default_factory=list)
    vitals: Dict[str, str] = Field(default_factory=dict)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: List[str] = Field(default_factory=list)
    duration: str = ""
class SOAPNote(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
class ClinicalAlert(BaseModel):
    alert_type: str = ""
    severity: str = "info"
    title: str = ""
    description: str = ""
    recommendation: str = ""
class ICD10Code(BaseModel):
    code: str = ""
    description: str = ""
    confidence: float = 0.0
class ImagingSuggestion(BaseModel):
    modality: str = ""                                        
    body_region: str = ""                                
    indication: str = ""                             
    urgency: str = "routine"                        
    contrast: str = ""                                  
    notes: str = ""
class ImageAnalysis(BaseModel):
    filename: str = ""
    image_type: str = ""                                             
    body_part: str = ""                                         
    findings: str = ""                                
    impression: str = ""                                         
    abnormalities: List[str] = Field(default_factory=list)
    recommendations: str = ""                                
class EncounterResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    patient_name: str = ""
    transcript: str = ""
    entities: ClinicalEntity = Field(default_factory=ClinicalEntity)
    soap_note: SOAPNote = Field(default_factory=SOAPNote)
    clinical_alerts: List[ClinicalAlert] = Field(default_factory=list)
    icd10_codes: List[ICD10Code] = Field(default_factory=list)
    imaging_suggestions: List[ImagingSuggestion] = Field(default_factory=list)
    image_analyses: List[ImageAnalysis] = Field(default_factory=list)
    processing_time_ms: float = 0.0
class GenerateRequest(BaseModel):
    transcript: str
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
