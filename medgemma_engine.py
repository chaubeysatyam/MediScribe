import json
import re
import time
import traceback
from models import ClinicalEntity, SOAPNote, ClinicalAlert, ICD10Code, ImagingSuggestion, ImageAnalysis, EncounterResult

pipe = None
MODEL_ID = "google/medgemma-4b-it"


def load_medgemma():
    """Load MedGemma using the pipeline() API - confirmed working."""
    global pipe
    from transformers import pipeline as hf_pipeline
    print(f"[MedGemma] Loading {MODEL_ID} via pipeline ...")
    t0 = time.time()
    pipe = hf_pipeline("image-text-to-text", model=MODEL_ID)
    print(f"[MedGemma] Ready in {time.time()-t0:.1f}s")
    return pipe


def _generate(prompt, max_tokens=1024):
    """Generate text using the pipeline."""
    if pipe is None:
        raise RuntimeError("MedGemma not loaded. Run Cell 6 first.")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    result = pipe(text=messages, max_new_tokens=max_tokens)
    return result[0]["generated_text"][-1]["content"].strip()


def _parse_json(text):
    """Try to extract JSON from model output."""
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try code blocks
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try finding a JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # Try finding a JSON array
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def extract_entities(transcript):
    system = ("You are a clinical entity extraction agent. "
              "Extract ONLY entities EXPLICITLY mentioned in the transcript. Never fabricate data. "
              "If vitals not mentioned, return empty {}. Empty fields use '' or []. "
              "Preserve exact anatomical wording. Include laterality, character, onset, frequency if mentioned. "
              "Return ONLY valid JSON with keys: chief_complaint, symptoms, "
              "vitals, medications, allergies, medical_history, family_history, "
              "social_history, duration.")
    result = _generate(system + "\n\nTranscript:\n" + transcript)
    print(f"[Entities] Raw: {result[:200]}")
    data = _parse_json(result)
    # Sanitize types to match Pydantic model
    clean = {}
    list_fields = {"symptoms", "medications", "allergies", "medical_history", "family_history", "social_history"}
    for k, v in data.items():
        if k not in ClinicalEntity.model_fields:
            continue
        if v is None:
            # Default: empty dict for vitals, empty string for strings, empty list for lists
            if k == "vitals":
                clean[k] = {}
            elif k in list_fields:
                clean[k] = []
            else:
                clean[k] = ""
        elif k == "vitals":
            clean[k] = v if isinstance(v, dict) else {}
        elif k in ("chief_complaint", "duration"):
            clean[k] = str(v)
        elif k in list_fields:
            clean[k] = [str(x) for x in v if x is not None] if isinstance(v, list) else []
        else:
            clean[k] = v
    return ClinicalEntity(**clean)


def generate_soap(transcript, entities):
    system = ("You are a board-certified medical documentation specialist. "
              "Generate a SOAP note as valid JSON with keys: subjective, objective, assessment, plan. "
              "All values MUST be strings, not dicts or arrays. "
              "Rules: "
              "- Objective: Only include vitals/findings EXPLICITLY in the transcript. If none, write 'No vitals provided. Recommend: BP, HR, RR, Temp, SpO2.' Never fabricate data. "
              "- Assessment: Structured differential for EACH complaint (e.g. 'Chest pain: ACS vs PE vs musculoskeletal'). No vague phrases. "
              "- Plan: Include labs (troponin, CBC, BMP etc.), management (medications, lifestyle), red flag criteria, and follow-up timeline. "
              "- Never invent data not in the transcript. Preserve exact anatomical descriptions.")
    ctx = system + "\n\nTranscript:\n" + transcript
    ctx += "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2)
    result = _generate(ctx, max_tokens=1500)
    print(f"[SOAP] Raw output ({len(result)} chars): {result[:300]}")
    data = _parse_json(result)
    print(f"[SOAP] Parsed keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    # Sanitize: MedGemma sometimes returns fields as dicts/lists instead of strings
    clean = {}
    for k in ("subjective", "objective", "assessment", "plan"):
        v = data.get(k, "")
        clean[k] = _flatten_soap_value(v)
        if not clean[k]:
            print(f"[SOAP] WARNING: '{k}' is empty after flattening, raw value was: {repr(v)[:100]}")
    return SOAPNote(**clean)


def _flatten_soap_value(v, depth=0):
    """Recursively convert dicts/lists into clean readable text."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for i, item in enumerate(v, 1):
            flat = _flatten_soap_value(item, depth + 1)
            if flat:
                parts.append(f"{i}. {flat}")
        return "\n".join(parts)
    if isinstance(v, dict):
        parts = []
        for dk, dv in v.items():
            # Convert snake_case keys to Title Case
            label = dk.replace("_", " ").title()
            flat = _flatten_soap_value(dv, depth + 1)
            if flat:
                parts.append(f"{label}: {flat}")
        return "\n".join(parts)
    return str(v)


def check_drug_interactions(medications):
    if len(medications) < 2:
        return []
    system = ("You are a pharmacology expert. Check for drug-drug interactions. "
              "Return ONLY a valid JSON array of objects with keys: "
              "alert_type, severity, title, description, recommendation. "
              "If none found return []")
    result = _generate(system + "\n\nMedications: " + ", ".join(medications))
    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def detect_red_flags(entities):
    system = ("You are an emergency triage specialist. Check for RED FLAG symptoms. "
              "Pay special attention to: "
              "- Headache red flags: thunderclap onset, worst headache of life, fever with neck stiffness, "
              "neurological deficits, new onset after age 50, progressive worsening. "
              "- Joint/extremity red flags: signs of systemic inflammatory conditions if joint swelling is present, "
              "potential septic arthritis, compartment syndrome, DVT. "
              "- General red flags: unexplained weight loss, night sweats, progressive symptoms. "
              "Return ONLY a valid JSON array of objects with keys: "
              "alert_type, severity (critical/high/medium/low), title, description, recommendation. "
              "If none return []")
    result = _generate(system + "\n\n" + json.dumps(entities.model_dump(), indent=2))
    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def suggest_icd10(assessment, entities_json=""):
    if not assessment:
        return []
    system = ("You are an expert ICD-10-CM medical coder. Your task is to assign accurate billing codes. "
              "You MUST suggest 3-5 ICD-10-CM codes that match the symptoms and diagnoses described. "
              "Use the most specific code available. Common examples: "
              "R51.9 (headache, unspecified), R51.0 (headache with orthostatic component), "
              "M79.641 (pain in right hand), M79.642 (pain in left hand), "
              "M79.645 (pain in left finger(s)), M25.462 (joint effusion, left knee), "
              "M79.89 (other specified soft tissue disorders). "
              "Each code MUST directly correspond to a symptom or diagnosis in the text. "
              "Do NOT suggest codes for body parts or conditions not mentioned. "
              "Return ONLY a valid JSON array of objects with keys: code, description, confidence (0.0-1.0). "
              "You MUST return at least 2 codes. Never return an empty array if symptoms are present.")
    ctx = system + "\n\nAssessment:\n" + assessment
    if entities_json:
        ctx += "\n\nExtracted Entities:\n" + entities_json
    result = _generate(ctx)
    print(f"[ICD-10] Raw: {result[:200]}")
    data = _parse_json(result)
    if isinstance(data, list):
        return [ICD10Code(**x) for x in data if isinstance(x, dict) and "code" in x]
    return []


def suggest_imaging(entities, assessment):
    """Suggest imaging studies (CT, MRI, X-ray, Ultrasound, etc.)"""
    if not assessment:
        return []
    system = ("You are a radiology consultant. Based on the clinical findings, "
              "suggest appropriate imaging studies. "
              "Return ONLY a valid JSON array of objects with keys: "
              "modality (CT/MRI/X-ray/Ultrasound/PET/Nuclear), "
              "body_region (Brain/Chest/Abdomen/Spine/etc), "
              "indication (why this scan is needed), "
              "urgency (stat/urgent/routine), "
              "contrast (with contrast/without contrast/with and without contrast/N/A), "
              "notes (any special instructions). "
              "Suggest 1-4 most relevant scans. If none needed return []")
    ctx = system + "\n\nAssessment:\n" + assessment
    ctx += "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2)
    result = _generate(ctx)
    print(f"[Imaging] Raw: {result[:200]}")
    data = _parse_json(result)
    if isinstance(data, list):
        out = []
        for x in data:
            if not isinstance(x, dict):
                continue
            # Sanitize: convert any non-string values to strings
            clean = {}
            for k in ("modality", "body_region", "indication", "urgency", "contrast", "notes"):
                v = x.get(k, "")
                clean[k] = str(v) if v is not None else ""
            out.append(ImagingSuggestion(**clean))
        return out
    return []


def analyze_medical_image(image, filename="uploaded_image"):
    """Analyze a medical image (X-ray, CT, MRI, etc.) using MedGemma vision."""
    if pipe is None:
        raise RuntimeError("MedGemma not loaded.")

    prompt = (
        "You are an expert radiologist. Analyze this medical image thoroughly. "
        "Return ONLY valid JSON with keys: "
        "image_type (X-ray/CT/MRI/Ultrasound/PET/Other), "
        "body_part (Chest/Brain/Abdomen/Spine/Extremity/etc), "
        "findings (detailed description of what you see), "
        "impression (overall clinical impression), "
        "abnormalities (JSON array of strings listing each abnormality), "
        "recommendations (follow-up recommendations). "
        "Be thorough and clinically precise."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    }]

    print(f"[ImageAnalysis] Analyzing {filename} ...")
    t0 = time.time()
    result = pipe(text=messages, max_new_tokens=1024)
    raw = result[0]["generated_text"][-1]["content"].strip()
    print(f"[ImageAnalysis] Done in {time.time()-t0:.1f}s, raw: {raw[:200]}")

    data = _parse_json(raw)

    # Sanitize
    clean = {}
    for k in ("image_type", "body_part", "findings", "impression", "recommendations"):
        v = data.get(k, "")
        clean[k] = str(v) if v is not None else ""
    abnormalities = data.get("abnormalities", [])
    if isinstance(abnormalities, list):
        clean["abnormalities"] = [str(x) for x in abnormalities if x is not None]
    else:
        clean["abnormalities"] = []
    clean["filename"] = filename

    return ImageAnalysis(**clean)

def run_clinical_pipeline(transcript, patient_age=None, patient_sex=None):
    t0 = time.time()
    encounter = EncounterResult(transcript=transcript)
    errors = []

    print(f"[Pipeline] pipe is {'LOADED' if pipe is not None else 'NONE'}")

    if pipe is None:
        encounter.soap_note = SOAPNote(
            subjective=f"Patient reports: {transcript.strip()}",
            objective="MedGemma not loaded.",
            assessment="Run Cell 6 to load MedGemma.",
            plan="Load AI model then retry.",
        )
        encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
        return encounter

    # Step 1: Entities
    print("[Pipeline] 1/6 Extracting entities ...")
    try:
        encounter.entities = extract_entities(transcript)
        print(f"[Pipeline] 1/6 OK - {len(encounter.entities.symptoms)} symptoms")
    except Exception as e:
        errors.append(f"Entities: {e}")
        print(f"[Pipeline] 1/6 FAILED: {e}")
        traceback.print_exc()

    # Step 2: SOAP
    print("[Pipeline] 2/6 Generating SOAP ...")
    try:
        encounter.soap_note = generate_soap(transcript, encounter.entities)
        print(f"[Pipeline] 2/6 OK")
    except Exception as e:
        errors.append(f"SOAP: {e}")
        print(f"[Pipeline] 2/6 FAILED: {e}")
        traceback.print_exc()

    # Step 3: Drug interactions
    print("[Pipeline] 3/6 Drug interactions ...")
    try:
        drug_alerts = check_drug_interactions(encounter.entities.medications)
        encounter.clinical_alerts.extend(drug_alerts)
        print(f"[Pipeline] 3/6 OK - {len(drug_alerts)} alerts")
    except Exception as e:
        errors.append(f"Drugs: {e}")
        print(f"[Pipeline] 3/6 FAILED: {e}")

    # Step 4: Red flags
    print("[Pipeline] 4/6 Red flags ...")
    try:
        red_flags = detect_red_flags(encounter.entities)
        encounter.clinical_alerts.extend(red_flags)
        print(f"[Pipeline] 4/6 OK - {len(red_flags)} alerts")
    except Exception as e:
        errors.append(f"RedFlags: {e}")
        print(f"[Pipeline] 4/6 FAILED: {e}")

    # Step 5: ICD-10
    print("[Pipeline] 5/6 ICD-10 codes ...")
    try:
        entities_json = json.dumps(encounter.entities.model_dump(), indent=2)
        encounter.icd10_codes = suggest_icd10(encounter.soap_note.assessment, entities_json)
        print(f"[Pipeline] 5/6 OK - {len(encounter.icd10_codes)} codes")
    except Exception as e:
        errors.append(f"ICD10: {e}")
        print(f"[Pipeline] 5/6 FAILED: {e}")

    # Step 6: Imaging suggestions
    print("[Pipeline] 6/6 Imaging suggestions ...")
    try:
        encounter.imaging_suggestions = suggest_imaging(encounter.entities, encounter.soap_note.assessment)
        print(f"[Pipeline] 6/6 OK - {len(encounter.imaging_suggestions)} scans")
    except Exception as e:
        errors.append(f"Imaging: {e}")
        print(f"[Pipeline] 6/6 FAILED: {e}")

    # Safety net
    if not encounter.soap_note.subjective:
        encounter.soap_note.subjective = f"Patient reports: {transcript.strip()}"
    if not encounter.soap_note.assessment:
        encounter.soap_note.assessment = "See transcript above."

    # NSAID-Aspirin allergy cross-check
    try:
        allergies_lower = [a.lower() for a in encounter.entities.allergies]
        aspirin_allergy = any(x in a for a in allergies_lower for x in ("aspirin", "asa", "nsaid"))
        if aspirin_allergy:
            nsaid_terms = ["nsaid", "ibuprofen", "naproxen", "diclofenac", "indomethacin", "celecoxib", "meloxicam", "ketorolac"]
            # Check BOTH the plan text AND current medications
            plan_lower = encounter.soap_note.plan.lower()
            meds_lower = " ".join(m.lower() for m in encounter.entities.medications)
            combined_text = plan_lower + " " + meds_lower
            mentioned = [t for t in nsaid_terms if t in combined_text]
            if mentioned:
                encounter.clinical_alerts.append(ClinicalAlert(
                    alert_type="drug_interaction", severity="critical",
                    title="⚠️ NSAID-Aspirin Cross-Reactivity Risk",
                    description=f"Patient is allergic to aspirin. Plan mentions {', '.join(mentioned).upper()}, "
                                f"which may cause cross-reactivity in aspirin-allergic patients (risk of bronchospasm, "
                                f"urticaria, anaphylaxis). Up to 30% cross-reactivity rate.",
                    recommendation="Avoid NSAIDs in aspirin-allergic patients. Consider acetaminophen as a safer "
                                   "alternative for pain relief. If NSAIDs are essential, consider COX-2 selective "
                                   "inhibitor with caution under supervised desensitization.",
                ))
                print(f"[Pipeline] SAFETY: NSAID-aspirin cross-reactivity alert added")
    except Exception as e:
        print(f"[Pipeline] Allergy cross-check error: {e}")

    if errors:
        encounter.clinical_alerts.append(ClinicalAlert(
            alert_type="system", severity="warning",
            title=f"{len(errors)} step(s) had errors",
            description="; ".join(errors),
            recommendation="Check runtime logs.",
        ))

    encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
    print(f"[Pipeline] Done in {encounter.processing_time_ms}ms, {len(errors)} errors")
    return encounter
