import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import ClinicalEntity, SOAPNote, ClinicalAlert, ICD10Code, ImagingSuggestion, ImageAnalysis, EncounterResult
from config import ICD10_CONFIDENCE_THRESHOLD, MEDGEMMA_MODEL

pipe = None
MODEL_ID = MEDGEMMA_MODEL


def load_medgemma():
    global pipe
    from transformers import pipeline as hf_pipeline
    print(f"[MedGemma] Loading {MODEL_ID} via pipeline ...")
    t0 = time.time()
    pipe = hf_pipeline("image-text-to-text", model=MODEL_ID)
    print(f"[MedGemma] Ready in {time.time()-t0:.1f}s")
    return pipe


def _build_patient_ctx(patient_age=None, patient_sex=None):
    parts = []
    if patient_age:
        parts.append("Patient age: " + str(patient_age) + " years old.")
    if patient_sex:
        parts.append("Patient sex: " + str(patient_sex) + ".")
    return " ".join(parts) + " " if parts else ""


def _generate(prompt, max_tokens=1024):
    if pipe is None:
        raise RuntimeError("MedGemma not loaded.")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    result = pipe(text=messages, max_new_tokens=max_tokens)
    return result[0]["generated_text"][-1]["content"].strip()


def _parse_json(text):
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
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
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def extract_entities(transcript, patient_age=None, patient_sex=None):
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are a clinical entity extraction agent. "
        + "Extract ONLY entities EXPLICITLY mentioned in the transcript. Never fabricate data. "
        + "If vitals not mentioned, return empty {}. Empty fields use '' or []. "
        + "Preserve exact anatomical wording. Include laterality, character, onset, frequency if mentioned. "
        + "Return ONLY valid JSON with keys: chief_complaint, symptoms, "
        + "vitals, medications, allergies, medical_history, family_history, "
        + "social_history, duration."
    )
    result = _generate(system + "\n\nTranscript:\n" + transcript, max_tokens=512)
    print("[Entities] Raw: " + result[:200])
    data = _parse_json(result)
    clean = {}
    list_fields = {"symptoms", "medications", "allergies", "medical_history", "family_history", "social_history"}
    for k, v in data.items():
        if k not in ClinicalEntity.model_fields:
            continue
        if v is None:
            clean[k] = {} if k == "vitals" else ([] if k in list_fields else "")
        elif k == "vitals":
            clean[k] = v if isinstance(v, dict) else {}
        elif k in ("chief_complaint", "duration"):
            clean[k] = str(v)
        elif k in list_fields:
            clean[k] = [str(x) for x in v if x is not None] if isinstance(v, list) else []
        else:
            clean[k] = v
    return ClinicalEntity(**clean)


def generate_soap(transcript, entities, patient_age=None, patient_sex=None):
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are a board-certified medical documentation specialist. "
        + "Generate a SOAP note as valid JSON with keys: subjective, objective, assessment, plan. "
        + "All values MUST be strings, not dicts or arrays. "
        + "Rules: "
        + "- Objective: Only include vitals/findings EXPLICITLY in the transcript. If none, write 'No vitals provided. Recommend: BP, HR, RR, Temp, SpO2.' Never fabricate data. "
        + "- Assessment: Structured differential for EACH complaint (e.g. 'Chest pain: ACS vs PE vs musculoskeletal'). Consider patient age and sex. No vague phrases. "
        + "- Plan: Include labs (troponin, CBC, BMP etc.), management (medications, lifestyle), red flag criteria, and follow-up timeline. "
        + "- Never invent data not in the transcript. Preserve exact anatomical descriptions."
    )
    context = system + "\n\nTranscript:\n" + transcript
    context += "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2)
    result = _generate(context, max_tokens=1024)
    print("[SOAP] Raw output (" + str(len(result)) + " chars): " + result[:300])
    data = _parse_json(result)
    print("[SOAP] Parsed keys: " + str(list(data.keys()) if isinstance(data, dict) else type(data)))
    clean = {}
    for k in ("subjective", "objective", "assessment", "plan"):
        v = data.get(k, "")
        clean[k] = _flatten_soap_value(v)
        if not clean[k]:
            print("[SOAP] WARNING: '" + k + "' empty after flattening, raw: " + repr(v)[:100])
    return SOAPNote(**clean)


def _flatten_soap_value(v, depth=0):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for i, item in enumerate(v, 1):
            flat = _flatten_soap_value(item, depth + 1)
            if flat:
                parts.append(str(i) + ". " + flat)
        return "\n".join(parts)
    if isinstance(v, dict):
        parts = []
        for dk, dv in v.items():
            label = dk.replace("_", " ").title()
            flat = _flatten_soap_value(dv, depth + 1)
            if flat:
                parts.append(label + ": " + flat)
        return "\n".join(parts)
    return str(v)


def check_drug_interactions(medications, patient_age=None, patient_sex=None):
    if len(medications) < 2:
        return []
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are a pharmacology expert. Check for drug-drug interactions. "
        + "Return ONLY a valid JSON array of objects with keys: "
        + "alert_type, severity, title, description, recommendation. "
        + "If none found return []"
    )
    result = _generate(system + "\n\nMedications: " + ", ".join(medications), max_tokens=300)
    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def detect_red_flags(entities, patient_age=None, patient_sex=None):
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are an emergency triage specialist reviewing clinical entities. "
        + "Your task is to identify RED FLAG symptoms that are EXPLICITLY present in the provided entities. "
        + "STRICT RULES - you MUST follow all of these: "
        + "1. ONLY flag symptoms that are DIRECTLY stated in the entities JSON below. "
        + "2. NEVER infer, assume, or extrapolate symptoms not mentioned. "
        + "3. NEVER flag thunderclap onset unless the word 'sudden' or 'thunderclap' or 'worst ever' appears explicitly. "
        + "4. NEVER flag fever or neck stiffness unless explicitly mentioned. "
        + "5. NEVER flag neurological deficits unless explicitly mentioned. "
        + "6. NEVER flag new onset after age 50 unless the patient age is explicitly over 50. "
        + "7. If a symptom is absent from the entities, do NOT create an alert for it. "
        + "8. Return ONLY alerts for symptoms that are clearly documented in the entities below. "
        + "Return ONLY a valid JSON array of objects with keys: "
        + "alert_type (set to 'red_flag'), severity (critical/high/medium/low), title, description, recommendation. "
        + "If no red flags are present in the entities, return []."
    )
    result = _generate(system + "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2), max_tokens=350)

    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def suggest_icd10(assessment, entities_json="", patient_age=None, patient_sex=None):
    if not assessment:
        return []
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are an expert ICD-10-CM medical coder. Your task is to assign accurate billing codes. "
        + "You MUST suggest 3-5 ICD-10-CM codes that match the symptoms and diagnoses described. "
        + "Use the most specific code available. Common examples: "
        + "R51.9 (headache, unspecified), R51.0 (headache with orthostatic component), "
        + "M79.641 (pain in right hand), M79.642 (pain in left hand), "
        + "M79.645 (pain in left finger(s)), M25.462 (joint effusion, left knee), "
        + "M79.89 (other specified soft tissue disorders). "
        + "Each code MUST directly correspond to a symptom or diagnosis in the text. "
        + "Do NOT suggest codes for body parts or conditions not mentioned. "
        + "Return ONLY a valid JSON array of objects with keys: code, description, confidence (0.0-1.0). "
        + "You MUST return at least 2 codes. Never return an empty array if symptoms are present."
    )
    context = system + "\n\nAssessment:\n" + assessment
    if entities_json:
        context += "\n\nExtracted Entities:\n" + entities_json
    result = _generate(context, max_tokens=300)
    print("[ICD-10] Raw: " + result[:200])
    data = _parse_json(result)
    if isinstance(data, list):
        codes = [ICD10Code(**x) for x in data if isinstance(x, dict) and "code" in x]
        return [c for c in codes if c.confidence >= ICD10_CONFIDENCE_THRESHOLD]
    return []


def suggest_imaging(entities, assessment, patient_age=None, patient_sex=None):
    if not assessment:
        return []
    ctx = _build_patient_ctx(patient_age, patient_sex)
    system = (
        ctx
        + "You are a radiology consultant. Based on the clinical findings, "
        + "suggest appropriate imaging studies. "
        + "Return ONLY a valid JSON array of objects with keys: "
        + "modality (CT/MRI/X-ray/Ultrasound/PET/Nuclear), "
        + "body_region (Brain/Chest/Abdomen/Spine/etc), "
        + "indication (why this scan is needed), "
        + "urgency (stat/urgent/routine), "
        + "contrast (with contrast/without contrast/with and without contrast/N/A), "
        + "notes (any special instructions). "
        + "Suggest 1-4 most relevant scans. If none needed return []"
    )
    context = system + "\n\nAssessment:\n" + assessment
    context += "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2)
    result = _generate(context, max_tokens=300)
    print("[Imaging] Raw: " + result[:200])
    data = _parse_json(result)
    if isinstance(data, list):
        out = []
        for x in data:
            if not isinstance(x, dict):
                continue
            clean = {}
            for k in ("modality", "body_region", "indication", "urgency", "contrast", "notes"):
                v = x.get(k, "")
                clean[k] = str(v) if v is not None else ""
            out.append(ImagingSuggestion(**clean))
        return out
    return []


def analyze_medical_image(image, filename="uploaded_image"):
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
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    print("[ImageAnalysis] Analyzing " + filename + " ...")
    t0 = time.time()
    result = pipe(text=messages, max_new_tokens=600)
    raw = result[0]["generated_text"][-1]["content"].strip()
    print("[ImageAnalysis] Done in " + str(round(time.time()-t0, 1)) + "s, raw: " + raw[:200])
    data = _parse_json(raw)
    clean = {}
    for k in ("image_type", "body_part", "findings", "impression", "recommendations"):
        v = data.get(k, "")
        clean[k] = str(v) if v is not None else ""
    abnormalities = data.get("abnormalities", [])
    clean["abnormalities"] = [str(x) for x in abnormalities if x is not None] if isinstance(abnormalities, list) else []
    clean["filename"] = filename
    return ImageAnalysis(**clean)


def _run_allergy_checks(encounter):
    alerts = []
    try:
        allergies_lower = [a.lower() for a in encounter.entities.allergies]
        plan_lower = encounter.soap_note.plan.lower()
        meds_lower = " ".join(m.lower() for m in encounter.entities.medications)
        combined = plan_lower + " " + meds_lower

        aspirin_allergy = any(x in a for a in allergies_lower for x in ("aspirin", "asa", "nsaid"))
        if aspirin_allergy:
            nsaid_terms = ["nsaid", "ibuprofen", "naproxen", "diclofenac", "indomethacin", "celecoxib", "meloxicam", "ketorolac"]
            mentioned = [t for t in nsaid_terms if t in combined]
            if mentioned:
                alerts.append(ClinicalAlert(
                    alert_type="drug_interaction",
                    severity="critical",
                    title="NSAID-Aspirin Cross-Reactivity Risk",
                    description=(
                        "Patient is allergic to aspirin. Plan mentions "
                        + ", ".join(mentioned).upper()
                        + ", which may cause cross-reactivity in aspirin-allergic patients "
                        + "(risk of bronchospasm, urticaria, anaphylaxis). Up to 30% cross-reactivity rate."
                    ),
                    recommendation=(
                        "Avoid NSAIDs in aspirin-allergic patients. Consider acetaminophen as a safer "
                        "alternative. If NSAIDs essential, consider COX-2 selective inhibitor with caution."
                    ),
                ))

        penicillin_allergy = any(x in a for a in allergies_lower for x in ("penicillin", "amoxicillin", "ampicillin", "pcn"))
        if penicillin_allergy:
            ceph_terms = ["cephalexin", "cefazolin", "ceftriaxone", "cefdinir", "cefuroxime", "cephalosporin"]
            mentioned_ceph = [t for t in ceph_terms if t in combined]
            if mentioned_ceph:
                alerts.append(ClinicalAlert(
                    alert_type="drug_interaction",
                    severity="high",
                    title="Penicillin-Cephalosporin Cross-Reactivity",
                    description=(
                        "Patient has penicillin allergy. Plan mentions "
                        + ", ".join(mentioned_ceph).upper()
                        + ". 5-10% cross-reactivity risk between penicillins and cephalosporins."
                    ),
                    recommendation="Confirm allergy severity. Consider non-beta-lactam alternatives (e.g., azithromycin, clindamycin).",
                ))

        sulfa_allergy = any(x in a for a in allergies_lower for x in ("sulfa", "sulfonamide", "bactrim", "trimethoprim"))
        if sulfa_allergy:
            thiazide_terms = ["hydrochlorothiazide", "hctz", "chlorthalidone", "metolazone", "indapamide"]
            mentioned_thz = [t for t in thiazide_terms if t in combined]
            if mentioned_thz:
                alerts.append(ClinicalAlert(
                    alert_type="drug_interaction",
                    severity="high",
                    title="Sulfonamide-Thiazide Cross-Reactivity",
                    description=(
                        "Patient has sulfonamide allergy. Plan mentions "
                        + ", ".join(mentioned_thz).upper()
                        + ". Potential cross-reactivity between sulfonamide antibiotics and thiazide diuretics."
                    ),
                    recommendation="Use non-thiazide antihypertensives (e.g., ACE inhibitors, CCBs) or confirm tolerance with allergist.",
                ))
    except Exception as e:
        print("[Pipeline] Allergy check error: " + str(e))
    return alerts


def run_clinical_pipeline(transcript, patient_age=None, patient_sex=None):
    t0 = time.time()
    encounter = EncounterResult(transcript=transcript)
    errors = []

    print("[Pipeline] pipe is " + ("LOADED" if pipe is not None else "NONE"))

    if pipe is None:
        encounter.soap_note = SOAPNote(
            subjective="Patient reports: " + transcript.strip(),
            objective="MedGemma not loaded.",
            assessment="Run Cell 6 to load MedGemma.",
            plan="Load AI model then retry.",
        )
        encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
        return encounter

    print("[Pipeline] 1/6 Extracting entities ...")
    try:
        encounter.entities = extract_entities(transcript, patient_age, patient_sex)
        print("[Pipeline] 1/6 OK - " + str(len(encounter.entities.symptoms)) + " symptoms")
    except Exception as e:
        errors.append("Entities: " + str(e))
        print("[Pipeline] 1/6 FAILED: " + str(e))
        traceback.print_exc()

    print("[Pipeline] 2/6 Generating SOAP ...")
    try:
        encounter.soap_note = generate_soap(transcript, encounter.entities, patient_age, patient_sex)
        print("[Pipeline] 2/6 OK")
    except Exception as e:
        errors.append("SOAP: " + str(e))
        print("[Pipeline] 2/6 FAILED: " + str(e))
        traceback.print_exc()

    entities_json = json.dumps(encounter.entities.model_dump(), indent=2)

    print("[Pipeline] 3-6/6 Running steps in parallel ...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(check_drug_interactions, encounter.entities.medications, patient_age, patient_sex): "drugs",
            executor.submit(detect_red_flags, encounter.entities, patient_age, patient_sex): "flags",
            executor.submit(suggest_icd10, encounter.soap_note.assessment, entities_json, patient_age, patient_sex): "icd10",
            executor.submit(suggest_imaging, encounter.entities, encounter.soap_note.assessment, patient_age, patient_sex): "imaging",
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                res = future.result()
                if name == "drugs":
                    encounter.clinical_alerts.extend(res)
                    print("[Pipeline] Drug check OK - " + str(len(res)) + " alerts")
                elif name == "flags":
                    encounter.clinical_alerts.extend(res)
                    print("[Pipeline] Red flags OK - " + str(len(res)) + " alerts")
                elif name == "icd10":
                    encounter.icd10_codes = res
                    print("[Pipeline] ICD-10 OK - " + str(len(res)) + " codes (filtered >= " + str(ICD10_CONFIDENCE_THRESHOLD) + ")")
                elif name == "imaging":
                    encounter.imaging_suggestions = res
                    print("[Pipeline] Imaging OK - " + str(len(res)) + " scans")
            except Exception as e:
                errors.append(name + ": " + str(e))
                print("[Pipeline] " + name + " FAILED: " + str(e))
                traceback.print_exc()

    if not encounter.soap_note.subjective:
        encounter.soap_note.subjective = "Patient reports: " + transcript.strip()
    if not encounter.soap_note.assessment:
        encounter.soap_note.assessment = "See transcript above."

    allergy_alerts = _run_allergy_checks(encounter)
    encounter.clinical_alerts.extend(allergy_alerts)

    if errors:
        encounter.clinical_alerts.append(ClinicalAlert(
            alert_type="system",
            severity="warning",
            title=str(len(errors)) + " step(s) had errors",
            description="; ".join(errors),
            recommendation="Check runtime logs.",
        ))

    encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
    print("[Pipeline] Done in " + str(encounter.processing_time_ms) + "ms, " + str(len(errors)) + " errors")
    return encounter
