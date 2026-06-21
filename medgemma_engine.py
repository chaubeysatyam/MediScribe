import json
import re
import time
import traceback
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import ClinicalEntity, SOAPNote, ClinicalAlert, ICD10Code, ImagingSuggestion, ImageAnalysis, EncounterResult

pipe = None
MODEL_ID = "google/medgemma-4b-it"
_executor = ThreadPoolExecutor(max_workers=4)


def load_medgemma():
    global pipe
    from transformers import pipeline as hf_pipeline
    print(f"[MedGemma] Loading {MODEL_ID} on GPU ...")
    t0 = time.time()
    pipe = hf_pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"[MedGemma] Ready on GPU in {time.time()-t0:.1f}s")
    return pipe


def _generate(prompt, max_tokens=1024):
    if pipe is None:
        raise RuntimeError("MedGemma not loaded.")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    result = pipe(text=messages, max_new_tokens=max_tokens)
    return result[0]["generated_text"][-1]["content"].strip()


def _generate_batch(prompts, max_tokens_list):
    if pipe is None:
        raise RuntimeError("MedGemma not loaded.")
    if len(prompts) == 0:
        return []
    if len(prompts) == 1:
        return [_generate(prompts[0], max_tokens_list[0])]

    max_tokens = max(max_tokens_list)

    try:
        batch_messages = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            batch_messages.append(messages)

        results = pipe(text=batch_messages, max_new_tokens=max_tokens, batch_size=len(prompts))

        outputs = []
        for r in results:
            text = r[0]["generated_text"][-1]["content"].strip()
            outputs.append(text)

        print(f"[Batch] Successfully processed {len(prompts)} prompts in one batch")
        return outputs
    except Exception as e:
        print(f"[Batch] Batching failed ({e}), falling back to sequential processing")
        traceback.print_exc()
        outputs = []
        for prompt, mt in zip(prompts, max_tokens_list):
            outputs.append(_generate(prompt, mt))
        return outputs


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
                parts.append(f"{i}. {flat}")
        return "\n".join(parts)
    if isinstance(v, dict):
        parts = []
        for dk, dv in v.items():
            label = dk.replace("_", " ").title()
            flat = _flatten_soap_value(dv, depth + 1)
            if flat:
                parts.append(f"{label}: {flat}")
        return "\n".join(parts)
    return str(v)


def _build_entities_prompt(transcript):
    return ("You are a clinical entity extraction agent. "
            "Extract ONLY entities EXPLICITLY mentioned in the transcript. Never fabricate data. "
            "If vitals not mentioned, return empty {}. Empty fields use '' or []. "
            "Preserve exact anatomical wording. Include laterality, character, onset, frequency if mentioned. "
            "Return ONLY valid JSON with keys: chief_complaint, symptoms, "
            "vitals, medications, allergies, medical_history, family_history, "
            "social_history, duration."
            "\n\nTranscript:\n" + transcript)


def _build_soap_prompt(transcript, entities):
    return ("You are a board-certified medical documentation specialist. "
            "Generate a SOAP note as valid JSON with keys: subjective, objective, assessment, plan. "
            "All values MUST be strings, not dicts or arrays. "
            "Rules: "
            "- Objective: Only include vitals/findings EXPLICITLY in the transcript. If none, write 'No vitals provided. Recommend: BP, HR, RR, Temp, SpO2.' Never fabricate data. "
            "- Assessment: Structured differential for EACH complaint (e.g. 'Chest pain: ACS vs PE vs musculoskeletal'). No vague phrases. "
            "- Plan: Include labs (troponin, CBC, BMP etc.), management (medications, lifestyle), red flag criteria, and follow-up timeline. "
            "- Never invent data not in the transcript. Preserve exact anatomical descriptions."
            "\n\nTranscript:\n" + transcript
            + "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2))


def _build_drug_interactions_prompt(medications):
    return ("You are a pharmacology expert. Check for drug-drug interactions. "
            "Return ONLY a valid JSON array of objects with keys: "
            "alert_type, severity, title, description, recommendation. "
            "If none found return []"
            "\n\nMedications: " + ", ".join(medications))


def _build_red_flags_prompt(entities):
    return ("You are an emergency triage specialist. Check for RED FLAG symptoms. "
            "Pay special attention to: "
            "- Headache red flags: thunderclap onset, worst headache of life, fever with neck stiffness, "
            "neurological deficits, new onset after age 50, progressive worsening. "
            "- Joint/extremity red flags: signs of systemic inflammatory conditions if joint swelling is present, "
            "potential septic arthritis, compartment syndrome, DVT. "
            "- General red flags: unexplained weight loss, night sweats, progressive symptoms. "
            "Return ONLY a valid JSON array of objects with keys: "
            "alert_type, severity (critical/high/medium/low), title, description, recommendation. "
            "If none return []"
            "\n\n" + json.dumps(entities.model_dump(), indent=2))


def _build_icd10_prompt(assessment, entities_json=""):
    prompt = ("You are an expert ICD-10-CM medical coder. Your task is to assign accurate billing codes. "
              "You MUST suggest 3-5 ICD-10-CM codes that match the symptoms and diagnoses described. "
              "Use the most specific code available. Common examples: "
              "R51.9 (headache, unspecified), R51.0 (headache with orthostatic component), "
              "M79.641 (pain in right hand), M79.642 (pain in left hand), "
              "M79.645 (pain in left finger(s)), M25.462 (joint effusion, left knee), "
              "M79.89 (other specified soft tissue disorders). "
              "Each code MUST directly correspond to a symptom or diagnosis in the text. "
              "Do NOT suggest codes for body parts or conditions not mentioned. "
              "Return ONLY a valid JSON array of objects with keys: code, description, confidence (0.0-1.0). "
              "You MUST return at least 2 codes. Never return an empty array if symptoms are present."
              "\n\nAssessment:\n" + assessment)
    if entities_json:
        prompt += "\n\nExtracted Entities:\n" + entities_json
    return prompt


def _build_imaging_prompt(entities, assessment):
    return ("You are a radiology consultant. Based on the clinical findings, "
            "suggest appropriate imaging studies. "
            "Return ONLY a valid JSON array of objects with keys: "
            "modality (CT/MRI/X-ray/Ultrasound/PET/Nuclear), "
            "body_region (Brain/Chest/Abdomen/Spine/etc), "
            "indication (why this scan is needed), "
            "urgency (stat/urgent/routine), "
            "contrast (with contrast/without contrast/with and without contrast/N/A), "
            "notes (any special instructions). "
            "Suggest 1-4 most relevant scans. If none needed return []"
            "\n\nAssessment:\n" + assessment
            + "\n\nEntities:\n" + json.dumps(entities.model_dump(), indent=2))


def _parse_entities_result(result):
    print(f"[Entities] Raw: {result[:200]}")
    data = _parse_json(result)
    clean = {}
    list_fields = {"symptoms", "medications", "allergies", "medical_history", "family_history", "social_history"}
    for k, v in data.items():
        if k not in ClinicalEntity.model_fields:
            continue
        if v is None:
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


def _parse_soap_result(result):
    print(f"[SOAP] Raw output ({len(result)} chars): {result[:300]}")
    data = _parse_json(result)
    print(f"[SOAP] Parsed keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    clean = {}
    for k in ("subjective", "objective", "assessment", "plan"):
        v = data.get(k, "")
        clean[k] = _flatten_soap_value(v)
        if not clean[k]:
            print(f"[SOAP] WARNING: '{k}' is empty after flattening, raw value was: {repr(v)[:100]}")
    return SOAPNote(**clean)


def _parse_drug_interactions_result(result):
    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def _parse_red_flags_result(result):
    data = _parse_json(result)
    if isinstance(data, list):
        return [ClinicalAlert(**x) for x in data if isinstance(x, dict)]
    return []


def _parse_icd10_result(result):
    print(f"[ICD-10] Raw: {result[:200]}")
    data = _parse_json(result)
    if isinstance(data, list):
        return [ICD10Code(**x) for x in data if isinstance(x, dict) and "code" in x]
    return []


def _parse_imaging_result(result):
    print(f"[Imaging] Raw: {result[:200]}")
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


def extract_entities(transcript):
    result = _generate(_build_entities_prompt(transcript))
    return _parse_entities_result(result)


def generate_soap(transcript, entities):
    result = _generate(_build_soap_prompt(transcript, entities), max_tokens=1500)
    return _parse_soap_result(result)


def check_drug_interactions(medications):
    if len(medications) < 2:
        return []
    result = _generate(_build_drug_interactions_prompt(medications))
    return _parse_drug_interactions_result(result)


def detect_red_flags(entities):
    result = _generate(_build_red_flags_prompt(entities))
    return _parse_red_flags_result(result)


def suggest_icd10(assessment, entities_json=""):
    if not assessment:
        return []
    result = _generate(_build_icd10_prompt(assessment, entities_json))
    return _parse_icd10_result(result)


def suggest_imaging(entities, assessment):
    if not assessment:
        return []
    result = _generate(_build_imaging_prompt(entities, assessment))
    return _parse_imaging_result(result)


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


def _check_nsaid_aspirin_crossreactivity(encounter):
    try:
        allergies_lower = [a.lower() for a in encounter.entities.allergies]
        aspirin_allergy = any(x in a for a in allergies_lower for x in ("aspirin", "asa", "nsaid"))
        if aspirin_allergy:
            nsaid_terms = ["nsaid", "ibuprofen", "naproxen", "diclofenac", "indomethacin", "celecoxib", "meloxicam", "ketorolac"]
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


def _run_stage(label, generate_fn, parse_fn):
    t0 = time.time()
    try:
        raw = generate_fn()
        parsed = parse_fn(raw)
        ms = round((time.time() - t0) * 1000, 1)
        return {"ok": True, "result": parsed, "time_ms": ms, "label": label}
    except Exception as e:
        ms = round((time.time() - t0) * 1000, 1)
        print(f"[Pipeline] {label} FAILED: {e}")
        traceback.print_exc()
        return {"ok": False, "error": str(e), "time_ms": ms, "label": label}


async def run_clinical_pipeline_async(transcript, queue, patient_age=None, patient_sex=None):
    import asyncio
    loop = asyncio.get_event_loop()
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
        await queue.put({"step": 7, "total": 7, "status": "complete", "label": "Pipeline complete",
               "result": encounter})
        return encounter

    await queue.put({"step": 1, "total": 7, "status": "running", "label": "Extracting clinical entities..."})
    step_t0 = time.time()
    try:
        raw = await loop.run_in_executor(_executor, lambda: _generate(_build_entities_prompt(transcript)))
        encounter.entities = _parse_entities_result(raw)
        step_ms = round((time.time() - step_t0) * 1000, 1)
        print(f"[Pipeline] 1/6 OK - {len(encounter.entities.symptoms)} symptoms")
        await queue.put({"step": 1, "total": 7, "status": "done", "label": "Entities extracted", "time_ms": step_ms})
    except Exception as e:
        errors.append(f"Entities: {e}")
        print(f"[Pipeline] 1/6 FAILED: {e}")
        traceback.print_exc()
        step_ms = round((time.time() - step_t0) * 1000, 1)
        await queue.put({"step": 1, "total": 7, "status": "error", "label": f"Entity extraction failed: {e}", "time_ms": step_ms})

    await queue.put({"step": 2, "total": 7, "status": "running", "label": "Generating SOAP note..."})
    await queue.put({"step": 3, "total": 7, "status": "running", "label": "Checking drug interactions..."})
    await queue.put({"step": 4, "total": 7, "status": "running", "label": "Checking red flags..."})

    phase2_t0 = time.time()

    async def _do_soap():
        try:
            raw = await loop.run_in_executor(_executor, lambda: _generate(_build_soap_prompt(transcript, encounter.entities), 1500))
            return _parse_soap_result(raw)
        except Exception as e:
            errors.append(f"SOAP: {e}")
            print(f"[Pipeline] SOAP FAILED: {e}")
            traceback.print_exc()
            return None

    async def _do_drugs():
        try:
            if len(encounter.entities.medications) >= 2:
                raw = await loop.run_in_executor(_executor, lambda: _generate(_build_drug_interactions_prompt(encounter.entities.medications)))
                return _parse_drug_interactions_result(raw)
            return []
        except Exception as e:
            errors.append(f"Drugs: {e}")
            print(f"[Pipeline] Drugs FAILED: {e}")
            return []

    async def _do_redflags():
        try:
            raw = await loop.run_in_executor(_executor, lambda: _generate(_build_red_flags_prompt(encounter.entities)))
            return _parse_red_flags_result(raw)
        except Exception as e:
            errors.append(f"RedFlags: {e}")
            print(f"[Pipeline] RedFlags FAILED: {e}")
            return []

    soap_task = asyncio.create_task(_do_soap())
    drugs_task = asyncio.create_task(_do_drugs())
    redflags_task = asyncio.create_task(_do_redflags())

    soap_result = await soap_task
    soap_ms = round((time.time() - phase2_t0) * 1000, 1)
    if soap_result is not None:
        encounter.soap_note = soap_result
        print(f"[Pipeline] 2/6 OK")
    await queue.put({"step": 2, "total": 7, "status": "done", "label": "SOAP note generated", "time_ms": soap_ms})

    drug_alerts = await drugs_task
    drugs_ms = round((time.time() - phase2_t0) * 1000, 1)
    encounter.clinical_alerts.extend(drug_alerts)
    print(f"[Pipeline] 3/6 OK - {len(drug_alerts)} alerts")
    await queue.put({"step": 3, "total": 7, "status": "done", "label": "Drug interactions checked", "time_ms": drugs_ms})

    red_flags = await redflags_task
    rf_ms = round((time.time() - phase2_t0) * 1000, 1)
    encounter.clinical_alerts.extend(red_flags)
    print(f"[Pipeline] 4/6 OK - {len(red_flags)} alerts")
    await queue.put({"step": 4, "total": 7, "status": "done", "label": "Red flags checked", "time_ms": rf_ms})

    await queue.put({"step": 5, "total": 7, "status": "running", "label": "Suggesting ICD-10 codes..."})
    await queue.put({"step": 6, "total": 7, "status": "running", "label": "Suggesting imaging studies..."})

    phase3_t0 = time.time()
    entities_json = json.dumps(encounter.entities.model_dump(), indent=2)

    async def _do_icd10():
        try:
            if encounter.soap_note.assessment:
                raw = await loop.run_in_executor(_executor, lambda: _generate(_build_icd10_prompt(encounter.soap_note.assessment, entities_json)))
                return _parse_icd10_result(raw)
            return []
        except Exception as e:
            errors.append(f"ICD10: {e}")
            print(f"[Pipeline] ICD10 FAILED: {e}")
            return []

    async def _do_imaging():
        try:
            if encounter.soap_note.assessment:
                raw = await loop.run_in_executor(_executor, lambda: _generate(_build_imaging_prompt(encounter.entities, encounter.soap_note.assessment)))
                return _parse_imaging_result(raw)
            return []
        except Exception as e:
            errors.append(f"Imaging: {e}")
            print(f"[Pipeline] Imaging FAILED: {e}")
            return []

    icd_task = asyncio.create_task(_do_icd10())
    img_task = asyncio.create_task(_do_imaging())

    icd_codes = await icd_task
    icd_ms = round((time.time() - phase3_t0) * 1000, 1)
    encounter.icd10_codes = icd_codes
    print(f"[Pipeline] 5/6 OK - {len(icd_codes)} codes")
    await queue.put({"step": 5, "total": 7, "status": "done", "label": "ICD-10 codes suggested", "time_ms": icd_ms})

    img_suggestions = await img_task
    img_ms = round((time.time() - phase3_t0) * 1000, 1)
    encounter.imaging_suggestions = img_suggestions
    print(f"[Pipeline] 6/6 OK - {len(img_suggestions)} scans")
    await queue.put({"step": 6, "total": 7, "status": "done", "label": "Imaging suggestions ready", "time_ms": img_ms})

    await queue.put({"step": 7, "total": 7, "status": "running", "label": "Running safety checks..."})
    step_t0_safety = time.time()

    if not encounter.soap_note.subjective:
        encounter.soap_note.subjective = f"Patient reports: {transcript.strip()}"
    if not encounter.soap_note.assessment:
        encounter.soap_note.assessment = "See transcript above."

    _check_nsaid_aspirin_crossreactivity(encounter)

    if errors:
        encounter.clinical_alerts.append(ClinicalAlert(
            alert_type="system", severity="warning",
            title=f"{len(errors)} step(s) had errors",
            description="; ".join(errors),
            recommendation="Check runtime logs.",
        ))

    encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)

    step_ms_safety = round((time.time() - step_t0_safety) * 1000, 1)
    await queue.put({"step": 7, "total": 7, "status": "done", "label": "Safety checks complete", "time_ms": step_ms_safety})

    print(f"[Pipeline] Done in {encounter.processing_time_ms}ms, {len(errors)} errors")

    await queue.put({"step": 7, "total": 7, "status": "complete", "label": "Pipeline complete", "result": encounter})
    return encounter


def run_clinical_pipeline_streaming(transcript, patient_age=None, patient_sex=None):
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
        yield {"step": 7, "total": 7, "status": "complete", "label": "Pipeline complete",
               "result": encounter}
        return

    yield {"step": 1, "total": 7, "status": "running", "label": "Extracting clinical entities..."}
    step_t0 = time.time()
    try:
        prompt = _build_entities_prompt(transcript)
        raw = _generate(prompt)
        encounter.entities = _parse_entities_result(raw)
        step_ms = round((time.time() - step_t0) * 1000, 1)
        print(f"[Pipeline] 1/6 OK - {len(encounter.entities.symptoms)} symptoms")
        yield {"step": 1, "total": 7, "status": "done", "label": "Entities extracted",
               "time_ms": step_ms}
    except Exception as e:
        errors.append(f"Entities: {e}")
        print(f"[Pipeline] 1/6 FAILED: {e}")
        traceback.print_exc()
        step_ms = round((time.time() - step_t0) * 1000, 1)
        yield {"step": 1, "total": 7, "status": "error", "label": f"Entity extraction failed: {e}",
               "time_ms": step_ms}

    yield {"step": 2, "total": 7, "status": "running", "label": "Generating SOAP note, checking drugs & red flags..."}
    step_t0 = time.time()

    soap_prompt = _build_soap_prompt(transcript, encounter.entities)
    has_drugs = len(encounter.entities.medications) >= 2
    drug_prompt = _build_drug_interactions_prompt(encounter.entities.medications) if has_drugs else None
    redflags_prompt = _build_red_flags_prompt(encounter.entities)

    batch_prompts = [soap_prompt]
    batch_max_tokens = [1500]
    batch_labels = ["soap"]

    if drug_prompt:
        batch_prompts.append(drug_prompt)
        batch_max_tokens.append(1024)
        batch_labels.append("drugs")

    batch_prompts.append(redflags_prompt)
    batch_max_tokens.append(1024)
    batch_labels.append("redflags")

    try:
        batch_results = _generate_batch(batch_prompts, batch_max_tokens)
    except Exception as e:
        errors.append(f"Phase2 batch: {e}")
        print(f"[Pipeline] Phase 2 batch FAILED: {e}")
        traceback.print_exc()
        batch_results = [None] * len(batch_prompts)

    try:
        soap_raw = batch_results[batch_labels.index("soap")]
        if soap_raw is not None:
            encounter.soap_note = _parse_soap_result(soap_raw)
            print(f"[Pipeline] 2/6 OK")
        else:
            raise RuntimeError("SOAP batch result was None")
    except Exception as e:
        errors.append(f"SOAP: {e}")
        print(f"[Pipeline] 2/6 FAILED: {e}")
        traceback.print_exc()

    step_ms = round((time.time() - step_t0) * 1000, 1)
    yield {"step": 2, "total": 7, "status": "done", "label": "SOAP note generated",
           "time_ms": step_ms}

    yield {"step": 3, "total": 7, "status": "running", "label": "Parsing drug interactions..."}
    step_t0_drugs = time.time()
    try:
        if has_drugs and "drugs" in batch_labels:
            drug_raw = batch_results[batch_labels.index("drugs")]
            if drug_raw is not None:
                drug_alerts = _parse_drug_interactions_result(drug_raw)
                encounter.clinical_alerts.extend(drug_alerts)
                print(f"[Pipeline] 3/6 OK - {len(drug_alerts)} alerts")
            else:
                raise RuntimeError("Drug batch result was None")
        else:
            print(f"[Pipeline] 3/6 OK - skipped (< 2 medications)")
    except Exception as e:
        errors.append(f"Drugs: {e}")
        print(f"[Pipeline] 3/6 FAILED: {e}")

    step_ms_drugs = round((time.time() - step_t0_drugs) * 1000, 1)
    yield {"step": 3, "total": 7, "status": "done", "label": "Drug interactions checked",
           "time_ms": step_ms_drugs}

    yield {"step": 4, "total": 7, "status": "running", "label": "Parsing red flag symptoms..."}
    step_t0_rf = time.time()
    try:
        rf_raw = batch_results[batch_labels.index("redflags")]
        if rf_raw is not None:
            red_flags = _parse_red_flags_result(rf_raw)
            encounter.clinical_alerts.extend(red_flags)
            print(f"[Pipeline] 4/6 OK - {len(red_flags)} alerts")
        else:
            raise RuntimeError("Red flags batch result was None")
    except Exception as e:
        errors.append(f"RedFlags: {e}")
        print(f"[Pipeline] 4/6 FAILED: {e}")

    step_ms_rf = round((time.time() - step_t0_rf) * 1000, 1)
    yield {"step": 4, "total": 7, "status": "done", "label": "Red flags checked",
           "time_ms": step_ms_rf}

    yield {"step": 5, "total": 7, "status": "running", "label": "Suggesting ICD-10 codes & imaging studies..."}
    step_t0_p3 = time.time()

    entities_json = json.dumps(encounter.entities.model_dump(), indent=2)

    batch3_prompts = []
    batch3_max_tokens = []
    batch3_labels = []

    if encounter.soap_note.assessment:
        batch3_prompts.append(_build_icd10_prompt(encounter.soap_note.assessment, entities_json))
        batch3_max_tokens.append(1024)
        batch3_labels.append("icd10")

        batch3_prompts.append(_build_imaging_prompt(encounter.entities, encounter.soap_note.assessment))
        batch3_max_tokens.append(1024)
        batch3_labels.append("imaging")

    if batch3_prompts:
        try:
            batch3_results = _generate_batch(batch3_prompts, batch3_max_tokens)
        except Exception as e:
            errors.append(f"Phase3 batch: {e}")
            print(f"[Pipeline] Phase 3 batch FAILED: {e}")
            traceback.print_exc()
            batch3_results = [None] * len(batch3_prompts)
    else:
        batch3_results = []

    try:
        if "icd10" in batch3_labels:
            icd_raw = batch3_results[batch3_labels.index("icd10")]
            if icd_raw is not None:
                encounter.icd10_codes = _parse_icd10_result(icd_raw)
                print(f"[Pipeline] 5/6 OK - {len(encounter.icd10_codes)} codes")
            else:
                raise RuntimeError("ICD-10 batch result was None")
    except Exception as e:
        errors.append(f"ICD10: {e}")
        print(f"[Pipeline] 5/6 FAILED: {e}")

    step_ms_p3 = round((time.time() - step_t0_p3) * 1000, 1)
    yield {"step": 5, "total": 7, "status": "done", "label": "ICD-10 codes suggested",
           "time_ms": step_ms_p3}

    yield {"step": 6, "total": 7, "status": "running", "label": "Parsing imaging recommendations..."}
    step_t0_img = time.time()
    try:
        if "imaging" in batch3_labels:
            img_raw = batch3_results[batch3_labels.index("imaging")]
            if img_raw is not None:
                encounter.imaging_suggestions = _parse_imaging_result(img_raw)
                print(f"[Pipeline] 6/6 OK - {len(encounter.imaging_suggestions)} scans")
            else:
                raise RuntimeError("Imaging batch result was None")
    except Exception as e:
        errors.append(f"Imaging: {e}")
        print(f"[Pipeline] 6/6 FAILED: {e}")

    step_ms_img = round((time.time() - step_t0_img) * 1000, 1)
    yield {"step": 6, "total": 7, "status": "done", "label": "Imaging suggestions ready",
           "time_ms": step_ms_img}

    yield {"step": 7, "total": 7, "status": "running", "label": "Running safety checks..."}
    step_t0_safety = time.time()

    if not encounter.soap_note.subjective:
        encounter.soap_note.subjective = f"Patient reports: {transcript.strip()}"
    if not encounter.soap_note.assessment:
        encounter.soap_note.assessment = "See transcript above."

    _check_nsaid_aspirin_crossreactivity(encounter)

    if errors:
        encounter.clinical_alerts.append(ClinicalAlert(
            alert_type="system", severity="warning",
            title=f"{len(errors)} step(s) had errors",
            description="; ".join(errors),
            recommendation="Check runtime logs.",
        ))

    encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)

    step_ms_safety = round((time.time() - step_t0_safety) * 1000, 1)
    yield {"step": 7, "total": 7, "status": "done", "label": "Safety checks complete",
           "time_ms": step_ms_safety}

    print(f"[Pipeline] Done in {encounter.processing_time_ms}ms, {len(errors)} errors")

    yield {"step": 7, "total": 7, "status": "complete", "label": "Pipeline complete",
           "result": encounter}


def run_clinical_pipeline(transcript, patient_age=None, patient_sex=None):
    result = None
    for event in run_clinical_pipeline_streaming(transcript, patient_age, patient_sex):
        if event.get("status") == "complete" and "result" in event:
            result = event["result"]
    if result is None:
        result = EncounterResult(transcript=transcript)
    return result
