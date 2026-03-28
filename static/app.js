const API = window.location.origin;
let mediaRecorder = null, audioChunks = [], isRecording = false, timerInterval = null, seconds = 0,
    currentResult = null, audioContext = null, analyser = null, animFrameId = null;
let uploadedFiles = [];
let histPage = 1, histLimit = 20;

const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s);
const on = (el, ev, fn) => { if (el) el.addEventListener(ev, fn); };

function safeHtml(str) {
    return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#x27;");
}

const micBtn = $("#micBtn"), micContainer = $("#micContainer"), recordTimer = $("#recordTimer"),
    recordStatus = $("#recordStatus"), waveformCanvas = $("#waveformCanvas"),
    transcriptInput = $("#transcriptInput"), generateBtn = $("#generateBtn"),
    liveTranscript = $("#liveTranscript"), liveTranscriptText = $("#liveTranscriptText"),
    processingOverlay = $("#processingOverlay"), alertsBadge = $("#alertsBadge"),
    imagingBadge = $("#imagingBadge"), patientName = $("#patientName"),
    patientAge = $("#patientAge"), patientSex = $("#patientSex"),
    imageUpload = $("#imageUpload"), imagePreviewGrid = $("#imagePreviewGrid"),
    transcriptLanguage = $("#transcriptLanguage"),
    charCounter = $("#charCounter"), clearTranscriptBtn = $("#clearTranscriptBtn"),
    historySearch = $("#historySearch"), histPrevBtn = $("#histPrevBtn"),
    histNextBtn = $("#histNextBtn"), histPageLabel = $("#histPageLabel");

$$(".nav-tab").forEach(t => {
    t.addEventListener("click", () => {
        const view = $(`#view${t.dataset.view.charAt(0).toUpperCase() + t.dataset.view.slice(1)}`);
        if (!view) return;
        $$(".nav-tab").forEach(x => x.classList.remove("active"));
        $$(".view").forEach(v => v.classList.remove("active"));
        t.classList.add("active");
        view.classList.add("active");
    });
});

function switchToTab(n) {
    const view = $(`#view${n.charAt(0).toUpperCase() + n.slice(1)}`);
    if (!view) return;
    $$(".nav-tab").forEach(t => t.classList.toggle("active", t.dataset.view === n));
    $$(".view").forEach(v => v.classList.remove("active"));
    view.classList.add("active");
}

const DRAFT_KEY_TRANSCRIPT = "ms_draft_transcript";
const DRAFT_KEY_NAME = "ms_draft_name";
const DRAFT_KEY_AGE = "ms_draft_age";
const DRAFT_KEY_SEX = "ms_draft_sex";
const DRAFT_KEY_LANG = "ms_draft_lang";

function saveDraft() {
    try {
        localStorage.setItem(DRAFT_KEY_TRANSCRIPT, transcriptInput ? transcriptInput.value : "");
        localStorage.setItem(DRAFT_KEY_NAME, patientName ? patientName.value : "");
        localStorage.setItem(DRAFT_KEY_AGE, patientAge ? patientAge.value : "");
        localStorage.setItem(DRAFT_KEY_SEX, patientSex ? patientSex.value : "");
        localStorage.setItem(DRAFT_KEY_LANG, transcriptLanguage ? transcriptLanguage.value : "en");
    } catch (e) { }
}

function restoreDraft() {
    try {
        const t = localStorage.getItem(DRAFT_KEY_TRANSCRIPT);
        const n = localStorage.getItem(DRAFT_KEY_NAME);
        const a = localStorage.getItem(DRAFT_KEY_AGE);
        const s = localStorage.getItem(DRAFT_KEY_SEX);
        const l = localStorage.getItem(DRAFT_KEY_LANG);
        if (t && transcriptInput) { transcriptInput.value = t; updateCharCounter(); }
        if (n && patientName) patientName.value = n;
        if (a && patientAge) patientAge.value = a;
        if (s && patientSex) patientSex.value = s;
        if (l && transcriptLanguage) transcriptLanguage.value = l;
    } catch (e) { }
}

function clearDraft() {
    try {
        [DRAFT_KEY_TRANSCRIPT, DRAFT_KEY_NAME, DRAFT_KEY_AGE, DRAFT_KEY_SEX, DRAFT_KEY_LANG]
            .forEach(k => localStorage.removeItem(k));
    } catch (e) { }
}

on(transcriptInput, "input", () => { saveDraft(); updateCharCounter(); checkGenerateBtn(); });
on(patientName, "input", saveDraft);
on(patientAge, "input", saveDraft);
on(patientSex, "change", saveDraft);
on(transcriptLanguage, "change", saveDraft);

on(clearTranscriptBtn, "click", () => {
    if (transcriptInput) { transcriptInput.value = ""; updateCharCounter(); checkGenerateBtn(); }
    if (liveTranscript) liveTranscript.style.display = "none";
    clearDraft();
});

function updateCharCounter() {
    if (!charCounter || !transcriptInput) return;
    const len = transcriptInput.value.length;
    charCounter.textContent = `${len.toLocaleString()} char${len !== 1 ? "s" : ""}`;
}

on(imageUpload, "change", e => {
    Array.from(e.target.files).forEach(file => addImageFile(file));
    imageUpload.value = "";
    checkGenerateBtn();
});

const uploadZone = $("#uploadZone");
if (uploadZone) {
    uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", e => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        Array.from(e.dataTransfer.files).forEach(f => addImageFile(f));
        checkGenerateBtn();
    });
}

function addImageFile(file) {
    const idx = uploadedFiles.length;
    uploadedFiles.push(file);
    const reader = new FileReader();
    reader.onload = ev => {
        const div = document.createElement("div");
        div.className = "image-preview-item";
        div.dataset.idx = idx;
        div.innerHTML = `<img src="${ev.target.result}" alt="${safeHtml(file.name)}">
            <button class="image-preview-remove" onclick="removeImage(${idx},this)">X</button>
            <div class="image-preview-name">${safeHtml(file.name)}</div>`;
        imagePreviewGrid.appendChild(div);
    };
    reader.readAsDataURL(file);
}

function removeImage(idx, btn) {
    uploadedFiles[idx] = null;
    btn.parentElement.remove();
}

function checkGenerateBtn() {
    if (!transcriptInput || !generateBtn) return;
    generateBtn.disabled = !transcriptInput.value.trim().length;
}

let REC_STATE = "idle";
if (micBtn) micBtn.querySelectorAll("svg, path, circle, rect, polyline").forEach(e => e.style.pointerEvents = "none");

let _lastMicFire = 0;
function handleMicToggle(e) {
    const now = Date.now();
    if (now - _lastMicFire < 400) return;
    _lastMicFire = now;
    if (REC_STATE === "recording") { _doStop(); return; }
    if (REC_STATE !== "idle") return;
    REC_STATE = "processing";
    _doStart().catch(err => {
        REC_STATE = "idle";
        if (recordStatus) recordStatus.textContent = "Microphone access denied";
        console.error(err);
    });
}
micBtn.addEventListener("click", handleMicToggle);
micBtn.addEventListener("touchend", handleMicToggle);
window.stopRecording = function () { _doStop(); };

async function _doStart() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
    let mt = "audio/webm";
    for (const t of types) if (MediaRecorder.isTypeSupported(t)) { mt = t; break; }

    mediaRecorder = new MediaRecorder(stream, { mimeType: mt });
    audioChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        REC_STATE = "processing";
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        await processAudio(blob);
        REC_STATE = "idle";
    };

    mediaRecorder.start(250);
    REC_STATE = "recording";
    isRecording = true;
    micBtn.classList.add("recording");
    micContainer.classList.add("recording");
    recordStatus.textContent = "Recording\u2026 tap again to stop";
    recordTimer.classList.add("active");
    seconds = 0; updateTimer();
    timerInterval = setInterval(() => { seconds++; updateTimer(); }, 1000);
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    audioContext.createMediaStreamSource(stream).connect(analyser);
    analyser.fftSize = 256;
    drawWave();
}

function _doStop() {
    if (REC_STATE !== "recording") return;
    isRecording = false;
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    micBtn.classList.remove("recording");
    micContainer.classList.remove("recording");
    recordStatus.textContent = "Processing audio\u2026";
    recordTimer.classList.remove("active");
    clearInterval(timerInterval);
    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (audioContext) audioContext.close();
    audioContext = null; analyser = null;
}

function stopRecording() { _doStop(); }
function startRecording() { return _doStart(); }

function updateTimer() {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(seconds % 60).padStart(2, "0");
    if (recordTimer) recordTimer.textContent = `${m}:${s}`;
}

function drawWave() {
    if (!analyser || !waveformCanvas) return;
    const c = waveformCanvas, ctx = c.getContext("2d");
    c.width = c.offsetWidth * 2; c.height = c.offsetHeight * 2; ctx.scale(2, 2);
    const buf = analyser.frequencyBinCount, data = new Uint8Array(buf);
    (function d() {
        animFrameId = requestAnimationFrame(d); analyser.getByteFrequencyData(data);
        const w = c.offsetWidth, h = c.offsetHeight; ctx.clearRect(0, 0, w, h);
        const bw = w / buf * 2; let x = 0;
        for (let i = 0; i < buf; i++) {
            const bh = data[i] / 255 * h * 0.8;
            const g = ctx.createLinearGradient(0, h, 0, h - bh);
            g.addColorStop(0, "rgba(14,165,233,.3)"); g.addColorStop(1, "rgba(6,182,212,.8)");
            ctx.fillStyle = g; ctx.fillRect(x, h - bh, bw - 1, bh); x += bw;
        }
    })();
}

on(generateBtn, "click", () => {
    if (!transcriptInput) return;
    const t = transcriptInput.value.trim();
    if (t) { showProcessing(); generateFromTranscript(t); }
});

async function processAudio(blob) {
    const lang = transcriptLanguage ? transcriptLanguage.value : "en";
    recordStatus.textContent = "Transcribing voice...";
    try {
        const fd = new FormData();
        fd.append("audio", blob, "recording.webm");
        fd.append("language", lang);
        const r = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
        if (!r.ok) {
            let d = `HTTP ${r.status}`;
            try { const j = await r.json(); d = j.detail || JSON.stringify(j); } catch { try { d = await r.text(); } catch { } }
            throw new Error(`Transcription failed: ${d}`);
        }
        const t = await r.json();
        if (liveTranscript) liveTranscript.style.display = "block";
        if (liveTranscriptText) liveTranscriptText.textContent = t.text || "";
        if (transcriptInput) { transcriptInput.value = t.text || ""; updateCharCounter(); }
        if (t.translated && t.original_text) {
            const badge = document.createElement("div");
            badge.style.cssText = "margin-top:8px;font-size:12px;color:var(--accent);opacity:0.85;";
            badge.textContent = `\u{1F310} Translated by Sarvam AI (${t.language} \u2192 English)`;
            liveTranscriptText.parentElement.appendChild(badge);
        }
        checkGenerateBtn(); saveDraft();
        recordStatus.textContent = "Transcription complete. Review and click Generate.";
        hideProcessing();
    } catch (e) { hideProcessing(); recordStatus.textContent = "Error: " + e.message; console.error(e); }
}

async function generateFromTranscript(transcript) {
    try {
        simulateSteps();
        const fd = new FormData();
        fd.append("transcript", transcript);
        fd.append("patient_name", patientName ? patientName.value || "" : "");
        fd.append("language", transcriptLanguage ? transcriptLanguage.value : "en");
        if (patientAge && patientAge.value) fd.append("patient_age", patientAge.value);
        if (patientSex && patientSex.value) fd.append("patient_sex", patientSex.value);
        uploadedFiles.forEach(f => { if (f) fd.append("images", f, f.name); });

        const r = await fetch(`${API}/api/generate`, { method: "POST", body: fd });
        if (!r.ok) {
            let d = `HTTP ${r.status}`;
            try { const j = await r.json(); d = j.detail || JSON.stringify(j); } catch { try { d = await r.text(); } catch { } }
            throw new Error(`Generation failed: ${d}`);
        }
        currentResult = await r.json();
        hideProcessing();
        renderResults(currentResult);
        renderAlerts(currentResult);
        renderImaging(currentResult);
        loadHistory();
        switchToTab("results");
        recordStatus.textContent = "Tap to start recording";
        uploadedFiles = [];
        imagePreviewGrid.innerHTML = "";
    } catch (e) { hideProcessing(); recordStatus.textContent = "Error: " + e.message; console.error(e); }
}

function showProcessing() { if (processingOverlay) processingOverlay.classList.add("active"); }
function hideProcessing() {
    if (processingOverlay) processingOverlay.classList.remove("active");
    $$(".processing-step").forEach(s => s.classList.remove("active", "done"));
}

function activateStep(n) {
    for (let i = 1; i <= 7; i++) {
        const s = $(`#step${i}`); if (!s) continue;
        if (i < n) { s.classList.remove("active"); s.classList.add("done"); }
        else if (i === n) { s.classList.add("active"); s.classList.remove("done"); }
        else s.classList.remove("active", "done");
    }
}

function simulateSteps() {
    activateStep(1);
    setTimeout(() => activateStep(2), 2000); setTimeout(() => activateStep(3), 5000);
    setTimeout(() => activateStep(4), 7000); setTimeout(() => activateStep(5), 9000);
    setTimeout(() => activateStep(6), 11000); setTimeout(() => activateStep(7), 13000);
}

function _soapPlainText(d) {
    const s = d.soap_note || {};
    return [
        `SOAP NOTE — ${d.patient_name || "Patient"}\n${"=".repeat(48)}`,
        `S — Subjective\n${s.subjective || ""}`,
        `O — Objective\n${s.objective || ""}`,
        `A — Assessment\n${s.assessment || ""}`,
        `P — Plan\n${s.plan || ""}`,
    ].join("\n\n");
}

async function copySoap() {
    if (!currentResult) return;
    try {
        await navigator.clipboard.writeText(_soapPlainText(currentResult));
        const btn = $("#copySoapBtn");
        const orig = btn ? btn.textContent : "";
        if (btn) { btn.textContent = "\u2713 Copied!"; setTimeout(() => { btn.textContent = orig; }, 2000); }
    } catch (e) { alert("Copy failed: " + e.message); }
}

function printSoap() {
    if (!currentResult) return;
    const win = window.open("", "_blank");
    const s = currentResult.soap_note || {};
    win.document.write(`<html><head><title>SOAP Note</title>
        <style>body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;color:#111}
        h1{font-size:20px;border-bottom:2px solid #333;padding-bottom:8px}
        h2{font-size:15px;color:#444;margin-bottom:4px}
        p{white-space:pre-wrap;line-height:1.7;margin-bottom:24px}</style></head><body>
        <h1>SOAP Note — ${safeHtml(currentResult.patient_name || "Patient")}</h1>
        <h2>S — Subjective</h2><p>${safeHtml(s.subjective)}</p>
        <h2>O — Objective</h2><p>${safeHtml(s.objective)}</p>
        <h2>A — Assessment</h2><p>${safeHtml(s.assessment)}</p>
        <h2>P — Plan</h2><p>${safeHtml(s.plan)}</p>
        </body></html>`);
    win.document.close();
    win.print();
}

function renderResults(d) {
    const s = d.soap_note || {}, e = d.entities || {}, icd = d.icd10_codes || [];
    const name = d.patient_name ? `<span style="color:var(--accent)">${safeHtml(d.patient_name)}</span> &mdash; ` : "";
    let h = `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px;">
        <h2 style="font-size:18px;font-weight:700">&#x1F4CB; ${name}Clinical Documentation</h2>
        <div style="display:flex;gap:8px;align-items:center;">
            <span class="time-badge">&#x26A1; ${(d.processing_time_ms / 1000).toFixed(1)}s</span>
            <button class="toolbar-btn" id="copySoapBtn" onclick="copySoap()" style="padding:6px 14px;">&#x1F4CB; Copy SOAP</button>
            <button class="toolbar-btn" onclick="printSoap()" style="padding:6px 14px;">&#x1F5A8; Print</button>
        </div>
    </div>
    <div class="results-grid">
        <div class="card">
            <div class="card-header"><h2>&#x1F4DD; SOAP Note</h2></div>
            <div class="card-body">
                ${soapSec("S", "Subjective", s.subjective)}
                ${soapSec("O", "Objective", s.objective)}
                ${soapSec("A", "Assessment", s.assessment)}
                ${soapSec("P", "Plan", s.plan)}
            </div>
        </div>
        <div>
            <div class="card" style="margin-bottom:20px">
                <div class="card-header"><h2>&#x1F3F7;&#xFE0F; ICD-10 Codes</h2></div>
                <div class="card-body">${icd.length
            ? `<div class="icd-list">${icd.map(c => `<div class="icd-chip"><span class="icd-code">${safeHtml(c.code)}</span><span class="icd-desc">${safeHtml(c.description)}</span><span class="icd-conf">${Math.round((c.confidence || 0) * 100)}%</span></div>`).join("")}</div>`
            : '<p style="color:var(--text-muted);font-size:13px">No codes</p>'}</div>
            </div>
            <div class="card">
                <div class="card-header"><h2>&#x1F50D; Extracted Entities</h2></div>
                <div class="card-body">
                    ${entGrp("Chief Complaint", e.chief_complaint ? [e.chief_complaint] : [])}
                    ${entGrp("Symptoms", e.symptoms)}
                    ${entGrp("Medications", e.medications)}
                    ${entGrp("Allergies", e.allergies)}
                    ${entGrp("Vitals", Object.entries(e.vitals || {}).map(([k, v]) => `${k}: ${v}`))}
                    ${entGrp("Medical History", e.medical_history)}
                </div>
            </div>
        </div>
    </div>`;
    const el = $("#resultsContent");
    if (el) el.innerHTML = h;
}

function soapSec(l, label, text) {
    const formatted = safeHtml(text || "No data").replace(/\n/g, "<br>");
    return `<div class="soap-section"><div class="soap-label ${l.toLowerCase()}">${l} &mdash; ${label}</div><div class="soap-text">${formatted}</div></div>`;
}

function entGrp(title, items) {
    if (!items || !items.length) return "";
    return `<div class="entity-group"><h4>${safeHtml(title)}</h4><div class="entity-tags">${items.map(i => `<span class="entity-tag">${safeHtml(i)}</span>`).join("")}</div></div>`;
}

function renderImaging(d) {
    const suggestions = d.imaging_suggestions || [], analyses = d.image_analyses || [];
    const total = suggestions.length + analyses.length;
    if (imagingBadge) { imagingBadge.textContent = total; imagingBadge.classList.toggle("visible", total > 0); }

    if (total === 0) {
        const empty = $("#imagingContent");
        if (empty) empty.innerHTML = '<div class="empty-state"><div class="icon">&#x1F3E5;</div><h3>No Imaging Data</h3><p>Upload medical images or generate notes to see imaging suggestions</p></div>';
        return;
    }

    let h = "";
    if (analyses.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F4F7; Uploaded Image Analysis (${analyses.length})</div>
        <div class="imaging-grid">${analyses.map(a => `
            <div class="image-analysis-card">
                ${a.filename ? `<img class="analysis-image" src="/uploads/${safeHtml(a.filename)}" alt="${safeHtml(a.filename)}">` : ""}
                <div class="analysis-body">
                    <span class="modality-badge ${getModalityClass(a.image_type)}">${safeHtml(a.image_type || "Unknown")}</span>
                    <h4>${safeHtml(a.body_part || "Medical Image")}</h4>
                    <div class="field-label">Findings</div><div class="field-value">${safeHtml(a.findings || "N/A")}</div>
                    <div class="field-label">Impression</div><div class="field-value">${safeHtml(a.impression || "N/A")}</div>
                    ${(a.abnormalities || []).length ? `<div class="field-label">Abnormalities</div><div>${a.abnormalities.map(x => `<span class="abnormality-tag">${safeHtml(x)}</span>`).join("")}</div>` : ""}
                    ${a.recommendations ? `<div class="field-label">Recommendations</div><div class="field-value">${safeHtml(a.recommendations)}</div>` : ""}
                </div>
            </div>`).join("")}</div></div>`;
    }

    if (suggestions.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F52C; Recommended Imaging Studies (${suggestions.length})</div>
        <div class="imaging-grid">${suggestions.map(s => `
            <div class="imaging-card">
                <span class="modality-badge ${getModalityClass(s.modality)}">${safeHtml(s.modality || "Imaging")}</span>
                <span class="urgency-badge ${safeHtml((s.urgency || "routine").toLowerCase())}">${safeHtml(s.urgency || "routine")}</span>
                <h4>${safeHtml(s.body_region || "Unspecified Region")}</h4>
                <div class="field-label">Indication</div><div class="field-value">${safeHtml(s.indication || "N/A")}</div>
                <div class="field-label">Contrast</div><div class="field-value">${safeHtml(s.contrast || "N/A")}</div>
                ${s.notes ? `<div class="field-label">Notes</div><div class="field-value">${safeHtml(s.notes)}</div>` : ""}
            </div>`).join("")}</div></div>`;
    }

    const imagingContent = $("#imagingContent");
    if (imagingContent) imagingContent.innerHTML = h;
}

function getModalityClass(mod) {
    if (!mod) return "default";
    const m = mod.toLowerCase();
    if (m.includes("x-ray") || m.includes("xray") || m.includes("radiograph")) return "xray";
    if (m.includes("ct")) return "ct";
    if (m.includes("mri") || m.includes("magnetic")) return "mri";
    if (m.includes("ultra")) return "ultrasound";
    if (m.includes("pet") || m.includes("nuclear")) return "pet";
    return "default";
}

function renderAlerts(d) {
    const alerts = d.clinical_alerts || [];
    const cnt = alerts.filter(a => a.severity === "critical" || a.severity === "warning").length;
    if (alertsBadge) { alertsBadge.textContent = cnt; alertsBadge.classList.toggle("visible", cnt > 0); }
    if (!alerts.length) {
        $("#alertsContent").innerHTML = '<div class="card"><div class="card-header"><h2>&#x1F6E1;&#xFE0F; Clinical Decision Support</h2></div><div class="card-body"><div class="empty-state"><div class="icon">&#x2705;</div><h3>No Alerts</h3><p>No concerns detected</p></div></div></div>';
        return;
    }
    const so = { critical: 0, warning: 1, info: 2 };
    alerts.sort((a, b) => (so[a.severity] || 3) - (so[b.severity] || 3));
    const icons = { critical: "&#x1F6A8;", warning: "&#x26A0;&#xFE0F;", info: "&#x2139;&#xFE0F;" };
    const tl = { drug_interaction: "&#x1F48A; Drug Interaction", red_flag: "&#x1F6A9; Red Flag", guideline: "&#x1F4CB; Guideline", screening: "&#x1F52C; Screening", system: "&#x2699;&#xFE0F; System" };
    $("#alertsContent").innerHTML = `<div class="card">
        <div class="card-header"><h2>&#x1F6E1;&#xFE0F; Clinical Decision Support</h2><span class="time-badge">${alerts.length} alert${alerts.length > 1 ? "s" : ""}</span></div>
        <div class="card-body">${alerts.map(a => `<div class="alert-item ${safeHtml(a.severity)}">
            <span class="alert-icon">${icons[a.severity] || "&#x2753;"}</span>
            <div class="alert-content">
                <span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">${tl[a.alert_type] || safeHtml(a.alert_type)}</span>
                <h4>${safeHtml(a.title)}</h4><p>${safeHtml(a.description)}</p>
                ${a.recommendation ? `<div class="recommendation">&#x1F4A1; ${safeHtml(a.recommendation)}</div>` : ""}
            </div>
        </div>`).join("")}</div></div>`;
}

async function loadHistory(page) {
    histPage = page || histPage;
    try {
        const q = historySearch ? historySearch.value.trim() : "";
        let url;
        if (q) {
            url = `${API}/api/encounters/search?q=${encodeURIComponent(q)}&patient=${encodeURIComponent(q)}`;
        } else {
            url = `${API}/api/encounters?page=${histPage}&limit=${histLimit}`;
        }
        const r = await fetch(url);
        const enc = await r.json();
        renderHistory(enc);
        if (histPageLabel) histPageLabel.textContent = `Page ${histPage}`;
    } catch (e) { console.error(e); }
}

let _histSearchTimer;
on(historySearch, "input", () => {
    clearTimeout(_histSearchTimer);
    _histSearchTimer = setTimeout(() => { histPage = 1; loadHistory(); }, 400);
});
on(histPrevBtn, "click", () => { if (histPage > 1) loadHistory(histPage - 1); });
on(histNextBtn, "click", () => loadHistory(histPage + 1));

function renderHistory(enc) {
    const l = $("#historyList");
    if (!l) return;
    const pagination = $("#historyPagination");
    if (!enc || !enc.length) {
        l.innerHTML = '<div class="empty-state"><div class="icon">&#x1F4C2;</div><h3>No Encounters</h3><p>Processed encounters stored locally</p></div>';
        if (pagination) pagination.style.display = "none";
        return;
    }
    if (pagination) pagination.style.display = "flex";
    window._encounters = enc;
    l.innerHTML = enc.map((e, i) => {
        const d = new Date(e.timestamp);
        const ac = (e.clinical_alerts || []).length;
        const ic = (e.image_analyses || []).length;
        const cc = safeHtml(e.patient_name || (e.entities?.chief_complaint) || ((e.transcript || "").substring(0, 60) + "\u2026"));
        return `<div class="history-item" onclick="viewEncounterDetail(${i})">
            <div class="history-meta"><h4>${cc}</h4>
            ${e.patient_name ? `<p style="color:var(--accent);font-size:11px;margin-bottom:2px">${safeHtml(e.patient_name)}</p>` : ""}
            <p>${d.toLocaleDateString()} ${d.toLocaleTimeString()}</p></div>
            <div class="history-badges">
                ${ic > 0 ? `<span class="history-badge images">&#x1F4F7; ${ic}</span>` : ""}
                ${ac > 0 ? `<span class="history-badge alerts">&#x26A0;&#xFE0F; ${ac}</span>` : ""}
                <span class="history-badge time">&#x26A1; ${(e.processing_time_ms / 1000).toFixed(1)}s</span>
                <button class="delete-btn" onclick="event.stopPropagation();deleteEnc('${safeHtml(e.id)}')">&#x1F5D1;&#xFE0F;</button>
            </div>
        </div>`;
    }).join("");
}

function viewEncounterDetail(idx) {
    const e = window._encounters[idx];
    if (!e) return;
    currentResult = e;
    const d = new Date(e.timestamp);
    const s = e.soap_note || {};
    const ent = e.entities || {};
    const icd = e.icd10_codes || [];
    const alerts = e.clinical_alerts || [];
    const imgs = e.image_analyses || [];
    const suggestions = e.imaging_suggestions || [];

    let h = `<button class="detail-back" onclick="switchToTab('history')">&larr; Back to History</button>
    <div class="detail-header">
        <div>
            <div class="detail-patient">${safeHtml(e.patient_name || "Unnamed Patient")}</div>
            <div style="color:var(--text-muted);font-size:13px">${d.toLocaleDateString()} ${d.toLocaleTimeString()} &bull; ${(e.processing_time_ms / 1000).toFixed(1)}s processing</div>
        </div>
        <div style="display:flex;gap:8px;margin-top:8px;">
            <button class="toolbar-btn" onclick="copySoap()" style="padding:6px 14px;">&#x1F4CB; Copy SOAP</button>
            <button class="toolbar-btn" onclick="printSoap()" style="padding:6px 14px;">&#x1F5A8; Print</button>
        </div>
    </div>`;

    h += `<div class="detail-section"><div class="detail-section-title">&#x1F399;&#xFE0F; Original Transcript</div>
        <div class="card"><div class="card-body" style="font-size:13px;line-height:1.8;color:var(--text-secondary)">${safeHtml(e.transcript || "N/A")}</div></div></div>`;

    h += `<div class="detail-section"><div class="detail-section-title">&#x1F4DD; SOAP Note</div>
        <div class="card"><div class="card-body">
            ${soapSec("S", "Subjective", s.subjective)}
            ${soapSec("O", "Objective", s.objective)}
            ${soapSec("A", "Assessment", s.assessment)}
            ${soapSec("P", "Plan", s.plan)}
        </div></div></div>`;

    const entHtml = entGrp("Chief Complaint", ent.chief_complaint ? [ent.chief_complaint] : []) +
        entGrp("Symptoms", ent.symptoms) + entGrp("Medications", ent.medications) +
        entGrp("Allergies", ent.allergies) +
        entGrp("Vitals", Object.entries(ent.vitals || {}).map(([k, v]) => `${k}: ${v}`)) +
        entGrp("Medical History", ent.medical_history);
    if (entHtml) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F50D; Extracted Entities</div>
            <div class="card"><div class="card-body">${entHtml}</div></div></div>`;
    }

    if (icd.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F3F7;&#xFE0F; ICD-10 Codes</div>
            <div class="card"><div class="card-body"><div class="icd-list">${icd.map(c => `<div class="icd-chip"><span class="icd-code">${safeHtml(c.code)}</span><span class="icd-desc">${safeHtml(c.description)}</span><span class="icd-conf">${Math.round((c.confidence || 0) * 100)}%</span></div>`).join("")}</div></div></div></div>`;
    }

    if (imgs.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F4F7; Medical Image Analysis (${imgs.length})</div>
        <div class="imaging-grid">${imgs.map(a => `
            <div class="image-analysis-card">
                ${a.filename ? `<img class="analysis-image" src="/uploads/${safeHtml(a.filename)}" alt="${safeHtml(a.filename)}">` : ""}
                <div class="analysis-body">
                    <span class="modality-badge ${getModalityClass(a.image_type)}">${safeHtml(a.image_type || "Unknown")}</span>
                    <h4>${safeHtml(a.body_part || "Medical Image")}</h4>
                    <div class="field-label">Findings</div><div class="field-value">${safeHtml(a.findings || "N/A")}</div>
                    <div class="field-label">Impression</div><div class="field-value">${safeHtml(a.impression || "N/A")}</div>
                    ${(a.abnormalities || []).length ? `<div class="field-label">Abnormalities</div><div>${a.abnormalities.map(x => `<span class="abnormality-tag">${safeHtml(x)}</span>`).join("")}</div>` : ""}
                    ${a.recommendations ? `<div class="field-label">Recommendations</div><div class="field-value">${safeHtml(a.recommendations)}</div>` : ""}
                </div>
            </div>`).join("")}</div></div>`;
    }

    if (suggestions.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F52C; Recommended Imaging (${suggestions.length})</div>
        <div class="imaging-grid">${suggestions.map(sg => `
            <div class="imaging-card">
                <span class="modality-badge ${getModalityClass(sg.modality)}">${safeHtml(sg.modality || "Imaging")}</span>
                <span class="urgency-badge ${safeHtml((sg.urgency || "routine").toLowerCase())}">${safeHtml(sg.urgency || "routine")}</span>
                <h4>${safeHtml(sg.body_region || "")}</h4>
                <div class="field-label">Indication</div><div class="field-value">${safeHtml(sg.indication || "N/A")}</div>
                ${sg.contrast ? `<div class="field-label">Contrast</div><div class="field-value">${safeHtml(sg.contrast)}</div>` : ""}
                ${sg.notes ? `<div class="field-label">Notes</div><div class="field-value">${safeHtml(sg.notes)}</div>` : ""}
            </div>`).join("")}</div></div>`;
    }

    if (alerts.length) {
        const icons = { critical: "&#x1F6A8;", warning: "&#x26A0;&#xFE0F;", info: "&#x2139;&#xFE0F;" };
        const tl = { drug_interaction: "&#x1F48A; Drug", red_flag: "&#x1F6A9; Red Flag", guideline: "&#x1F4CB; Guideline", screening: "&#x1F52C; Screening", system: "&#x2699;&#xFE0F; System" };
        h += `<div class="detail-section"><div class="detail-section-title">&#x26A0;&#xFE0F; Clinical Alerts (${alerts.length})</div>
        <div class="card"><div class="card-body">${alerts.map(a => `<div class="alert-item ${safeHtml(a.severity)}">
            <span class="alert-icon">${icons[a.severity] || ""}</span>
            <div class="alert-content"><span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">${tl[a.alert_type] || safeHtml(a.alert_type)}</span>
            <h4>${safeHtml(a.title)}</h4><p>${safeHtml(a.description)}</p>
            ${a.recommendation ? `<div class="recommendation">&#x1F4A1; ${safeHtml(a.recommendation)}</div>` : ""}</div>
        </div>`).join("")}</div></div></div>`;
    }

    const detailContent = $("#detailContent");
    if (detailContent) detailContent.innerHTML = h;
    switchToTab("detail");
}

async function deleteEnc(id) {
    try { await fetch(`${API}/api/encounters/${id}`, { method: "DELETE" }); loadHistory(); } catch (e) { console.error(e); }
}

async function checkHealth() {
    const s = $("#serverStatus");
    if (!s) return;
    try {
        const r = await fetch(`${API}/health`); const d = await r.json();
        if (d.medgemma_loaded && d.whisper_loaded) {
            s.innerHTML = '<span class="status-dot"></span><span>MedGemma Online</span>';
            s.removeAttribute("style");
        } else {
            s.innerHTML = '<span class="status-dot" style="background:var(--warning)"></span><span>Loading Models...</span>';
            s.style.borderColor = "rgba(245,158,11,.2)"; s.style.background = "var(--warning-bg)"; s.style.color = "var(--warning)";
        }
    } catch {
        s.innerHTML = '<span class="status-dot" style="background:var(--critical)"></span><span>Server Offline</span>';
        s.style.borderColor = "rgba(239,68,68,.2)"; s.style.background = "var(--critical-bg)"; s.style.color = "var(--critical)";
    }
}

restoreDraft();
updateCharCounter();
checkGenerateBtn();
checkHealth();
setInterval(checkHealth, 15000);
if ($("#historyList")) loadHistory();
