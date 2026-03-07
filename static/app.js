const API = window.location.origin;
let mediaRecorder = null, audioChunks = [], isRecording = false, timerInterval = null, seconds = 0, currentResult = null, audioContext = null, analyser = null, animFrameId = null;
let uploadedFiles = []; // Track uploaded image files

const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s);
const on = (el, event, handler) => { if (el) el.addEventListener(event, handler); };

const micBtn = $("#micBtn"), micContainer = $("#micContainer"), recordTimer = $("#recordTimer"), recordStatus = $("#recordStatus"), waveformCanvas = $("#waveformCanvas"), transcriptInput = $("#transcriptInput"), generateBtn = $("#generateBtn"), liveTranscript = $("#liveTranscript"), liveTranscriptText = $("#liveTranscriptText"), processingOverlay = $("#processingOverlay"), alertsBadge = $("#alertsBadge"), imagingBadge = $("#imagingBadge"), patientName = $("#patientName"), patientAge = $("#patientAge"), patientSex = $("#patientSex"), imageUpload = $("#imageUpload"), imagePreviewGrid = $("#imagePreviewGrid");

// Tab navigation
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

// Image upload handling
on(imageUpload, "change", (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => {
        uploadedFiles.push(file);
        const reader = new FileReader();
        reader.onload = (ev) => {
            const div = document.createElement("div");
            div.className = "image-preview-item";
            div.innerHTML = `<img src="${ev.target.result}" alt="${file.name}">
                <button class="image-preview-remove" onclick="removeImage(${uploadedFiles.length - 1},this)">X</button>
                <div class="image-preview-name">${file.name}</div>`;
            imagePreviewGrid.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
    checkGenerateBtn();
    imageUpload.value = ""; // Reset to allow re-uploading same file
});

function removeImage(idx, btn) {
    uploadedFiles[idx] = null;
    btn.parentElement.remove();
}

function checkGenerateBtn() {
    if (!transcriptInput || !generateBtn) return;
    const hasText = transcriptInput.value.trim().length > 0;
    generateBtn.disabled = !hasText;
}

// Recording
let isProcessingMic = false;
on(micBtn, "click", async () => {
    if (isProcessingMic) return;
    if (isRecording) {
        stopRecording();
    } else {
        isProcessingMic = true;
        await startRecording();
        isProcessingMic = false;
    }
});

async function startRecording() {
    try {
        const s = await navigator.mediaDevices.getUserMedia({ audio: true });
        const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
        let mt = "audio/webm";
        for (const t of types) if (MediaRecorder.isTypeSupported(t)) { mt = t; break }
        mediaRecorder = new MediaRecorder(s, { mimeType: mt });
        audioChunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data) };
        mediaRecorder.onstop = async () => {
            s.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            await processAudio(blob);
        };
        mediaRecorder.start(1000);
        isRecording = true;
        micBtn.classList.add("recording"); micContainer.classList.add("recording");
        recordStatus.textContent = "Recording... Tap to stop";
        recordTimer.classList.add("active"); seconds = 0; updateTimer();
        timerInterval = setInterval(() => { seconds++; updateTimer() }, 1000);
        audioContext = new (window.AudioContext || window.webkitAudioContext);
        analyser = audioContext.createAnalyser();
        audioContext.createMediaStreamSource(s).connect(analyser);
        analyser.fftSize = 256; drawWave();
    } catch (e) { if (recordStatus) recordStatus.textContent = "Microphone access denied"; console.error(e) }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    if (micBtn) micBtn.classList.remove("recording");
    if (micContainer) micContainer.classList.remove("recording");
    if (recordStatus) recordStatus.textContent = "Processing audio...";
    if (recordTimer) recordTimer.classList.remove("active");
    clearInterval(timerInterval);
    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (audioContext) { audioContext.close(); }
    audioContext = null;
    analyser = null;
}

function updateTimer() {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0"), s = String(seconds % 60).padStart(2, "0");
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
            const bh = data[i] / 255 * h * .8;
            const g = ctx.createLinearGradient(0, h, 0, h - bh);
            g.addColorStop(0, "rgba(14,165,233,.3)"); g.addColorStop(1, "rgba(6,182,212,.8)");
            ctx.fillStyle = g; ctx.fillRect(x, h - bh, bw - 1, bh); x += bw;
        }
    })();
}

// Generate
on(transcriptInput, "input", checkGenerateBtn);
on(generateBtn, "click", () => {
    if (!transcriptInput) return;
    const t = transcriptInput.value.trim();
    if (t) { showProcessing(); generateFromTranscript(t) }
});

async function processAudio(blob) {
    console.log("processAudio triggered (MediScribe version)");
    recordStatus.textContent = "Transcribing voice...";
    try {
        const fd = new FormData; fd.append("audio", blob, "recording.webm");
        const r = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
        if (!r.ok) { let d = `HTTP ${r.status}`; try { const j = await r.json(); d = j.detail || JSON.stringify(j) } catch { try { d = await r.text() } catch { } } throw new Error(`Transcription failed: ${d}`) }
        const t = await r.json();
        console.log("Transcription successful:", t.text?.substring(0, 30));

        if (liveTranscript) liveTranscript.style.display = "block";
        if (liveTranscriptText) liveTranscriptText.textContent = t.text || "";
        if (transcriptInput) transcriptInput.value = t.text || "";

        checkGenerateBtn();
        recordStatus.textContent = "Transcription complete. Review and click Generate.";
        console.log("WAITING FOR USER (MediScribe version)...");

        hideProcessing();
        // await generateFromTranscript(t.text || "");
    } catch (e) { hideProcessing(); recordStatus.textContent = "Error: " + e.message; console.error(e) }
}

async function generateFromTranscript(transcript) {
    try {
        simulateSteps();
        // Build FormData with transcript + images
        const fd = new FormData();
        fd.append("transcript", transcript);
        fd.append("patient_name", patientName.value || "");
        if (patientAge.value) fd.append("patient_age", patientAge.value);
        if (patientSex.value) fd.append("patient_sex", patientSex.value);
        // Add uploaded images
        uploadedFiles.forEach(f => {
            if (f) fd.append("images", f, f.name);
        });

        const r = await fetch(`${API}/api/generate`, { method: "POST", body: fd });
        if (!r.ok) { let d = `HTTP ${r.status}`; try { const j = await r.json(); d = j.detail || JSON.stringify(j) } catch { try { d = await r.text() } catch { } } throw new Error(`Generation failed: ${d}`) }
        currentResult = await r.json();
        hideProcessing();
        renderResults(currentResult);
        renderAlerts(currentResult);
        renderImaging(currentResult);
        loadHistory();
        switchToTab("results");
        recordStatus.textContent = "Tap to start recording";
        // Clear uploaded images
        uploadedFiles = [];
        imagePreviewGrid.innerHTML = "";
    } catch (e) { hideProcessing(); recordStatus.textContent = "Error: " + e.message; console.error(e) }
}

function showProcessing() { if (processingOverlay) processingOverlay.classList.add("active") }
function hideProcessing() { if (processingOverlay) processingOverlay.classList.remove("active"); $$(".processing-step").forEach(s => { s.classList.remove("active", "done") }) }
function activateStep(n) {
    for (let i = 1; i <= 7; i++) {
        const s = $(`#step${i}`); if (!s) continue;
        if (i < n) { s.classList.remove("active"); s.classList.add("done") }
        else if (i === n) { s.classList.add("active"); s.classList.remove("done") }
        else s.classList.remove("active", "done");
    }
}
function simulateSteps() {
    activateStep(1);
    setTimeout(() => activateStep(2), 2000); setTimeout(() => activateStep(3), 5000);
    setTimeout(() => activateStep(4), 7000); setTimeout(() => activateStep(5), 9000);
    setTimeout(() => activateStep(6), 11000); setTimeout(() => activateStep(7), 13000);
}

// Render results
function renderResults(d) {
    const s = d.soap_note || {}, e = d.entities || {}, icd = d.icd10_codes || [];
    const name = d.patient_name ? `<span style="color:var(--accent)">${d.patient_name}</span> - ` : "";
    let h = `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <h2 style="font-size:18px;font-weight:700">&#x1F4CB; ${name}Clinical Documentation</h2>
        <span class="time-badge">&#x26A1; ${(d.processing_time_ms / 1000).toFixed(1)}s</span>
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
            ? `<div class="icd-list">${icd.map(c => `<div class="icd-chip"><span class="icd-code">${c.code}</span><span class="icd-desc">${c.description}</span><span class="icd-conf">${Math.round((c.confidence || 0) * 100)}%</span></div>`).join("")}</div>`
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
    const formatted = (text || "No data").replace(/\n/g, "<br>");
    return `<div class="soap-section"><div class="soap-label ${l.toLowerCase()}">${l} &#x2014; ${label}</div><div class="soap-text">${formatted}</div></div>`;
}
function entGrp(title, items) {
    if (!items || !items.length) return "";
    return `<div class="entity-group"><h4>${title}</h4><div class="entity-tags">${items.map(i => `<span class="entity-tag">${i}</span>`).join("")}</div></div>`;
}

// Render imaging
function renderImaging(d) {
    const suggestions = d.imaging_suggestions || [];
    const analyses = d.image_analyses || [];
    const total = suggestions.length + analyses.length;
    if (imagingBadge) {
        imagingBadge.textContent = total;
        imagingBadge.classList.toggle("visible", total > 0);
    }

    if (total === 0) {
        const empty = $("#imagingContent");
        if (empty) empty.innerHTML = '<div class="empty-state"><div class="icon">&#x1F3E5;</div><h3>No Imaging Data</h3><p>Upload medical images or generate notes to see imaging suggestions</p></div>';
        return;
    }

    let h = '';

    // Image analyses
    if (analyses.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F4F7; Uploaded Image Analysis (${analyses.length})</div>
        <div class="imaging-grid">${analyses.map(a => `
            <div class="image-analysis-card">
                ${a.filename ? `<img class="analysis-image" src="/uploads/${a.filename}" alt="${a.filename}">` : ""}
                <div class="analysis-body">
                    <span class="modality-badge ${getModalityClass(a.image_type)}">${a.image_type || "Unknown"}</span>
                    <h4>${a.body_part || "Medical Image"}</h4>
                    <div class="field-label">Findings</div>
                    <div class="field-value">${a.findings || "N/A"}</div>
                    <div class="field-label">Impression</div>
                    <div class="field-value">${a.impression || "N/A"}</div>
                    ${(a.abnormalities || []).length ? `<div class="field-label">Abnormalities</div><div>${a.abnormalities.map(x => `<span class="abnormality-tag">${x}</span>`).join("")}</div>` : ""}
                    ${a.recommendations ? `<div class="field-label">Recommendations</div><div class="field-value">${a.recommendations}</div>` : ""}
                </div>
            </div>`).join("")}</div></div>`;
    }

    // Imaging suggestions
    if (suggestions.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F52C; Recommended Imaging Studies (${suggestions.length})</div>
        <div class="imaging-grid">${suggestions.map(s => `
            <div class="imaging-card">
                <span class="modality-badge ${getModalityClass(s.modality)}">${s.modality || "Imaging"}</span>
                <span class="urgency-badge ${(s.urgency || 'routine').toLowerCase()}">${s.urgency || "routine"}</span>
                <h4>${s.body_region || "Unspecified Region"}</h4>
                <div class="field-label">Indication</div>
                <div class="field-value">${s.indication || "N/A"}</div>
                <div class="field-label">Contrast</div>
                <div class="field-value">${s.contrast || "N/A"}</div>
                ${s.notes ? `<div class="field-label">Notes</div><div class="field-value">${s.notes}</div>` : ""}
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

// Render alerts
function renderAlerts(d) {
    const alerts = d.clinical_alerts || [];
    const cnt = alerts.filter(a => a.severity === "critical" || a.severity === "warning").length;
    if (alertsBadge) {
        alertsBadge.textContent = cnt;
        alertsBadge.classList.toggle("visible", cnt > 0);
    }
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
        <div class="card-body">${alerts.map(a => `<div class="alert-item ${a.severity}">
            <span class="alert-icon">${icons[a.severity] || "&#x2753;"}</span>
            <div class="alert-content">
                <span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">${tl[a.alert_type] || a.alert_type}</span>
                <h4>${a.title}</h4><p>${a.description}</p>
                ${a.recommendation ? `<div class="recommendation">&#x1F4A1; ${a.recommendation}</div>` : ""}
            </div>
        </div>`).join("")}</div></div>`;
}

// History
async function loadHistory() {
    try { const r = await fetch(`${API}/api/encounters`); const enc = await r.json(); renderHistory(enc) } catch (e) { console.error(e) }
}

function renderHistory(enc) {
    const l = $("#historyList");
    if (!l) return;
    if (!enc || !enc.length) {
        l.innerHTML = '<div class="empty-state"><div class="icon">&#x1F4C2;</div><h3>No Encounters</h3><p>Processed encounters stored locally</p></div>';
        return;
    }
    // Store for detail view
    window._encounters = enc;
    l.innerHTML = enc.map((e, i) => {
        const d = new Date(e.timestamp);
        const ac = (e.clinical_alerts || []).length;
        const ic = (e.image_analyses || []).length;
        const cc = e.patient_name || (e.entities?.chief_complaint) || ((e.transcript || "").substring(0, 60) + "...");
        return `<div class="history-item" onclick="viewEncounterDetail(${i})">
            <div class="history-meta"><h4>${cc}</h4>
            ${e.patient_name ? `<p style="color:var(--accent);font-size:11px;margin-bottom:2px">${e.patient_name}</p>` : ""}
            <p>${d.toLocaleDateString()} ${d.toLocaleTimeString()}</p></div>
            <div class="history-badges">
                ${ic > 0 ? `<span class="history-badge images">&#x1F4F7; ${ic}</span>` : ""}
                ${ac > 0 ? `<span class="history-badge alerts">&#x26A0;&#xFE0F; ${ac}</span>` : ""}
                <span class="history-badge time">&#x26A1; ${(e.processing_time_ms / 1000).toFixed(1)}s</span>
                <button class="delete-btn" onclick="event.stopPropagation();deleteEnc('${e.id}')">&#x1F5D1;&#xFE0F;</button>
            </div>
        </div>`;
    }).join("");
}

// Full detail view
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

    let h = `<button class="detail-back" onclick="switchToTab('history')">&#x2190; Back to History</button>
    <div class="detail-header">
        <div>
            <div class="detail-patient">${e.patient_name || "Unnamed Patient"}</div>
            <div style="color:var(--text-muted);font-size:13px">${d.toLocaleDateString()} ${d.toLocaleTimeString()} &#x2022; ${(e.processing_time_ms / 1000).toFixed(1)}s processing</div>
        </div>
    </div>`;

    // Transcript
    h += `<div class="detail-section"><div class="detail-section-title">&#x1F399;&#xFE0F; Original Transcript</div>
        <div class="card"><div class="card-body" style="font-size:13px;line-height:1.8;color:var(--text-secondary)">${e.transcript || "N/A"}</div></div></div>`;

    // SOAP
    h += `<div class="detail-section"><div class="detail-section-title">&#x1F4DD; SOAP Note</div>
        <div class="card"><div class="card-body">
            ${soapSec("S", "Subjective", s.subjective)}
            ${soapSec("O", "Objective", s.objective)}
            ${soapSec("A", "Assessment", s.assessment)}
            ${soapSec("P", "Plan", s.plan)}
        </div></div></div>`;

    // Entities
    const entHtml = entGrp("Chief Complaint", ent.chief_complaint ? [ent.chief_complaint] : []) +
        entGrp("Symptoms", ent.symptoms) + entGrp("Medications", ent.medications) +
        entGrp("Allergies", ent.allergies) + entGrp("Vitals", Object.entries(ent.vitals || {}).map(([k, v]) => `${k}: ${v}`)) +
        entGrp("Medical History", ent.medical_history);
    if (entHtml) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F50D; Extracted Entities</div>
            <div class="card"><div class="card-body">${entHtml}</div></div></div>`;
    }

    // ICD-10
    if (icd.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F3F7;&#xFE0F; ICD-10 Codes</div>
            <div class="card"><div class="card-body"><div class="icd-list">${icd.map(c => `<div class="icd-chip"><span class="icd-code">${c.code}</span><span class="icd-desc">${c.description}</span><span class="icd-conf">${Math.round((c.confidence || 0) * 100)}%</span></div>`).join("")}</div></div></div></div>`;
    }

    // Image analyses
    if (imgs.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F4F7; Medical Image Analysis (${imgs.length})</div>
        <div class="imaging-grid">${imgs.map(a => `
            <div class="image-analysis-card">
                ${a.filename ? `<img class="analysis-image" src="/uploads/${a.filename}" alt="${a.filename}">` : ""}
                <div class="analysis-body">
                    <span class="modality-badge ${getModalityClass(a.image_type)}">${a.image_type || "Unknown"}</span>
                    <h4>${a.body_part || "Medical Image"}</h4>
                    <div class="field-label">Findings</div><div class="field-value">${a.findings || "N/A"}</div>
                    <div class="field-label">Impression</div><div class="field-value">${a.impression || "N/A"}</div>
                    ${(a.abnormalities || []).length ? `<div class="field-label">Abnormalities</div><div>${a.abnormalities.map(x => `<span class="abnormality-tag">${x}</span>`).join("")}</div>` : ""}
                    ${a.recommendations ? `<div class="field-label">Recommendations</div><div class="field-value">${a.recommendations}</div>` : ""}
                </div>
            </div>`).join("")}</div></div>`;
    }

    // Imaging suggestions
    if (suggestions.length) {
        h += `<div class="detail-section"><div class="detail-section-title">&#x1F52C; Recommended Imaging (${suggestions.length})</div>
        <div class="imaging-grid">${suggestions.map(sg => `
            <div class="imaging-card">
                <span class="modality-badge ${getModalityClass(sg.modality)}">${sg.modality || "Imaging"}</span>
                <span class="urgency-badge ${(sg.urgency || 'routine').toLowerCase()}">${sg.urgency || "routine"}</span>
                <h4>${sg.body_region || ""}</h4>
                <div class="field-label">Indication</div><div class="field-value">${sg.indication || "N/A"}</div>
                ${sg.contrast ? `<div class="field-label">Contrast</div><div class="field-value">${sg.contrast}</div>` : ""}
                ${sg.notes ? `<div class="field-label">Notes</div><div class="field-value">${sg.notes}</div>` : ""}
            </div>`).join("")}</div></div>`;
    }

    // Alerts
    if (alerts.length) {
        const icons = { critical: "&#x1F6A8;", warning: "&#x26A0;&#xFE0F;", info: "&#x2139;&#xFE0F;" };
        const tl = { drug_interaction: "&#x1F48A; Drug", red_flag: "&#x1F6A9; Red Flag", guideline: "&#x1F4CB; Guideline", screening: "&#x1F52C; Screening", system: "&#x2699;&#xFE0F; System" };
        h += `<div class="detail-section"><div class="detail-section-title">&#x26A0;&#xFE0F; Clinical Alerts (${alerts.length})</div>
        <div class="card"><div class="card-body">${alerts.map(a => `<div class="alert-item ${a.severity}">
            <span class="alert-icon">${icons[a.severity] || ""}</span>
            <div class="alert-content"><span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">${tl[a.alert_type] || a.alert_type}</span>
            <h4>${a.title}</h4><p>${a.description}</p>
            ${a.recommendation ? `<div class="recommendation">&#x1F4A1; ${a.recommendation}</div>` : ""}</div>
        </div>`).join("")}</div></div></div>`;
    }

    const detailContent = $("#detailContent");
    if (detailContent) detailContent.innerHTML = h;
    switchToTab("detail");
}

async function deleteEnc(id) {
    try { await fetch(`${API}/api/encounters/${id}`, { method: "DELETE" }); loadHistory() } catch (e) { console.error(e) }
}

// Health check
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

checkGenerateBtn();
checkHealth();
setInterval(checkHealth, 15000);
if ($("#historyList")) loadHistory();
