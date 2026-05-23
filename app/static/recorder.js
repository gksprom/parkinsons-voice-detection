// Two-step recorder
// Step 1: αααα → /predict/vowel
// Step 2: reading → /predict/reading → final result

let audioContext = null;
let mediaStream = null;
let processor = null;
let recordedSamples = [];
let sampleRate = 44100;
let activeStep = 1;

const elements = {
    1: {
        rec: document.getElementById('rec1Btn'),
        stop: document.getElementById('stop1Btn'),
        status: document.getElementById('status1'),
        player: document.getElementById('player1'),
        result: document.getElementById('result1'),
    },
    2: {
        rec: document.getElementById('rec2Btn'),
        stop: document.getElementById('stop2Btn'),
        status: document.getElementById('status2'),
        player: document.getElementById('player2'),
    },
};

elements[1].rec.addEventListener('click', () => startRecording(1));
elements[1].stop.addEventListener('click', () => stopRecording(1));
elements[2].rec.addEventListener('click', () => startRecording(2));
elements[2].stop.addEventListener('click', () => stopRecording(2));

async function startRecording(step) {
    activeStep = step;
    const el = elements[step];
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 44100,
                echoCancellation: false,
                autoGainControl: false,    // Απενεργοποιημένο: το AGC μεταβάλλει
                                            // το amplitude και αλλοιώνει την αρμονική σταθερότητα
                noiseSuppression: false,   // Απενεργοποιημένο: το NS αφαιρεί φασματικές περιοχές,
                                            // δημιουργεί τεχνητές discontinuities
            }
        });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sampleRate = audioContext.sampleRate;
        const source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (e) => {
            recordedSamples.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        source.connect(processor);
        processor.connect(audioContext.destination);
        recordedSamples = [];
        el.rec.disabled = true;
        el.rec.classList.add('recording');
        el.stop.disabled = false;
        el.status.textContent = step === 1 ? 'Recording... εκφωνούμε "αααα"' : 'Recording... διαβάζουμε το κείμενο';
        el.status.classList.remove('error');
    } catch (err) {
        el.status.textContent = 'Error: ' + err.message;
        el.status.classList.add('error');
    }
}

async function stopRecording(step) {
    const el = elements[step];
    processor.disconnect();
    mediaStream.getTracks().forEach(t => t.stop());
    await audioContext.close();

    el.rec.disabled = false;
    el.rec.classList.remove('recording');
    el.stop.disabled = true;
    el.status.textContent = 'Processing...';

    const totalLen = recordedSamples.reduce((s, a) => s + a.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of recordedSamples) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    const wavBlob = encodeWAV(merged, sampleRate);

    // Audio playback
    if (el.player.src) URL.revokeObjectURL(el.player.src);
    el.player.src = URL.createObjectURL(wavBlob);
    el.player.style.display = 'block';

    // Download button για debugging
    let dlBtn = el.player.parentElement.querySelector('.dl-btn-' + step);
    if (!dlBtn) {
        dlBtn = document.createElement('a');
        dlBtn.className = 'dl-btn-' + step;
        dlBtn.style.cssText = 'display:inline-block; margin-top:6px; font-size:12px; color:#3498db; text-decoration:underline;';
        dlBtn.textContent = '⬇ Download WAV (για ανάλυση στο notebook)';
        el.player.parentElement.insertBefore(dlBtn, el.player.nextSibling);
    }
    dlBtn.href = URL.createObjectURL(wavBlob);
    dlBtn.download = `recording_step${step}_${Date.now()}.wav`;

    // Send to backend
    const endpoint = step === 1 ? '/predict/vowel' : '/predict/reading';
    const fd = new FormData();
    fd.append('audio', wavBlob, 'recording.wav');
    try {
        const resp = await fetch(endpoint, { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) {
            el.status.textContent = data.error;
            el.status.classList.add('error');
            return;
        }
        el.status.textContent = 'Done';
        el.status.classList.remove('error');
        if (step === 1) handleStep1Result(data);
        else handleStep2Result(data);
    } catch (err) {
        el.status.textContent = 'Error: ' + err.message;
        el.status.classList.add('error');
    }
}

function handleStep1Result(data) {
    if (data.quality) renderQuality('quality1', data.quality);
    document.getElementById('r1-uci-rf').textContent = (data.uci_rf_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-uci-svm').textContent = (data.uci_svm_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-uci').textContent = (data.uci_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-iyer-rf').textContent = (data.iyer_rf_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-iyer-svm').textContent = (data.iyer_svm_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-iyer').textContent = (data.iyer_prob * 100).toFixed(1) + '% PD';
    document.getElementById('r1-combined').innerHTML = '<b>' + (data.combined * 100).toFixed(1) + '% PD</b>';
    document.getElementById('result1').style.display = 'block';

    renderFeatures('uci-features', data.uci_top_features);
    renderFeatures('iyer-features', data.iyer_top_features);

    // Activate Step 2
    document.getElementById('step1-num').classList.remove('active');
    document.getElementById('step1-num').classList.add('done');
    document.getElementById('step2-num').classList.add('active');
    elements[2].rec.disabled = false;
    elements[2].status.textContent = 'Ready';
}

function renderFeatures(containerId, features) {
    const el = document.getElementById(containerId);
    el.innerHTML = '';
    for (const f of features) {
        const row = document.createElement('div');
        row.className = 'feat-row';

        // Bar visualization: HC mean (πράσινο) και PD mean (κόκκινο) με marker στην τιμή του user
        const range = computeRange(f);
        const userPct = pctOnRange(f.user_value, range);
        const hcPct = pctOnRange(f.hc_mean, range);
        const pdPct = pctOnRange(f.pd_mean, range);

        const shapClass = f.direction === 'PD' ? 'shap-pd' : 'shap-hc';
        const shapSign = f.shap_impact > 0 ? '+' : '';

        // closer_to indicator
        const closerLabel = f.closer_to === 'HC'
            ? '<span style="color:#27ae60">πιο κοντά σε HC</span>'
            : '<span style="color:#c0392b">πιο κοντά σε PD</span>';

        row.innerHTML = `
            <div>
                <div class="feat-name">${f.label_gr}</div>
                <div class="feat-mini">
                    🟢 HC: ${f.hc_mean} · 🔴 PD: ${f.pd_mean} · Εσύ: <b>${f.user_value}</b> · ${closerLabel}
                </div>
            </div>
            <div>
                <div class="feat-bar-wrap">
                    <div class="feat-dot-hc" style="left:${Math.max(0, Math.min(100, hcPct))}%" title="HC μέσος"></div>
                    <div class="feat-dot-pd" style="left:${Math.max(0, Math.min(100, pdPct))}%" title="PD μέσος"></div>
                    <div class="feat-marker" style="left:${Math.max(0, Math.min(100, userPct))}%" title="Η τιμή σου"></div>
                </div>
            </div>
            <div class="shap-impact ${shapClass}">${shapSign}${f.shap_impact.toFixed(3)}</div>
        `;
        el.appendChild(row);
    }

    // Legend
    const legend = document.createElement('div');
    legend.style.cssText = 'font-size: 10px; color: #7f8c8d; margin-top: 10px; line-height: 1.6;';
    legend.innerHTML = '🟢 HC μέσος · 🔴 PD μέσος · ⬛ η μετρούμενη τιμή<br>SHAP: <span class="shap-impact shap-pd">+x</span> = το μοντέλο μετατοπίζει την πρόβλεψη προς PD · <span class="shap-impact shap-hc">-x</span> = προς HC<br><i>Σημείωση: η τιμή SHAP δεν αντικατοπτρίζει απλώς την απόσταση από τον μέσο όρο, αλλά τον τρόπο που το μοντέλο χρησιμοποιεί το feature στην απόφαση. Όταν η τιμή βρίσκεται εκτός του εύρους HC και PD, το μοντέλο τείνει να την ερμηνεύσει ως PD-like.</i>';
    el.appendChild(legend);
}

function computeRange(f) {
    const vals = [f.user_value, f.hc_mean, f.pd_mean];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const padding = (max - min) * 0.3 || 1;
    return { min: min - padding, max: max + padding };
}

function pctOnRange(val, range) {
    if (range.max === range.min) return 50;
    return ((val - range.min) / (range.max - range.min)) * 100;
}

function handleStep2Result(data) {
    document.getElementById('step2-num').classList.remove('active');
    document.getElementById('step2-num').classList.add('done');

    const final = data.final;
    const pct = final * 100;

    document.getElementById('finalProb').textContent = pct.toFixed(1) + '%';
    document.getElementById('finalMarker').style.left = `calc(${pct}% - 2px)`;

    const interp = document.getElementById('finalInterp');
    const bin = data.bin;
    interp.innerHTML = `<div class="bin-label">${bin.label}</div>${bin.description}`;
    if (pct < 35) interp.className = 'interpretation interp-low';
    else if (pct < 55) interp.className = 'interpretation interp-mid';
    else interp.className = 'interpretation interp-high';

    document.getElementById('b-uci').textContent = (data.breakdown.uci_prob * 100).toFixed(1) + '%';
    document.getElementById('b-iyer').textContent = (data.breakdown.iyer_prob * 100).toFixed(1) + '%';
    document.getElementById('b-mdvr').textContent = (data.breakdown.mdvr_prob * 100).toFixed(1) + '%';
    document.getElementById('b-final').textContent = (final * 100).toFixed(1) + '%';

    // RF/SVM breakdown
    const vowel = data.breakdown;  // shorthand
    document.getElementById('b-uci-rf').textContent = ((vowel.uci_rf_prob !== undefined ? vowel.uci_rf_prob : 0) * 100).toFixed(1) + '%';
    document.getElementById('b-uci-svm').textContent = ((vowel.uci_svm_prob !== undefined ? vowel.uci_svm_prob : 0) * 100).toFixed(1) + '%';
    document.getElementById('b-iyer-rf').textContent = ((vowel.iyer_rf_prob !== undefined ? vowel.iyer_rf_prob : 0) * 100).toFixed(1) + '%';
    document.getElementById('b-iyer-svm').textContent = ((vowel.iyer_svm_prob !== undefined ? vowel.iyer_svm_prob : 0) * 100).toFixed(1) + '%';
    document.getElementById('b-mdvr-rf').textContent = ((data.mdvr_rf_prob || 0) * 100).toFixed(1) + '%';
    document.getElementById('b-mdvr-svm').textContent = ((data.mdvr_svm_prob || 0) * 100).toFixed(1) + '%';

    // Agreement indicator
    if (data.agreement !== undefined) {
        const agreementEl = document.getElementById('agreementBox');
        if (agreementEl) {
            const agreePct = (data.agreement * 100).toFixed(0);
            let agreeText, agreeClass;
            if (data.agreement > 0.8) { agreeText = 'Συμφωνία μοντέλων: ΥΨΗΛΗ'; agreeClass = 'quality-good'; }
            else if (data.agreement > 0.6) { agreeText = 'Συμφωνία μοντέλων: ΜΕΤΡΙΑ'; agreeClass = 'quality-fair'; }
            else { agreeText = 'Συμφωνία μοντέλων: ΧΑΜΗΛΗ — η πρόβλεψη είναι αναξιόπιστη'; agreeClass = 'quality-poor'; }
            agreementEl.innerHTML = `<div class="quality-box ${agreeClass}"><b>${agreeText} (${agreePct}%)</b><div class="quality-metrics">Spread μεταξύ μοντέλων: ${((1 - data.agreement) * 100).toFixed(1)}%. Όσο μεγαλύτερη η διαφωνία, τόσο πιο αβέβαιη η συνολική πρόβλεψη.</div></div>`;
        }
    }

    if (data.mdvr_top_features) {
        renderFeatures('mdvr-features', data.mdvr_top_features);
    }
    if (data.quality) renderQuality('quality2', data.quality);

    document.getElementById('finalCard').style.display = 'block';
    document.getElementById('finalCard').scrollIntoView({ behavior: 'smooth' });
}

function renderQuality(containerId, quality) {
    const el = document.getElementById(containerId);
    const cls = quality.verdict === 'good' ? 'quality-good' :
                quality.verdict === 'fair' ? 'quality-fair' : 'quality-poor';
    const icon = quality.verdict === 'good' ? '✓' :
                 quality.verdict === 'fair' ? '⚠' : '✗';
    const label = quality.verdict === 'good' ? 'Καλή ποιότητα ηχογράφησης' :
                  quality.verdict === 'fair' ? 'Μέτρια ποιότητα' :
                  'Κακή ποιότητα — η πρόβλεψη μπορεί να μην είναι αξιόπιστη';

    let warningsHtml = '';
    if (quality.warnings && quality.warnings.length > 0) {
        warningsHtml = '<ul class="quality-warnings">' +
            quality.warnings.map(w => `<li>${w.text}</li>`).join('') +
            '</ul>';
    }

    const m = quality.metrics;
    const metricsHtml = `<div class="quality-metrics">` +
        `Διάρκεια: ${m.duration}s · RMS: ${m.rms_db} dB · HNR: ${m.hnr_db} dB · Voiced: ${(m.voiced_ratio*100).toFixed(0)}%` +
        `</div>`;

    el.innerHTML = `<div class="quality-box ${cls}">` +
        `<b>${icon} Recording quality: ${quality.score}/100 — ${label}</b>` +
        warningsHtml +
        metricsHtml +
        `</div>`;
}

function encodeWAV(samples, sr) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        off += 2;
    }
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}
