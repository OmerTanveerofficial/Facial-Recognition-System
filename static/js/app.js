let cameraActive = false;
let statusPollInterval = null;
let registrationCaptures = [];
let regCameraActive = false;

function navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');

    const navBtn = document.querySelector(`.nav-btn[data-page="${page}"]`);
    if (navBtn) navBtn.classList.add('active');

    if (page === 'gallery') loadGallery();
    if (page === 'settings') loadStatus();
    if (page === 'register') resetRegistration();
}

function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        startCamera();
    }
}

function startCamera() {
    const stream = document.getElementById('video-stream');
    const placeholder = document.getElementById('camera-placeholder');
    const btn = document.getElementById('btn-toggle-camera');

    stream.src = '/video_feed';
    stream.style.display = 'block';
    placeholder.style.display = 'none';

    btn.innerHTML = '&#9632; Stop Camera';
    btn.classList.remove('btn-success');
    btn.classList.add('btn-danger');

    cameraActive = true;
    updateStatusBar(true);

    statusPollInterval = setInterval(pollStatus, 500);

    showToast('Camera started', 'success');
}

function stopCamera() {
    const stream = document.getElementById('video-stream');
    const placeholder = document.getElementById('camera-placeholder');
    const btn = document.getElementById('btn-toggle-camera');

    stream.src = '';
    stream.style.display = 'none';
    placeholder.style.display = 'block';

    btn.innerHTML = '&#9654; Start Camera';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-success');

    cameraActive = false;
    updateStatusBar(false);

    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }

    fetch('/api/camera/stop', { method: 'POST' });

    document.getElementById('face-cards-list').innerHTML =
        '<div class="empty-state" id="no-faces-msg"><p>No faces detected</p></div>';

    showToast('Camera stopped', 'info');
}

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('stat-faces').textContent = data.face_count;
        const known = data.faces.filter(f => f.is_known).length;
        const unknown = data.face_count - known;
        document.getElementById('stat-known').textContent = known;
        document.getElementById('stat-unknown').textContent = unknown;
        document.getElementById('stat-fps').textContent = data.fps;

        const fpsEl = document.getElementById('stat-fps');
        fpsEl.className = 'stat-value ' +
            (data.fps >= 20 ? 'text-success' : data.fps >= 10 ? 'text-warning' : 'text-danger');

        document.getElementById('status-fps').textContent = `FPS: ${data.fps}`;
        document.getElementById('status-face-count').textContent = `Faces Detected: ${data.face_count}`;
        document.getElementById('status-registered').textContent = `Registered: ${data.registered_count}`;

        updateFaceCards(data.faces);

    } catch (e) {
    }
}

function updateFaceCards(faces) {
    const container = document.getElementById('face-cards-list');

    if (!faces || faces.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No faces detected</p></div>';
        return;
    }

    let html = '';
    for (const face of faces) {
        const isKnown = face.is_known;
        const conf = Math.round((face.confidence || 0) * 100);
        const confColor = conf >= 70 ? 'var(--success)' : conf >= 50 ? 'var(--warning)' : 'var(--danger)';

        html += `
        <div class="face-card ${isKnown ? 'known' : 'unknown'}">
            <div class="face-card-info">
                <div class="face-card-name" style="color: ${isKnown ? 'var(--success)' : 'var(--warning)'}">
                    ${face.name}
                </div>
                <div class="face-card-detail">
                    Match: ${conf}%
                    ${face.emotion ? `<br>${face.emotion_emoji || ''} ${face.emotion}` : ''}
                    ${face.gender ? `<br>${face.gender} &bull; Age: ${face.age_range}` : ''}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${conf}%; background: ${confColor}"></div>
                </div>
            </div>
        </div>`;
    }

    container.innerHTML = html;
}

function updateStatusBar(active) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-camera');

    if (active) {
        dot.classList.add('active');
        label.textContent = 'Camera: On';
    } else {
        dot.classList.remove('active');
        label.textContent = 'Camera: Off';
        document.getElementById('status-fps').textContent = 'FPS: --';
        document.getElementById('status-face-count').textContent = 'Faces Detected: 0';
    }
}

function resetRegistration() {
    registrationCaptures = [];
    regCameraActive = false;
    setRegStep(1);

    document.getElementById('reg-camera-placeholder').style.display = 'block';
    document.getElementById('reg-video-stream').style.display = 'none';
    document.getElementById('btn-start-registration').style.display = 'block';
    document.getElementById('btn-start-capture').style.display = 'none';
    document.getElementById('capture-thumbnails').innerHTML = '';
    document.getElementById('save-thumbnails').innerHTML = '';

    const nameInput = document.getElementById('reg-name');
    const notesInput = document.getElementById('reg-notes');
    if (nameInput) nameInput.value = '';
    if (notesInput) notesInput.value = '';
}

function setRegStep(step) {
    document.getElementById('register-step-1').style.display = step === 1 ? 'block' : 'none';
    document.getElementById('register-step-2').style.display = step === 2 ? 'block' : 'none';
    document.getElementById('register-step-3').style.display = step === 3 ? 'block' : 'none';

    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById(`step-${i}`);
        el.classList.remove('active', 'completed');
        if (i === step) el.classList.add('active');
        if (i < step) el.classList.add('completed');
    }

    for (let i = 1; i <= 2; i++) {
        const line = document.getElementById(`step-line-${i}`);
        line.classList.toggle('completed', i < step);
    }
}

async function startRegistrationCamera() {
    if (cameraActive) {
        stopCamera();
        await sleep(500);
    }

    const stream = document.getElementById('reg-video-stream');
    const placeholder = document.getElementById('reg-camera-placeholder');

    stream.src = '/video_feed';
    stream.style.display = 'block';
    placeholder.style.display = 'none';

    document.getElementById('btn-start-registration').style.display = 'none';
    document.getElementById('btn-start-capture').style.display = 'block';

    regCameraActive = true;

    setTimeout(() => {
        document.getElementById('btn-start-capture').disabled = false;
        document.getElementById('quality-text').textContent = 'Ready to capture';
        document.getElementById('quality-text').style.color = 'var(--success)';
        document.getElementById('quality-bar').style.width = '100%';
    }, 1500);

    showToast('Position your face and click Capture', 'info');
}

async function startCapture() {
    const regStream = document.getElementById('reg-video-stream');
    regStream.src = '';
    regCameraActive = false;

    await fetch('/api/camera/stop', { method: 'POST' });
    await sleep(800);

    setRegStep(2);
    registrationCaptures = [];

    const progressBar = document.getElementById('capture-progress');
    const statusText = document.getElementById('capture-status');
    const thumbContainer = document.getElementById('capture-thumbnails');
    thumbContainer.innerHTML = '';

    const targetCaptures = 5;
    let retries = 0;
    const maxRetries = 15;

    for (let i = 0; i < targetCaptures; i++) {
        statusText.textContent = `Capturing ${i + 1}/${targetCaptures}`;
        progressBar.style.width = `${((i + 1) / targetCaptures) * 100}%`;

        try {
            const res = await fetch('/api/register/capture', { method: 'POST' });
            const data = await res.json();

            if (data.error) {
                retries++;
                if (retries > maxRetries) {
                    statusText.textContent = 'Could not detect face. Please try again.';
                    showToast('Face capture failed — try better lighting', 'error');
                    return;
                }
                statusText.textContent = `${data.error} — retrying...`;
                i--;
                await sleep(800);
                continue;
            }

            registrationCaptures.push(data);
            retries = 0;

            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${data.face_image}`;
            img.className = 'capture-thumb';
            thumbContainer.appendChild(img);

            await sleep(700);

        } catch (e) {
            retries++;
            if (retries > maxRetries) {
                statusText.textContent = 'Capture failed. Please try again.';
                return;
            }
            statusText.textContent = 'Error capturing — retrying...';
            i--;
            await sleep(800);
        }
    }

    statusText.textContent = 'Capture complete!';

    setTimeout(() => showSaveStep(), 500);
}

function showSaveStep() {
    setRegStep(3);

    const thumbContainer = document.getElementById('save-thumbnails');
    thumbContainer.innerHTML = '';

    for (const capture of registrationCaptures) {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${capture.face_image}`;
        img.className = 'capture-thumb';
        thumbContainer.appendChild(img);
    }

    setTimeout(() => document.getElementById('reg-name').focus(), 200);
}

async function saveRegistration() {
    const name = document.getElementById('reg-name').value.trim();
    const notes = document.getElementById('reg-notes').value.trim();

    if (!name) {
        showToast('Please enter a name', 'warning');
        return;
    }

    if (registrationCaptures.length === 0) {
        showToast('No captures available', 'error');
        return;
    }

    try {
        const res = await fetch('/api/register/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                notes,
                captures: registrationCaptures.map(c => ({
                    face_image: c.face_image,
                    encoding: c.encoding,
                })),
            }),
        });

        const data = await res.json();

        if (data.status === 'registered') {
            showToast(`'${name}' registered successfully!`, 'success');
            resetRegistration();
            navigate('gallery');
        } else {
            showToast(data.error || 'Registration failed', 'error');
        }
    } catch (e) {
        showToast('Registration failed — network error', 'error');
    }
}

async function loadGallery() {
    try {
        const res = await fetch('/api/persons');
        const persons = await res.json();

        document.getElementById('person-count').textContent = `${persons.length} persons registered`;
        renderGallery(persons);
    } catch (e) {
        showToast('Failed to load gallery', 'error');
    }
}

async function filterGallery() {
    const query = document.getElementById('gallery-search').value.trim();

    if (!query) {
        loadGallery();
        return;
    }

    try {
        const res = await fetch(`/api/persons/search?q=${encodeURIComponent(query)}`);
        const persons = await res.json();
        document.getElementById('person-count').textContent = `${persons.length} results`;
        renderGallery(persons);
    } catch (e) {}
}

function renderGallery(persons) {
    const grid = document.getElementById('gallery-grid');

    if (!persons || persons.length === 0) {
        grid.innerHTML = `<div class="empty-state">
            <p>No registered faces yet.<br>Go to "Register Face" to add someone.</p>
        </div>`;
        return;
    }

    let html = '';
    for (const person of persons) {
        const created = person.created_at ? person.created_at.substring(0, 10) : 'Unknown';
        const thumbHtml = person.thumbnail
            ? `<img src="data:image/jpeg;base64,${person.thumbnail}" class="person-thumb" alt="${person.name}">`
            : `<div class="person-thumb-placeholder">&#128100;</div>`;

        html += `
        <div class="person-card">
            ${thumbHtml}
            <div class="person-name">${escapeHtml(person.name)}</div>
            <div class="person-meta">
                Added: ${created}<br>
                Recognized ${person.recognition_count || 0} times
            </div>
            <div class="person-actions">
                <button class="btn btn-secondary" onclick="editPerson(${person.id}, '${escapeHtml(person.name)}', '${escapeHtml(person.notes || '')}')">Edit</button>
                <button class="btn btn-danger" onclick="confirmDeletePerson(${person.id}, '${escapeHtml(person.name)}')">Delete</button>
            </div>
        </div>`;
    }

    grid.innerHTML = html;
}

function editPerson(id, name, notes) {
    const newName = prompt('Edit name:', name);
    if (newName && newName.trim()) {
        const newNotes = prompt('Edit notes:', notes);
        fetch(`/api/persons/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName.trim(), notes: (newNotes || '').trim() }),
        }).then(() => {
            showToast(`Updated '${newName.trim()}'`, 'success');
            loadGallery();
        });
    }
}

function confirmDeletePerson(id, name) {
    const overlay = document.getElementById('modal-overlay');
    const title = document.getElementById('modal-title');
    const message = document.getElementById('modal-message');
    const confirmBtn = document.getElementById('modal-confirm-btn');

    title.textContent = `Delete '${name}'?`;
    message.textContent = 'This will remove all their face data permanently.';

    confirmBtn.onclick = async () => {
        closeModal();
        await fetch(`/api/persons/${id}`, { method: 'DELETE' });
        showToast(`'${name}' deleted`, 'info');
        loadGallery();
    };

    overlay.style.display = 'flex';
}

function closeModal() {
    document.getElementById('modal-overlay').style.display = 'none';
}

function updateSliderLabel(slider, labelId, suffix, multiplier) {
    const val = multiplier ? (slider.value * multiplier).toFixed(2) : slider.value;
    document.getElementById(labelId).textContent = val + suffix;
}

async function applySettings() {
    const settings = {
        detection_confidence: document.getElementById('slider-confidence').value / 100,
        recognition_tolerance: document.getElementById('slider-tolerance').value / 100,
        analysis_interval: parseInt(document.getElementById('slider-interval').value),
        emotion_enabled: document.getElementById('toggle-emotion').checked,
        age_gender_enabled: document.getElementById('toggle-age-gender').checked,
    };

    try {
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });

        if (res.ok) {
            showToast('Settings applied', 'success');
            loadStatus();
        }
    } catch (e) {
        showToast('Failed to apply settings', 'error');
    }
}

function updateSettings() {
    applySettings();
}

async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const status = data.pipeline_status;

        const grid = document.getElementById('status-grid');
        const items = [
            { name: 'Face Detector', ready: status.detector },
            { name: 'Face Recognizer', ready: status.recognizer },
            { name: 'Emotion Analyzer', ready: status.emotion },
            { name: 'Age/Gender Estimator', ready: status.age_gender },
        ];

        let html = '';
        for (const item of items) {
            html += `
            <div class="status-item">
                <div class="status-item-left">
                    <div class="status-item-dot ${item.ready ? 'ready' : 'unavailable'}"></div>
                    <span class="status-item-label">${item.name}</span>
                </div>
                <span class="status-item-value">${item.ready ? 'Ready' : 'Not Available'}</span>
            </div>`;
        }

        html += `
        <div class="status-item">
            <div class="status-item-left">
                <div class="status-item-dot ready"></div>
                <span class="status-item-label">Registered Faces</span>
            </div>
            <span class="status-item-value" style="color: var(--accent); font-weight: 600;">
                ${status.registered_faces}
            </span>
        </div>`;

        grid.innerHTML = html;

    } catch (e) {}
}

async function downloadModels() {
    const btn = document.getElementById('btn-download-models');
    btn.disabled = true;
    btn.textContent = 'Downloading...';

    try {
        const res = await fetch('/api/models/download', { method: 'POST' });
        const data = await res.json();

        if (data.status === 'success') {
            showToast('All models downloaded!', 'success');
        } else {
            showToast('Some downloads failed', 'warning');
        }

        loadStatus();
    } catch (e) {
        showToast('Download failed', 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Download Models';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: '\u2713', error: '\u2717', warning: '\u26a0', info: '\u2139' };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span>${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }
    }, 3000);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/status').then(r => r.json()).then(data => {
        document.getElementById('status-registered').textContent =
            `Registered: ${data.registered_count}`;
    }).catch(() => {});

    document.getElementById('modal-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'modal-overlay') closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
});
