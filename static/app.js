/**
 * COFFEE LEAF DISEASE DETECTION — Frontend JavaScript
 * ====================================================
 * Camera API, Drag & Drop Upload, Fetch API, Results Rendering
 */

(function () {
    'use strict';

    // =========================================================================
    // DOM Elements
    // =========================================================================
    const $ = (sel) => document.querySelector(sel);
    const uploadArea     = $('#upload-area');
    const uploadContent  = $('#upload-content');
    const uploadPreview  = $('#upload-preview');
    const previewImage   = $('#preview-image');
    const fileInput      = $('#file-input');
    const btnClear       = $('#btn-clear');
    const btnCamera      = $('#btn-camera');
    const btnAnalyze     = $('#btn-analyze');
    const btnNew         = $('#btn-new');

    // Camera
    const cameraModal    = $('#camera-modal');
    const cameraBackdrop = $('#camera-backdrop');
    const cameraVideo    = $('#camera-video');
    const cameraCanvas   = $('#camera-canvas');
    const btnCapture     = $('#btn-capture');
    const btnCameraClose = $('#btn-camera-close');
    const btnSwitchCam   = $('#btn-switch-camera');

    // Loading & Results
    const loadingEl      = $('#loading');
    const resultsEl      = $('#results');
    const actionButtons  = $('#action-buttons');

    // State
    let selectedFile = null;
    let cameraStream = null;
    let facingMode = 'environment'; // Default: rear camera

    // Disease icons mapping
    const diseaseIcons = {
        'Healthy':    '🌿',
        'Rust':       '🟠',
        'Phoma':      '🔴',
        'Miner':      '🟤',
        'Corticium':  '⬛',
        'Mealy':      '⚪',
        'Nematode':   '🟡'
    };

    // =========================================================================
    // UPLOAD: Drag & Drop + Click
    // =========================================================================
    uploadArea.addEventListener('click', (e) => {
        // Don't trigger file input when clicking clear button or preview
        if (e.target.closest('#btn-clear')) return;
        if (uploadPreview.style.display !== 'none') return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('drag-over');
        });
    });

    uploadArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    });

    // Clear button
    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelection();
    });

    function handleFile(file) {
        // Validate type
        if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
            showNotification('Chỉ hỗ trợ file ảnh JPG, JPEG, PNG', 'error');
            return;
        }

        selectedFile = file;

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'block';
            uploadArea.style.cursor = 'default';
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelection() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadContent.style.display = 'flex';
        uploadPreview.style.display = 'none';
        uploadArea.style.cursor = 'pointer';
        btnAnalyze.disabled = true;
    }

    // =========================================================================
    // CAMERA
    // =========================================================================
    btnCamera.addEventListener('click', openCamera);
    btnCameraClose.addEventListener('click', closeCamera);
    cameraBackdrop.addEventListener('click', closeCamera);
    btnCapture.addEventListener('click', capturePhoto);
    btnSwitchCam.addEventListener('click', switchCamera);

    async function openCamera() {
        cameraModal.style.display = 'flex';
        try {
            const constraints = {
                video: {
                    facingMode: facingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 960 }
                }
            };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraVideo.srcObject = cameraStream;
        } catch (err) {
            showNotification('Không thể truy cập camera: ' + err.message, 'error');
            closeCamera();
        }
    }

    function closeCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        cameraVideo.srcObject = null;
        cameraModal.style.display = 'none';
    }

    async function switchCamera() {
        facingMode = facingMode === 'environment' ? 'user' : 'environment';
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
        }
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 960 } }
            });
            cameraVideo.srcObject = cameraStream;
        } catch (err) {
            showNotification('Không thể đổi camera: ' + err.message, 'error');
        }
    }

    function capturePhoto() {
        const video = cameraVideo;
        const canvas = cameraCanvas;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const file = new File([blob], `camera_${timestamp}.jpg`, { type: 'image/jpeg' });
            handleFile(file);
            closeCamera();
        }, 'image/jpeg', 0.92);
    }

    // =========================================================================
    // ANALYZE
    // =========================================================================
    btnAnalyze.addEventListener('click', analyze);
    btnNew.addEventListener('click', resetToUpload);

    async function analyze() {
        if (!selectedFile) return;

        // Show loading
        actionButtons.style.display = 'none';
        uploadArea.style.display = 'none';
        resultsEl.style.display = 'none';
        loadingEl.style.display = 'block';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            showNotification('Lỗi phân tích: ' + err.message, 'error');
            resetToUpload();
        }
    }

    function resetToUpload() {
        clearSelection();
        loadingEl.style.display = 'none';
        resultsEl.style.display = 'none';
        uploadArea.style.display = 'flex';
        actionButtons.style.display = 'flex';
        
        // Scroll to diagnose section
        document.getElementById('diagnose').scrollIntoView({ behavior: 'smooth' });
    }

    // =========================================================================
    // RENDER RESULTS
    // =========================================================================
    function renderResults(data) {
        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';

        const pred = data.prediction;
        const big = data.big_branch;
        const small = data.small_branch;

        // Determine disease icon
        const icon = diseaseIcons[pred.class] || '🍃';

        // ---- Main Result Card ----
        $('#result-icon').textContent = icon;
        $('#result-class').textContent = pred.class;
        $('#result-class-vi').textContent = pred.class_vi;
        $('#result-confidence').textContent = (pred.confidence * 100).toFixed(1) + '%';
        $('#result-desc').textContent = pred.description;

        // Method badge
        const methodShort = pred.method.split('(')[0].trim();
        $('#result-method').textContent = methodShort;

        // Confidence bar
        const confFill = $('#result-conf-fill');
        confFill.style.width = '0%';
        requestAnimationFrame(() => {
            confFill.style.width = (pred.confidence * 100) + '%';
        });

        // Result card color accent
        const resultCard = $('#result-card');
        resultCard.style.borderLeftColor = pred.color;

        // ---- Images ----
        $('#result-original').src = 'data:image/jpeg;base64,' + data.original_image;
        $('#result-annotated').src = 'data:image/jpeg;base64,' + data.annotated_image;

        // ---- Detection Count ----
        $('#detection-count').textContent = data.num_boxes + ' vùng bệnh';

        // ---- Big Branch ----
        $('#big-result').textContent = `${big.class} (${(big.confidence * 100).toFixed(1)}%)`;
        renderProbBars('big-probs', big.all_probs, big.class);

        // ---- Small Branch ----
        $('#small-result').textContent = `${small.class} (${(small.confidence * 100).toFixed(1)}%)`;
        renderProbBars('small-probs', small.all_probs, small.class);

        // Small box predictions
        const boxesContainer = $('#small-boxes');
        boxesContainer.innerHTML = '';
        if (small.box_predictions && small.box_predictions.length > 0) {
            const title = document.createElement('div');
            title.style.cssText = 'font-size:0.85rem;font-weight:700;color:#444;margin-bottom:8px;';
            title.textContent = `${small.box_predictions.length} nốt bệnh phân tích:`;
            boxesContainer.appendChild(title);

            small.box_predictions.forEach((bp, idx) => {
                const row = document.createElement('div');
                row.className = 'box-pred';
                row.innerHTML = `
                    <span class="box-pred__label">Nốt #${idx + 1}:</span>
                    <span>${bp.pred_class} — ${(bp.confidence * 100).toFixed(1)}%</span>
                `;
                boxesContainer.appendChild(row);
            });
        }

        // Close branch bodies by default
        document.querySelectorAll('.branch__body').forEach(el => el.classList.remove('open'));

        // Scroll to results
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderProbBars(containerId, probs, topClass) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        // Sort by probability descending
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        sorted.forEach(([cls, prob]) => {
            const isTop = cls === topClass;
            const row = document.createElement('div');
            row.className = 'prob-row' + (isTop ? ' is-top' : '');

            const pct = (prob * 100).toFixed(1);
            row.innerHTML = `
                <span class="prob-row__name">${cls}</span>
                <div class="prob-row__bar">
                    <div class="prob-row__fill" style="width: ${pct}%"></div>
                </div>
                <span class="prob-row__val">${pct}%</span>
            `;
            container.appendChild(row);
        });
    }

    // =========================================================================
    // BRANCH TOGGLE
    // =========================================================================
    document.querySelectorAll('.branch__header').forEach(header => {
        header.addEventListener('click', () => {
            const targetId = header.getAttribute('data-toggle');
            const body = document.getElementById(targetId);
            const toggle = header.querySelector('.branch__toggle');

            body.classList.toggle('open');
            toggle.style.transform = body.classList.contains('open') ? 'rotate(180deg)' : '';
        });
    });

    // =========================================================================
    // NOTIFICATION HELPER
    // =========================================================================
    function showNotification(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 24px;
            right: 24px;
            z-index: 9999;
            background: ${type === 'error' ? '#d63031' : '#2d6a4f'};
            color: #fff;
            padding: 14px 24px;
            border-radius: 12px;
            font-size: 0.92rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
            max-width: 400px;
            animation: fadeInUp 0.3s ease-out;
            cursor: pointer;
        `;
        toast.textContent = message;
        toast.addEventListener('click', () => toast.remove());
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // =========================================================================
    // SMOOTH SCROLL FOR CTA
    // =========================================================================
    const ctaButton = $('#cta-button');
    if (ctaButton) {
        ctaButton.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('diagnose').scrollIntoView({ behavior: 'smooth' });
        });
    }

})();
