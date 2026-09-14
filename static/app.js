/**
 * COFFEE LEAF DISEASE DETECTION & TREATMENT — Frontend JavaScript
 * ===============================================================
 * Tương thích chuẩn RESTful coung21/cf-api:
 * - Camera API, Geolocation API, File Drag & Drop
 * - Gọi POST /predictor/predict
 * - Hiển thị Phác đồ điều trị (Solution) & Nguyên nhân bệnh (Cause) từ data.json
 * - Tích hợp xem Lịch sử chẩn đoán (GET /histories/{user_id})
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

    // History elements
    const btnViewHistoryHero = $('#btn-view-history-hero');
    const btnViewHistory     = $('#btn-view-history');
    const historySection     = $('#history-section');
    const btnCloseHistory    = $('#btn-close-history');
    const historyGrid        = $('#history-grid');
    const gpsText            = $('#gps-text');
    const btnRefreshGps      = $('#btn-refresh-gps');

    // Validation Banner
    const valBanner          = $('#validation-banner');
    const valBannerTitle     = $('#val-banner-title');
    const valBannerMsg       = $('#val-banner-msg');
    const btnCloseVal        = $('#btn-close-val');

    // State
    let selectedFile = null;
    let cameraStream = null;
    let facingMode = 'environment';
    let userCoordinates = "14.0583,108.2772"; // Mặc định Tây Nguyên

    // Disease icons mapping
    const diseaseIcons = {
        'healthy':    '🌿',
        'rust':       '🟠',
        'phoma':      '🔴',
        'miner':      '🟤',
        'corticium':  '⬛',
        'mealy':      '⚪',
        'nematode':   '🟡'
    };

    const diseaseColors = {
        'healthy':    '#52b788',
        'rust':       '#e76f51',
        'phoma':      '#9b2226',
        'miner':      '#bb3e03',
        'corticium':  '#6d4c41',
        'mealy':      '#adb5bd',
        'nematode':   '#e9c46a'
    };

    // =========================================================================
    // GEOLOCATION INITIALIZATION (High Accuracy + Refresh)
    // =========================================================================
    function fetchGeolocation(isManual = false) {
        if (!('geolocation' in navigator)) {
            if (gpsText) gpsText.textContent = `Tọa độ vườn: 14.0583° N, 108.2772° E (Trình duyệt không hỗ trợ GPS)`;
            return;
        }

        if (btnRefreshGps) btnRefreshGps.classList.add('loading');
        if (gpsText) gpsText.textContent = `Đang định vị GPS độ chính xác cao...`;

        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = Number(pos.coords.latitude).toFixed(6);
                const lon = Number(pos.coords.longitude).toFixed(6);
                const acc = Math.round(pos.coords.accuracy || 0);
                userCoordinates = `${lat},${lon}`;
                if (gpsText) {
                    gpsText.textContent = `Tọa độ vườn: ${lat}° N, ${lon}° E (Sai số ±${acc}m)`;
                }
                if (btnRefreshGps) btnRefreshGps.classList.remove('loading');
                if (isManual) {
                    showNotification(`Đã cập nhật vị trí GPS: ${lat}, ${lon} (±${acc}m)`, 'success');
                }
            },
            (err) => {
                userCoordinates = "14.0583,108.2772";
                if (gpsText) {
                    gpsText.textContent = `Tọa độ vườn: 14.0583° N, 108.2772° E (Tây Nguyên - Mặc định)`;
                }
                if (btnRefreshGps) btnRefreshGps.classList.remove('loading');
                if (isManual) {
                    showNotification('Không thể lấy vị trí: ' + (err.message || 'Quá thời gian'), 'error');
                }
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    }

    // Khởi tạo GPS lần đầu
    fetchGeolocation(false);

    if (btnRefreshGps) {
        btnRefreshGps.addEventListener('click', (e) => {
            e.preventDefault();
            fetchGeolocation(true);
        });
    }

    // =========================================================================
    // VALIDATION WARNING BANNER HELPERS
    // =========================================================================
    function showValidationWarning(title, msg) {
        if (valBanner && valBannerMsg) {
            if (valBannerTitle) valBannerTitle.textContent = title || 'Ảnh không đạt tiêu chuẩn phân tích';
            valBannerMsg.textContent = msg;
            valBanner.style.display = 'flex';
            valBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function hideValidationWarning() {
        if (valBanner) {
            valBanner.style.display = 'none';
        }
    }

    if (btnCloseVal) {
        btnCloseVal.addEventListener('click', () => {
            hideValidationWarning();
        });
    }

    // =========================================================================
    // UPLOAD: Drag & Drop + Click
    // =========================================================================
    uploadArea.addEventListener('click', (e) => {
        if (e.target.closest('#btn-clear')) return;
        if (uploadPreview.style.display !== 'none') return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

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

    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelection();
    });

    function handleFile(file) {
        if (!file.type.match(/^image\/(jpeg|jpg|png|webp)$/)) {
            showNotification('Chỉ hỗ trợ file ảnh JPG, JPEG, PNG, WEBP', 'error');
            return;
        }

        hideValidationWarning();
        selectedFile = file;

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
        hideValidationWarning();
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
    // ANALYZE (GỌI ENDPOINT /predictor/predict CHUẨN cf-api)
    // =========================================================================
    btnAnalyze.addEventListener('click', analyze);
    btnNew.addEventListener('click', resetToUpload);

    async function analyze() {
        if (!selectedFile) return;

        hideValidationWarning();
        actionButtons.style.display = 'none';
        uploadArea.style.display = 'none';
        resultsEl.style.display = 'none';
        if (historySection) historySection.style.display = 'none';
        loadingEl.style.display = 'block';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('user_id', 'default_user');
            formData.append('croods', userCoordinates);
            formData.append('coords', userCoordinates);

            // Gọi endpoint chuẩn hóa /predictor/predict
            const response = await fetch('/predictor/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                if (errData.error_code && errData.message) {
                    loadingEl.style.display = 'none';
                    uploadArea.style.display = 'flex';
                    actionButtons.style.display = 'flex';
                    showValidationWarning(`⚠️ Không đạt tiêu chuẩn: ${errData.error_code}`, errData.message);
                    showNotification(errData.message, 'error');
                    return;
                }
                throw new Error(errData.detail || errData.message || `Lỗi server (${response.status})`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            showNotification('Lỗi phân tích: ' + err.message, 'error');
            resetToUpload();
        }
    }

    function resetToUpload() {
        hideValidationWarning();
        clearSelection();
        loadingEl.style.display = 'none';
        resultsEl.style.display = 'none';
        uploadArea.style.display = 'flex';
        actionButtons.style.display = 'flex';
        document.getElementById('diagnose').scrollIntoView({ behavior: 'smooth' });
    }

    // =========================================================================
    // RENDER RESULTS (HIỂN THỊ KẾT QUẢ, NGUYÊN NHÂN & PHÁC ĐỒ ĐIỀU TRỊ)
    // =========================================================================
    function renderResults(data) {
        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';

        const resObj = data.result || {};
        const conf = data.confidence || 0.0;
        const details = data.details || {};
        const big = details.big_branch || {};
        const small = details.small_branch || {};

        const diseaseKey = (resObj.name || '').toLowerCase();
        const icon = diseaseIcons[diseaseKey] || '🍃';
        const color = diseaseColors[diseaseKey] || '#16a34a';

        // ---- Main Result Card ----
        $('#result-icon').textContent = icon;
        $('#result-class').textContent = (resObj.name || '').toUpperCase();
        $('#result-class-vi').textContent = resObj.disease || resObj.name || 'Không xác định';
        $('#result-confidence').textContent = (conf * 100).toFixed(1) + '%';

        // Method badge
        const methodText = details.method ? details.method.split('(')[0].trim() : 'Scale-Aware Ensemble';
        $('#result-method').textContent = methodText;

        // Confidence bar animation
        const confFill = $('#result-conf-fill');
        confFill.style.width = '0%';
        requestAnimationFrame(() => {
            confFill.style.width = (conf * 100) + '%';
        });

        const resultCard = $('#result-card');
        resultCard.style.borderLeftColor = color;

        // ---- Treatment: Cause & Solutions ----
        const causeEl = $('#result-cause');
        if (causeEl) {
            causeEl.textContent = resObj.cause || 'Không có mô tả nguyên nhân cụ thể.';
        }

        const solutionsList = $('#solutions-list');
        if (solutionsList) {
            solutionsList.innerHTML = '';
            const solutions = Array.isArray(resObj.solution) ? resObj.solution : [resObj.solution].filter(Boolean);

            if (solutions.length > 0) {
                solutions.forEach((solText, idx) => {
                    const li = document.createElement('li');
                    li.className = 'solution-item';
                    li.innerHTML = `
                        <span class="solution-num">${idx + 1}</span>
                        <div class="solution-content">${solText}</div>
                    `;
                    solutionsList.appendChild(li);
                });
            } else {
                solutionsList.innerHTML = '<li class="solution-empty">Chưa có phác đồ điều trị chi tiết cho bệnh này.</li>';
            }
        }

        // ---- Images ----
        const originalSrc = data.original_image ? 'data:image/jpeg;base64,' + data.original_image : data.image_url;
        const annotatedSrc = data.annotated_image ? 'data:image/jpeg;base64,' + data.annotated_image : data.image_url;
        $('#result-original').src = originalSrc;
        $('#result-annotated').src = annotatedSrc;

        // ---- Detection Count ----
        const numBoxes = details.num_boxes || 0;
        $('#detection-count').textContent = numBoxes + ' vùng tổn thương phát hiện';

        // ---- Big Branch ----
        if (big.class) {
            $('#big-result').textContent = `${big.class} (${(big.confidence * 100).toFixed(1)}%)`;
            if (big.all_probs) renderProbBars('big-probs', big.all_probs, big.class);
        }

        // ---- Small Branch ----
        if (small.class) {
            $('#small-result').textContent = `${small.class} (${(small.confidence * 100).toFixed(1)}%)`;
            if (small.all_probs) renderProbBars('small-probs', small.all_probs, small.class);
        }

        // Small box predictions
        const boxesContainer = $('#small-boxes');
        boxesContainer.innerHTML = '';
        if (small.box_predictions && small.box_predictions.length > 0) {
            const title = document.createElement('div');
            title.style.cssText = 'font-size:0.85rem;font-weight:700;color:#334155;margin-bottom:8px;';
            title.textContent = `${small.box_predictions.length} vết bệnh soi vi mô:`;
            boxesContainer.appendChild(title);

            small.box_predictions.forEach((bp, idx) => {
                const row = document.createElement('div');
                row.className = 'box-pred';
                row.innerHTML = `
                    <span class="box-pred__label">Vết #${idx + 1}:</span>
                    <span><b>${bp.pred_class}</b> — ${(bp.confidence * 100).toFixed(1)}%</span>
                `;
                boxesContainer.appendChild(row);
            });
        }

        document.querySelectorAll('.branch__body').forEach(el => el.classList.remove('open'));
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderProbBars(containerId, probs, topClass) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([cls, prob]) => {
            const isTop = cls.toLowerCase() === (topClass || '').toLowerCase();
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
    // HISTORY SECTION (GET /histories/default_user & GET /histories/{id}/detail)
    // =========================================================================
    if (btnViewHistoryHero) btnViewHistoryHero.addEventListener('click', toggleHistory);
    if (btnViewHistory) btnViewHistory.addEventListener('click', toggleHistory);
    if (btnCloseHistory) btnCloseHistory.addEventListener('click', () => { historySection.style.display = 'none'; });

    async function toggleHistory() {
        if (historySection.style.display === 'block') {
            historySection.style.display = 'none';
            return;
        }

        historySection.style.display = 'block';
        historyGrid.innerHTML = '<p class="history-loading">⏳ Đang nạp lịch sử chẩn đoán...</p>';
        historySection.scrollIntoView({ behavior: 'smooth' });

        try {
            const resp = await fetch('/histories/default_user');
            if (!resp.ok) throw new Error('Không thể nạp lịch sử');
            const histories = await resp.json();

            if (!histories || histories.length === 0) {
                historyGrid.innerHTML = '<p class="history-empty">🍃 Chưa có lịch sử chẩn đoán nào được lưu.</p>';
                return;
            }

            historyGrid.innerHTML = '';
            histories.forEach(item => {
                const card = document.createElement('div');
                card.className = 'history-card';
                const dateStr = item.created_at ? new Date(item.created_at).toLocaleString('vi-VN') : 'Vừa xong';
                const diseaseDisplay = item.disease_name || item.result || 'Bệnh lá';
                const confPercent = ((item.confidence || 0) * 100).toFixed(1) + '%';
                const croodsText = item.croods ? `${item.croods.lat}, ${item.croods.long}` : 'Chưa có tọa độ';

                card.innerHTML = `
                    <div class="history-card__thumb">
                        <img src="${item.image_url}" alt="Ảnh chẩn đoán" onerror="this.src='/static/style.css'">
                    </div>
                    <div class="history-card__body">
                        <div class="history-card__disease">${diseaseDisplay}</div>
                        <div class="history-card__conf">Độ tin cậy: <strong>${confPercent}</strong></div>
                        <div class="history-card__date">🕒 ${dateStr}</div>
                        <div class="history-card__gps">📍 ${croodsText}</div>
                    </div>
                `;
                card.addEventListener('click', () => viewHistoryDetail(item.id));
                historyGrid.appendChild(card);
            });

        } catch (err) {
            historyGrid.innerHTML = `<p class="history-error">❌ Lỗi: ${err.message}</p>`;
        }
    }

    async function viewHistoryDetail(historyId) {
        try {
            const resp = await fetch(`/histories/${historyId}/detail`);
            if (!resp.ok) throw new Error('Không lấy được chi tiết');
            const data = await resp.json();

            // Render lại màn hình kết quả với bản ghi lịch sử
            renderResults({
                result: data.result,
                confidence: data.confidence,
                image_url: data.image_url,
                original_image: null,
                annotated_image: null,
                details: {
                    pred_class: data.result ? data.result.name : 'Unknown',
                    method: 'Lịch sử lưu trữ',
                    num_boxes: 0,
                    big_branch: {},
                    small_branch: {}
                }
            });
            $('#result-original').src = data.image_url;
            $('#result-annotated').src = data.image_url;
            historySection.style.display = 'none';

        } catch (err) {
            showNotification('Lỗi xem chi tiết: ' + err.message, 'error');
        }
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
    // NOTIFICATION TOAST
    // =========================================================================
    function showNotification(msg, type = 'info') {
        const existing = $('.notification');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = `notification notification--${type}`;
        toast.textContent = msg;
        document.body.appendChild(toast);

        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

})();
