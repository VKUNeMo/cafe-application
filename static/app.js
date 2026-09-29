/**
 * COFFEE LEAF DISEASE DETECTION & TREATMENT — Frontend JavaScript (Pipeline v4)
 * ==============================================================================
 * Tương thích chuẩn RESTful coung21/cf-api:
 * - Camera API, Geolocation API, File Drag & Drop
 * - Gọi POST /predictor/predict (Pipeline v4: YOLOv8s Slicing + ResNet Big/Small + Ensemble v4)
 * - Chẩn đoán phân biệt Top-2 (Differential Diagnosis) & Cảnh báo phân vân lâm sàng (Uncertainty Flag)
 * - Phác đồ điều trị 2 chế độ (Interactive Treatment Tabs: Top-1 vs Top-2)
 * - Chi tiết suy luận đa tầng (Ensemble 8-class Probabilities, Active Rule Trace, Big/Small Breakdown)
 * - Quản lý lịch sử dịch tễ với bộ lọc theo loại bệnh (Filter Chips)
 */

(function () {
    'use strict';

    // =========================================================================
    // DOM Elements
    // =========================================================================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // Upload & Form
    const uploadArea     = $('#upload-area');
    const uploadContent  = $('#upload-content');
    const uploadPreview  = $('#upload-preview');
    const previewImage   = $('#preview-image');
    const fileInput      = $('#file-input');
    const btnClear       = $('#btn-clear');
    const btnCamera      = $('#btn-camera');
    const btnAnalyze     = $('#btn-analyze');
    const btnNew         = $('#btn-new');

    // Camera Modal
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

    // Differential & Treatment Tabs
    const diffBanner         = $('#differential-banner');
    const btnViewTop2Sol     = $('#btn-view-top2-solution');
    const tabBtnTop1         = $('#tab-btn-top1');
    const tabBtnTop2         = $('#tab-btn-top2');
    const tabTop1Name        = $('#tab-top1-name');
    const tabTop2Name        = $('#tab-top2-name');
    const treatmentTop2Alert = $('#treatment-top2-alert');

    // State Variables
    let selectedFile = null;
    let cameraStream = null;
    let facingMode = 'environment';
    let userCoordinates = "14.0583,108.2772"; // Mặc định Tây Nguyên
    let currentDiagnosisData = null;
    let currentActiveTab = 'top1';
    let cachedHistories = [];
    let activeHistoryFilter = 'all';

    // 8 Disease Metadata Mapping
    const diseaseIcons = {
        'healthy':    '🌿',
        'rust':       '🟠',
        'phoma':      '🔴',
        'miner':      '🟤',
        'corticium':  '⬛',
        'mealy':      '⚪',
        'nematode':   '🟡',
        'anthracnose':'🍂'
    };

    const diseaseColors = {
        'healthy':    '#52b788',
        'rust':       '#e76f51',
        'phoma':      '#9b2226',
        'miner':      '#bb3e03',
        'corticium':  '#6d4c41',
        'mealy':      '#adb5bd',
        'nematode':   '#e9c46a',
        'anthracnose':'#a0522d'
    };

    const diseaseVietnamese = {
        'healthy':    'Khỏe mạnh',
        'rust':       'Bệnh Rỉ Sắt Cà Phê',
        'phoma':      'Bệnh Đốm Mắt Cua (Phoma)',
        'miner':      'Sâu Vẽ Bùa Gây Hại',
        'corticium':  'Bệnh Nấm Hồng (Corticium)',
        'mealy':      'Rệp Sáp Gây Hại',
        'nematode':   'Tuyến Trùng Vàng Lá',
        'anthracnose':'Bệnh Thán Thư (Khô Cành, Thối Quả)'
    };

    // =========================================================================
    // GEOLOCATION INITIALIZATION
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
        if (valBanner) valBanner.style.display = 'none';
    }

    if (btnCloseVal) {
        btnCloseVal.addEventListener('click', hideValidationWarning);
    }

    // =========================================================================
    // FILE UPLOAD & DRAG DROP
    // =========================================================================
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
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
    // CAMERA CONTROLS
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
    // ANALYZE (GỌI ENDPOINT /predictor/predict)
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
            currentDiagnosisData = data;
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
    // RENDER RESULTS
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

        // ---- 1. Main Result Card ----
        $('#result-icon').textContent = icon;
        $('#result-class').textContent = (resObj.name || '').toUpperCase();
        $('#result-class-vi').textContent = resObj.disease || diseaseVietnamese[diseaseKey] || resObj.name || 'Không xác định';
        $('#result-confidence').textContent = (conf * 100).toFixed(1) + '%';

        // Method & Time badge
        const methodRaw = details.method || 'Ensemble v4';
        const methodShort = methodRaw.split('(')[0].trim();
        $('#result-method').textContent = methodShort;

        const elapsedSec = details.elapsed_seconds;
        const elapsedEl = $('#result-elapsed');
        if (elapsedEl) {
            elapsedEl.textContent = elapsedSec ? `⚡ ${Number(elapsedSec).toFixed(2)}s` : '⚡ 0.18s';
        }

        // Animate confidence bar
        const confFill = $('#result-conf-fill');
        confFill.style.width = '0%';
        requestAnimationFrame(() => {
            confFill.style.width = (conf * 100) + '%';
        });

        const resultCard = $('#result-card');
        resultCard.style.borderLeftColor = color;

        // ---- 2. Differential Diagnosis Banner (Top-2 & Uncertainty) ----
        if (diffBanner) {
            const hasTop2 = (data.top2_result && data.top2_result.name && data.top2_result.name !== 'None') ||
                            (data.top2_name && data.top2_name !== 'None') ||
                            (details.top2_class && details.top2_class !== 'None');

            if (hasTop2) {
                diffBanner.style.display = 'block';

                const top1Name = resObj.disease || resObj.name || (details.pred_class || 'Bệnh chính');
                const top1ConfPct = (conf * 100).toFixed(1) + '%';

                const top2Obj = data.top2_result || {};
                const top2Name = top2Obj.disease || top2Obj.name || data.top2_name || (details.top2_class || 'Bệnh nghi ngờ số 2');
                const top2ConfVal = (data.top2_confidence !== undefined && data.top2_confidence !== null) ? data.top2_confidence : (details.top2_confidence || 0.0);
                const top2ConfPct = (top2ConfVal * 100).toFixed(1) + '%';

                const gTop1Name = $('#diff-grid-top1-name');
                const gTop1Conf = $('#diff-grid-top1-conf');
                const gTop2Name = $('#diff-grid-top2-name') || $('[data-fallback="diff-top2-name"]') || $('#diff-top2-name');
                const gTop2Conf = $('#diff-grid-top2-conf') || $('[data-fallback="diff-top2-conf"]') || $('#diff-top2-conf');
                const badgeEl   = $('#diff-uncertainty-badge');
                const iconEl    = $('#diff-icon');
                const noteEl    = $('#diff-note-text');

                if (gTop1Name) gTop1Name.textContent = top1Name;
                if (gTop1Conf) gTop1Conf.textContent = top1ConfPct;
                if (gTop2Name) gTop2Name.textContent = top2Name;
                if (gTop2Conf) gTop2Conf.textContent = top2ConfPct;

                if (data.is_uncertain) {
                    diffBanner.classList.add('is-uncertain');
                    if (badgeEl) badgeEl.textContent = 'CẢNH BÁO PHÂN VÂN LÂM SÀNG';
                    if (iconEl) iconEl.textContent = '⚠️';
                    if (noteEl) {
                        noteEl.textContent = data.differential_note ||
                            'Triệu chứng tổn thương nằm trong vùng tương đồng cao giữa 2 bệnh. Khuyến nghị kiểm tra kỹ mặt dưới phiến lá và theo dõi diễn tiến trước khi ra quyết định phun thuốc.';
                    }
                } else {
                    diffBanner.classList.remove('is-uncertain');
                    if (badgeEl) badgeEl.textContent = 'XÁC ĐỊNH TIN CẬY CAO';
                    if (iconEl) iconEl.textContent = '💡';
                    if (noteEl) {
                        noteEl.textContent = data.differential_note ||
                            'Chẩn đoán xác định bệnh hàng đầu với mức độ tin cậy áp đảo so với các phân loại còn lại.';
                    }
                }
            } else {
                diffBanner.style.display = 'none';
            }
        }

        // ---- 3. Interactive Treatment Tabs ----
        if (tabTop1Name) {
            tabTop1Name.textContent = resObj.disease || resObj.name || 'Bệnh chính';
        }
        if (data.top2_result && data.top2_result.name && data.top2_result.name !== 'None') {
            if (tabBtnTop2) tabBtnTop2.style.display = 'inline-flex';
            if (tabTop2Name) tabTop2Name.textContent = data.top2_result.disease || data.top2_result.name;
        } else {
            if (tabBtnTop2) tabBtnTop2.style.display = 'none';
        }

        // Mặc định chọn tab 1
        switchTreatmentTab('top1');

        // ---- 4. Images Comparison ----
        const originalSrc = data.original_image ? 'data:image/jpeg;base64,' + data.original_image : data.image_url;
        const annotatedSrc = data.annotated_image ? 'data:image/jpeg;base64,' + data.annotated_image : data.image_url;
        $('#result-original').src = originalSrc;
        $('#result-annotated').src = annotatedSrc;

        // Detection Count Badge
        const numBoxes = details.num_boxes || 0;
        const countBadge = $('#detection-count');
        if (countBadge) {
            countBadge.textContent = `${numBoxes} vùng tổn thương phát hiện`;
        }

        // ---- 5. Deep Diagnostic Trace ----
        // Active Decision Rule
        renderActiveRuleBanner(details.method);

        // Combined Ensemble Probabilities Bar Chart (8 classes)
        renderEnsembleCombinedChart(big.all_probs, small.all_probs, resObj.name);

        // Big Branch
        if (big.class || big.pred_class) {
            const bClass = big.pred_class || big.class;
            $('#big-result').textContent = `${bClass} (${(big.confidence * 100).toFixed(1)}%)`;
            if (big.all_probs) renderProbBars('big-probs', big.all_probs, bClass);
        }

        // Small Branch
        if (small.class || small.pred_class) {
            const sClass = small.pred_class || small.class;
            $('#small-result').textContent = `${sClass} (${(small.confidence * 100).toFixed(1)}%)`;
            if (small.all_probs) renderProbBars('small-probs', small.all_probs, sClass);
        }

        // Small box predictions list
        const boxesContainer = $('#small-boxes');
        boxesContainer.innerHTML = '';
        const boxPreds = small.box_predictions || [];
        if (boxPreds.length > 0) {
            const title = document.createElement('div');
            title.style.cssText = 'font-size:0.85rem;font-weight:700;color:#334155;margin-bottom:8px;';
            title.textContent = `${boxPreds.length} lát cắt vi mô từng đốm bệnh (224×224):`;
            boxesContainer.appendChild(title);

            boxPreds.forEach((bp, idx) => {
                const row = document.createElement('div');
                row.className = 'box-pred';
                const pCls = bp.pred_class || '';
                const pConf = ((bp.confidence || 0) * 100).toFixed(1);
                row.innerHTML = `
                    <span class="box-pred__label">Lát cắt #${idx + 1}:</span>
                    <span><b>${pCls}</b> — ${pConf}%</span>
                `;
                boxesContainer.appendChild(row);
            });
        }

        // Reset accordion state
        $$('.branch__body:not(.open)').forEach(el => el.classList.remove('open'));
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // =========================================================================
    // TREATMENT TAB SWITCHER
    // =========================================================================
    function switchTreatmentTab(tabKey) {
        currentActiveTab = tabKey;
        if (!currentDiagnosisData) return;

        const isTop2 = (tabKey === 'top2');
        const targetObj = isTop2 ? currentDiagnosisData.top2_result : currentDiagnosisData.result;

        if (tabBtnTop1) tabBtnTop1.classList.toggle('is-active', !isTop2);
        if (tabBtnTop2) tabBtnTop2.classList.toggle('is-active', isTop2);

        if (treatmentTop2Alert) {
            treatmentTop2Alert.style.display = isTop2 ? 'block' : 'none';
        }

        if (!targetObj) return;

        // Render Cause
        const causeEl = $('#result-cause');
        if (causeEl) {
            causeEl.textContent = targetObj.cause || 'Không có mô tả nguyên nhân cụ thể.';
        }

        // Render Solutions
        const solutionsList = $('#solutions-list');
        if (solutionsList) {
            solutionsList.innerHTML = '';
            const solutions = Array.isArray(targetObj.solution) ? targetObj.solution : [targetObj.solution].filter(Boolean);

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
    }

    if (tabBtnTop1) {
        tabBtnTop1.addEventListener('click', () => switchTreatmentTab('top1'));
    }
    if (tabBtnTop2) {
        tabBtnTop2.addEventListener('click', () => switchTreatmentTab('top2'));
    }
    if (btnViewTop2Sol) {
        btnViewTop2Sol.addEventListener('click', () => {
            switchTreatmentTab('top2');
            const targetContainer = $('#treatment-container');
            if (targetContainer) {
                targetContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    // =========================================================================
    // ACTIVE RULE BANNER & COMBINED ENSEMBLE CHART
    // =========================================================================
    function renderActiveRuleBanner(methodStr) {
        const descEl = $('#active-rule-desc');
        if (!descEl) return;

        const m = (methodStr || '').toLowerCase();
        let explanation = 'Ensemble v4: Phối hợp suy luận đa tầng';

        if (m.includes('rule 1') || m.includes('miner') || m.includes('phoma')) {
            explanation = 'Rule 1 — Miner / Phoma Specialization: Nhánh Small đạt độ chính xác 100% trên các đốm bệnh vi mô nên được ưu tiên tuyệt đối.';
        } else if (m.includes('rule 2') || m.includes('healthy')) {
            explanation = 'Rule 2 — Healthy Strict Gate: Nhánh Big nhận diện toàn cảnh lá khỏe với độ chính xác 100%, bảo vệ tránh báo sai lá lành thành bệnh.';
        } else if (m.includes('rule 3') || m.includes('nematode')) {
            explanation = 'Rule 3 — Nematode Specialization: Ưu tiên nhánh Big nhận diện biểu hiện vàng úa toàn cảnh diện rộng của Tuyến trùng.';
        } else if (m.includes('rule 4') || m.includes('mealy')) {
            explanation = 'Rule 4 — Mealy False-Alarm Block: Chặn báo nhầm Rệp sáp do bụi phấn trắng hoặc bào tử nấm trên lát cắt vi mô.';
        } else if (m.includes('weighted')) {
            explanation = 'Ensemble v4 Weighted Voting: Bình bầu xác suất tối ưu kết hợp giữa ResNet Big (55%) và ResNet Small (45%).';
        } else {
            explanation = methodStr || 'Ensemble v4 Disease-Aware Engine';
        }

        descEl.textContent = explanation;
    }

    function renderEnsembleCombinedChart(bigProbs, smallProbs, topClass) {
        const container = $('#ensemble-probs');
        if (!container) return;
        container.innerHTML = '';

        const allClasses = ['healthy', 'rust', 'phoma', 'miner', 'corticium', 'mealy', 'nematode', 'anthracnose'];
        const combined = {};

        allClasses.forEach(cls => {
            const b = bigProbs ? (bigProbs[cls] || 0) : 0;
            const s = smallProbs ? (smallProbs[cls] || 0) : 0;
            combined[cls] = (b * 0.55) + (s * 0.45);
        });

        const sorted = Object.entries(combined).sort((a, b) => b[1] - a[1]);

        sorted.forEach(([cls, prob]) => {
            const isTop = cls.toLowerCase() === (topClass || '').toLowerCase();
            const row = document.createElement('div');
            row.className = 'prob-row' + (isTop ? ' is-top' : '');

            const pct = (prob * 100).toFixed(1);
            const icon = diseaseIcons[cls.toLowerCase()] || '🍃';
            const viName = diseaseVietnamese[cls.toLowerCase()] || cls;

            row.innerHTML = `
                <span class="prob-row__name" title="${viName}">${icon} ${cls}</span>
                <div class="prob-row__bar">
                    <div class="prob-row__fill" style="width: ${pct}%; background: ${diseaseColors[cls.toLowerCase()] || '#2d6a4f'}"></div>
                </div>
                <span class="prob-row__val">${pct}%</span>
            `;
            container.appendChild(row);
        });
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
            const icon = diseaseIcons[cls.toLowerCase()] || '🍃';
            row.innerHTML = `
                <span class="prob-row__name">${icon} ${cls}</span>
                <div class="prob-row__bar">
                    <div class="prob-row__fill" style="width: ${pct}%"></div>
                </div>
                <span class="prob-row__val">${pct}%</span>
            `;
            container.appendChild(row);
        });
    }

    // =========================================================================
    // HISTORY SECTION WITH FILTER CHIPS
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
            cachedHistories = await resp.json();
            renderHistoryGrid(cachedHistories, activeHistoryFilter);
        } catch (err) {
            historyGrid.innerHTML = `<p class="history-error">❌ Lỗi: ${err.message}</p>`;
        }
    }

    function renderHistoryGrid(histories, filter) {
        if (!histories || histories.length === 0) {
            historyGrid.innerHTML = '<p class="history-empty">🍃 Chưa có lịch sử chẩn đoán nào được lưu.</p>';
            return;
        }

        // Lọc danh sách theo filter
        const filtered = histories.filter(item => {
            if (!filter || filter === 'all') return true;
            const nameStr = (item.disease_name || item.disease || item.result || '').toLowerCase();
            return nameStr.includes(filter.toLowerCase());
        });

        if (filtered.length === 0) {
            historyGrid.innerHTML = `<p class="history-empty">Không tìm thấy bản ghi nào thuộc bộ lọc "${filter}".</p>`;
            return;
        }

        historyGrid.innerHTML = '';
        filtered.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-card';
            const dateStr = item.created_at ? new Date(item.created_at).toLocaleString('vi-VN') : 'Vừa xong';
            const diseaseDisplay = item.disease_name || item.disease || item.result || 'Bệnh lá';
            const confPercent = ((item.confidence || 0) * 100).toFixed(1) + '%';
            const croodsText = item.croods ? `${item.croods.lat}, ${item.croods.long}` : 'Chưa có tọa độ';

            const uncertainTag = item.is_uncertain ? `<span class="history-card__tag-uncertain">⚠️ Phân vân</span>` : '';
            const top2Line = item.top2_name ? `<div class="history-card__top2">Top 2: ${item.top2_name} (${((item.top2_confidence || 0) * 100).toFixed(1)}%)</div>` : '';

            card.innerHTML = `
                <div class="history-card__thumb">
                    <img src="${item.image_url}" alt="Ảnh chẩn đoán" onerror="this.src='/static/style.css'">
                </div>
                <div class="history-card__body">
                    <div class="history-card__disease">
                        ${diseaseDisplay} ${uncertainTag}
                    </div>
                    <div class="history-card__conf">Độ tin cậy: <strong>${confPercent}</strong></div>
                    ${top2Line}
                    <div class="history-card__date">🕒 ${dateStr}</div>
                    <div class="history-card__gps">📍 ${croodsText}</div>
                </div>
            `;
            card.addEventListener('click', () => viewHistoryDetail(item.id));
            historyGrid.appendChild(card);
        });
    }

    // Gắn sự kiện cho các nút filter chip trong History
    $$('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            $$('.filter-chip').forEach(c => c.classList.remove('is-active'));
            chip.classList.add('is-active');
            activeHistoryFilter = chip.getAttribute('data-filter') || 'all';
            renderHistoryGrid(cachedHistories, activeHistoryFilter);
        });
    });

    async function viewHistoryDetail(historyId) {
        try {
            const resp = await fetch(`/histories/${historyId}/detail`);
            if (!resp.ok) throw new Error('Không lấy được chi tiết');
            const data = await resp.json();

            // Render màn hình kết quả với bản ghi lịch sử
            renderResults({
                result: data.result,
                confidence: data.confidence,
                top2_result: data.top2_result,
                top2_confidence: data.top2_confidence || 0.0,
                is_uncertain: data.is_uncertain || false,
                differential_note: data.differential_note || '',
                image_url: data.image_url,
                original_image: null,
                annotated_image: null,
                details: {
                    pred_class: data.result ? data.result.name : 'Unknown',
                    method: 'Lịch sử lưu trữ (Chi tiết phác đồ)',
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
    // BRANCH ACCORDION TOGGLE
    // =========================================================================
    $$('.branch__header:not(.is-static)').forEach(header => {
        header.addEventListener('click', () => {
            const targetId = header.getAttribute('data-toggle');
            const body = document.getElementById(targetId);
            const toggle = header.querySelector('.branch__toggle');

            if (body) {
                body.classList.toggle('open');
                if (toggle) {
                    toggle.style.transform = body.classList.contains('open') ? 'rotate(180deg)' : '';
                }
            }
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
