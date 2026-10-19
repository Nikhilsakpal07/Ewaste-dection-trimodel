document.addEventListener("DOMContentLoaded", () => {
    if (window.gsap && window.ScrollTrigger) {
        gsap.registerPlugin(ScrollTrigger);
    }

    // --- Custom Cursor ---
    const cursor = document.getElementById("cursor");
    const follower = document.getElementById("cursor-follower");
    if (cursor && follower) {
        document.addEventListener("mousemove", (e) => {
            gsap.to(cursor, { x: e.clientX, y: e.clientY, duration: 0.08 });
            gsap.to(follower, { x: e.clientX, y: e.clientY, duration: 0.22 });
        });
    }

    // --- HUD Clock ---
    const clockEl = document.getElementById("hud-clock");
    if (clockEl) {
        setInterval(() => {
            const d = new Date();
            clockEl.innerText = d.toISOString().split('T')[1].split('.')[0] + " UTC";
        }, 1000);
    }

    // --- HUD Random Hex Generator ---
    const hexEl = document.getElementById("random-hex");
    if (hexEl) {
        setInterval(() => {
            const hex = () => "0x" + Math.floor(Math.random() * 65535).toString(16).toUpperCase().padStart(4, '0');
            hexEl.innerText = `${hex()} ${hex()} ${hex()}`;
        }, 800);
    }

    // --- 3D Tilt Effect on Cards ---
    const tiltCards = document.querySelectorAll(".tilt-card");
    tiltCards.forEach(card => {
        card.addEventListener("mousemove", (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const xPct = x / rect.width - 0.5;
            const yPct = y / rect.height - 0.5;
            gsap.to(card, {
                rotationY: xPct * 12,
                rotationX: -yPct * 12,
                transformPerspective: 1000,
                ease: "power1.out",
                duration: 0.4
            });
        });
        card.addEventListener("mouseleave", () => {
            gsap.to(card, { rotationY: 0, rotationX: 0, duration: 0.4, ease: "power1.out" });
        });
    });

    // --- State & DOM References ---
    let activeInputMode = "samples"; // samples | upload | webcam | live
    let currentSample = "pcb";
    let currentUploadedFile = null;
    let currentSnapshotBlob = null;
    let webcamStream = null;
    let lastScanData = null;

    // Elements
    const dataStreamEl = document.getElementById("data-stream");
    const globalStatusText = document.getElementById("global-status-text");
    const btnRunAll = document.getElementById("btn-run-all");

    // Mode Buttons
    const btnModeSamples = document.getElementById("btn-mode-samples");
    const btnModeUpload = document.getElementById("btn-mode-upload");
    const btnModeWebcam = document.getElementById("btn-mode-webcam");
    const btnModeLive = document.getElementById("btn-mode-live");

    // Subpanels
    const samplesContainer = document.getElementById("samples-container");
    const uploadPanel = document.getElementById("upload-panel");
    const webcamPanel = document.getElementById("webcam-panel");
    const livePanel = document.getElementById("live-panel");
    const streamSelectorBox = document.getElementById("stream-selector-box");

    // Upload & Webcam elements
    const fileInput = document.getElementById("file-input");
    const btnBrowseFile = document.getElementById("btn-browse-file");
    const uploadDropzone = document.getElementById("upload-dropzone");
    const webcamVideo = document.getElementById("webcam-stream");
    const webcamCanvas = document.getElementById("webcam-snapshot-canvas");
    const btnTakeSnapshot = document.getElementById("btn-take-snapshot");
    const liveVideoFeed = document.getElementById("live-video-feed");

    // Report elements
    const reportContent = document.getElementById("report-content");
    const btnCopyReport = document.getElementById("btn-copy-report");
    const btnRegenReport = document.getElementById("btn-regen-report");
    const btnLocate = document.getElementById("btn-locate");
    const locatorStatus = document.getElementById("locator-status");

    // --- Mode Switching Logic ---
    function setInputMode(mode) {
        activeInputMode = mode;
        [btnModeSamples, btnModeUpload, btnModeWebcam, btnModeLive].forEach(b => b.classList.remove("active"));
        samplesContainer.style.display = "none";
        uploadPanel.style.display = "none";
        webcamPanel.style.display = "none";
        livePanel.style.display = "none";
        streamSelectorBox.style.display = "none";

        // Stop live stream if leaving live mode
        if (mode !== "live" && liveVideoFeed.src) {
            liveVideoFeed.src = "";
            fetch("/api/stop_stream", { method: "POST" }).catch(() => {});
        }

        // Stop webcam stream if leaving webcam mode
        if (mode !== "webcam" && webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }

        if (mode === "samples") {
            btnModeSamples.classList.add("active");
            samplesContainer.style.display = "block";
            updateDataStream(`[ READY ] Select preset hardware sample.`);
        } else if (mode === "upload") {
            btnModeUpload.classList.add("active");
            uploadPanel.style.display = "block";
            updateDataStream(`[ READY ] Select or drag an e-waste photograph.`);
        } else if (mode === "webcam") {
            btnModeWebcam.classList.add("active");
            webcamPanel.style.display = "block";
            initWebcam();
            updateDataStream(`[ WEBCAM ] Position hardware in frame and take snapshot.`);
        } else if (mode === "live") {
            btnModeLive.classList.add("active");
            livePanel.style.display = "block";
            streamSelectorBox.style.display = "flex";
            startLiveStream("yolov8");
            updateDataStream(`[ LIVE STREAM ] Real-time video detection active.`);
        }
    }

    btnModeSamples.addEventListener("click", () => setInputMode("samples"));
    btnModeUpload.addEventListener("click", () => setInputMode("upload"));
    btnModeWebcam.addEventListener("click", () => setInputMode("webcam"));
    btnModeLive.addEventListener("click", () => setInputMode("live"));

    // --- Sample Presets Selection ---
    const sampleBtns = document.querySelectorAll(".sample-btn");
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            sampleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentSample = btn.dataset.sample;
            runSampleAudit(currentSample);
        });
    });

    // Default activate first sample
    if (sampleBtns.length > 0) {
        sampleBtns[0].classList.add("active");
    }

    // --- Upload Dropzone Handling ---
    if (btnBrowseFile && fileInput) {
        btnBrowseFile.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (uploadDropzone) {
        uploadDropzone.addEventListener("click", () => fileInput.click());
        uploadDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadDropzone.style.borderColor = "var(--primary)";
            uploadDropzone.style.background = "rgba(0, 255, 163, 0.1)";
        });
        uploadDropzone.addEventListener("dragleave", () => {
            uploadDropzone.style.borderColor = "";
            uploadDropzone.style.background = "";
        });
        uploadDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadDropzone.style.borderColor = "";
            uploadDropzone.style.background = "";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    function handleFile(file) {
        currentUploadedFile = file;
        updateDataStream(`[ FILE LOADED ] ${file.name} (${Math.round(file.size / 1024)} KB)`);
        runUploadedFileAudit(file);
    }

    // --- Webcam Capture ---
    function initWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640, facingMode: "environment" } })
                .then(stream => {
                    webcamStream = stream;
                    webcamVideo.srcObject = stream;
                })
                .catch(err => {
                    console.error("Webcam error:", err);
                    alert("Could not access camera. Please verify camera permissions in browser.");
                });
        }
    }

    if (btnTakeSnapshot) {
        btnTakeSnapshot.addEventListener("click", () => {
            if (!webcamVideo || webcamVideo.videoWidth === 0) return;
            webcamCanvas.width = webcamVideo.videoWidth;
            webcamCanvas.height = webcamVideo.videoHeight;
            const ctx = webcamCanvas.getContext("2d");
            ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
            webcamCanvas.toBlob(blob => {
                currentSnapshotBlob = blob;
                const file = new File([blob], "webcam_snap.jpg", { type: "image/jpeg" });
                updateDataStream("[ SNAPSHOT CAPTURED ] Executing 3-model scan...");
                runUploadedFileAudit(file);
            }, "image/jpeg", 0.9);
        });
    }

    // --- Live Stream Model Switching ---
    const streamBtns = document.querySelectorAll(".stream-btn");
    streamBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            streamBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const m = btn.dataset.model;
            startLiveStream(m);
        });
    });

    function startLiveStream(modelName) {
        fetch("/api/set_stream_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelName })
        }).then(() => {
            liveVideoFeed.src = `/video_feed?model=${modelName}&t=${Date.now()}`;
        }).catch(err => console.error("Error setting stream model:", err));
    }

    // --- Master Execution Button ---
    if (btnRunAll) {
        btnRunAll.addEventListener("click", () => {
            if (activeInputMode === "samples") {
                runSampleAudit(currentSample);
            } else if (activeInputMode === "upload") {
                if (currentUploadedFile) {
                    runUploadedFileAudit(currentUploadedFile);
                } else {
                    fileInput.click();
                }
            } else if (activeInputMode === "webcam") {
                if (btnTakeSnapshot) btnTakeSnapshot.click();
            } else if (activeInputMode === "live") {
                // In live mode, capture current frame and run 3 models
                captureLiveStreamSnapshot();
            }
        });
    }

    function captureLiveStreamSnapshot() {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(liveVideoFeed, 0, 0, 640, 640);
        canvas.toBlob(blob => {
            const file = new File([blob], "live_snap.jpg", { type: "image/jpeg" });
            runUploadedFileAudit(file);
        }, "image/jpeg", 0.9);
    }

    // --- Simultaneous Multi-Model Execution APIs ---
    function setScanningUI(isScanning) {
        if (btnRunAll) {
            if (isScanning) {
                btnRunAll.classList.add("loading");
                btnRunAll.innerText = "⏳ SCANNING ALL 3 ENGINES PARALLEL...";
            } else {
                btnRunAll.classList.remove("loading");
                btnRunAll.innerText = "⚡ RUN SIMULTANEOUS 3-MODEL SCAN";
            }
        }
        if (globalStatusText) {
            globalStatusText.innerText = isScanning ? "INSPECTING WITH YOLOv8, YOLOv5, AND FASTER R-CNN..." : "ENGINES READY (YOLOv8 // YOLOv5 // FASTER R-CNN)";
        }
    }

    function updateDataStream(msg) {
        if (dataStreamEl) dataStreamEl.innerText = msg;
    }

    function runSampleAudit(sampleName) {
        setScanningUI(true);
        updateDataStream(`[ DISPATCH ] Dispatching sample '${sampleName}' across all 3 neural networks...`);

        fetch("/api/analyze_sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sample_name: sampleName, conf: 0.15 })
        })
        .then(res => res.json())
        .then(data => {
            setScanningUI(false);
            if (data.error) {
                console.error(data.error);
                updateDataStream(`[ ERROR ] ${data.error}`);
                alert("Inspection Error: " + data.error);
                return;
            }
            renderMultiModelResults(data);
        })
        .catch(err => {
            setScanningUI(false);
            console.error(err);
            updateDataStream("[ NETWORK ERROR ] Could not reach vision server.");
        });
    }

    function runUploadedFileAudit(file) {
        setScanningUI(true);
        updateDataStream(`[ DISPATCH ] Uploading ${file.name} to 3-model parallel vision suite...`);

        const formData = new FormData();
        formData.append("image", file);
        formData.append("conf", "0.15");

        fetch("/api/analyze_multi_models", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            setScanningUI(false);
            if (data.error) {
                console.error(data.error);
                updateDataStream(`[ ERROR ] ${data.error}`);
                alert("Inspection Error: " + data.error);
                return;
            }
            renderMultiModelResults(data);
        })
        .catch(err => {
            setScanningUI(false);
            console.error(err);
            updateDataStream("[ NETWORK ERROR ] Could not reach vision server.");
        });
    }

    // --- Render All 3 Models in the Dashboard Simultaneously ---
    function renderMultiModelResults(data) {
        lastScanData = data;
        const models = data.models || {};
        const comp = data.comparison || {};
        const report = data.report || "";

        // 1. UPDATE DASHBOARD SECTION 1: YOLOv8
        const v8 = models.yolov8 || {};
        updateSingleModelCard("v8", v8);

        // 2. UPDATE DASHBOARD SECTION 2: YOLOv5
        const v5 = models.yolov5 || {};
        updateSingleModelCard("v5", v5);

        // 3. UPDATE DASHBOARD SECTION 3: FASTER R-CNN
        const frcnn = models.faster_rcnn || {};
        updateSingleModelCard("frcnn", frcnn);

        // 4. UPDATE SIDE-BY-SIDE BENCHMARK MATRIX TABLE
        updateBenchmarkMatrix(v8, v5, frcnn);

        // 5. UPDATE GEMINI ENVIRONMENTAL IMPACT REPORT
        renderGeminiReport(report);

        updateDataStream(`[ TRI-MODEL AUDIT COMPLETE ] Consensus: ${comp.consensus_top ? comp.consensus_top.class : 'None'} | Fastest: ${comp.fastest_model} (${comp.fastest_latency}ms)`);

        // GSAP Stagger Highlight on update
        if (window.gsap) {
            gsap.from(".model-card", {
                y: 12,
                opacity: 0.8,
                stagger: 0.08,
                duration: 0.35,
                ease: "power2.out"
            });
        }
    }

    function updateSingleModelCard(prefix, m) {
        const imgEl = document.getElementById(`${prefix}-img`);
        const latEl = document.getElementById(`${prefix}-latency`);
        const countEl = document.getElementById(`${prefix}-count`);
        const fpsEl = document.getElementById(`${prefix}-fps`);
        const clsEl = document.getElementById(`${prefix}-class`);
        const confEl = document.getElementById(`${prefix}-conf`);
        const meterEl = document.getElementById(`${prefix}-meter`);
        const valEl = document.getElementById(`${prefix}-value`);
        const hazEl = document.getElementById(`${prefix}-hazard`);
        const hazBox = document.getElementById(`${prefix}-hazard-box`);
        const tagsEl = document.getElementById(`${prefix}-tags`);

        if (imgEl && m.image) imgEl.src = m.image;
        if (latEl) latEl.innerText = (m.latency_ms !== undefined) ? `${m.latency_ms} ms` : "--";
        if (countEl) countEl.innerText = m.count !== undefined ? m.count : 0;
        if (fpsEl && prefix !== "frcnn") fpsEl.innerText = m.fps ? `${m.fps} FPS` : "--";

        const top = m.top_detection;
        if (top) {
            if (clsEl) clsEl.innerText = top.class.toUpperCase();
            if (confEl) confEl.innerText = `${top.confidence}%`;
            if (meterEl) meterEl.style.width = `${top.confidence}%`;
        } else {
            if (clsEl) clsEl.innerText = m.count === 0 ? "NONE DETECTED" : "SCANNING...";
            if (confEl) confEl.innerText = "0%";
            if (meterEl) meterEl.style.width = "0%";
        }

        if (valEl) valEl.innerText = m.total_value || "₹0";

        if (hazEl && hazBox) {
            if (m.has_hazard) {
                hazEl.innerText = "⚠️ HAZARD RISK";
                hazBox.classList.add("danger");
            } else {
                hazEl.innerText = "SECURE";
                hazBox.classList.remove("danger");
            }
        }

        if (tagsEl) {
            const dets = m.detections || [];
            if (dets.length === 0) {
                tagsEl.innerHTML = `<span class="no-tags">No items detected above threshold</span>`;
            } else {
                tagsEl.innerHTML = dets.map(d => {
                    const hClass = d.hazard ? "hazard-tag" : "";
                    return `<span class="b-tag ${hClass}">${d.class} (${d.confidence}%)</span>`;
                }).join(" ");
            }
        }
    }

    function updateBenchmarkMatrix(v8, v5, frcnn) {
        const setTxt = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.innerText = txt;
        };

        // Latencies
        setTxt("tbl-v8-latency", v8.latency_ms ? `${v8.latency_ms} ms (${v8.fps} FPS)` : "--");
        setTxt("tbl-v5-latency", v5.latency_ms ? `${v5.latency_ms} ms (${v5.fps} FPS)` : "--");
        setTxt("tbl-frcnn-latency", frcnn.latency_ms ? `${frcnn.latency_ms} ms (Two-Stage)` : "--");

        // Object Counts
        setTxt("tbl-v8-count", v8.count !== undefined ? v8.count : 0);
        setTxt("tbl-v5-count", v5.count !== undefined ? v5.count : 0);
        setTxt("tbl-frcnn-count", frcnn.count !== undefined ? frcnn.count : 0);

        // Primary Classification
        const fmtPrimary = (m) => m.top_detection ? `${m.top_detection.class} (${m.top_detection.confidence}%)` : "None";
        setTxt("tbl-v8-primary", fmtPrimary(v8));
        setTxt("tbl-v5-primary", fmtPrimary(v5));
        setTxt("tbl-frcnn-primary", fmtPrimary(frcnn));

        // Total Value
        setTxt("tbl-v8-val", v8.total_value || "₹0");
        setTxt("tbl-v5-val", v5.total_value || "₹0");
        setTxt("tbl-frcnn-val", frcnn.total_value || "₹0");

        // Hazard status
        const fmtHaz = (m) => m.has_hazard ? "⚠️ HAZARD FLAGGED" : "SECURE";
        setTxt("tbl-v8-hazard", fmtHaz(v8));
        setTxt("tbl-v5-hazard", fmtHaz(v5));
        setTxt("tbl-frcnn-hazard", fmtHaz(frcnn));
    }

    function renderGeminiReport(markdownText) {
        if (!reportContent) return;
        if (window.marked) {
            reportContent.innerHTML = marked.parse(markdownText);
        } else {
            reportContent.innerText = markdownText;
        }
    }

    // --- Report Action Buttons ---
    if (btnCopyReport) {
        btnCopyReport.addEventListener("click", () => {
            if (lastScanData && lastScanData.report) {
                navigator.clipboard.writeText(lastScanData.report).then(() => {
                    btnCopyReport.innerText = "✓ COPIED TO CLIPBOARD";
                    setTimeout(() => { btnCopyReport.innerText = "📋 COPY REPORT"; }, 2500);
                });
            } else {
                alert("Please perform a scan first to generate a report.");
            }
        });
    }

    if (btnRegenReport) {
        btnRegenReport.addEventListener("click", () => {
            if (!lastScanData) {
                alert("No active scan available to audit.");
                return;
            }
            updateDataStream("[ GEMINI LLM ] Re-synthesizing technical e-waste audit...");
            reportContent.innerHTML = `<div class="report-placeholder"><span class="spinner-icon">⚙️</span><p>Re-generating deep environmental analysis report with Gemini...</p></div>`;

            const classes = lastScanData.comparison ? lastScanData.comparison.classes_detected : [];
            fetch("/api/generate_report", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ detected_classes: classes })
            })
            .then(res => res.json())
            .then(data => {
                if (data.report) {
                    renderGeminiReport(data.report);
                    updateDataStream("[ GEMINI REPORT READY ] Re-generated successfully.");
                }
            })
            .catch(err => {
                console.error(err);
                updateDataStream("[ ERROR ] Re-generation failed.");
            });
        });
    }

    // --- Geolocation Drop-Off Locator ---
    if (btnLocate) {
        btnLocate.addEventListener("click", () => {
            if (!locatorStatus) return;
            locatorStatus.innerText = "Requesting GPS coordinates...";

            const detectedClass = (lastScanData && lastScanData.comparison && lastScanData.comparison.consensus_top) 
                ? lastScanData.comparison.consensus_top.class 
                : "e-waste";

            const query = `${detectedClass} recycling centers near me`;

            if ("geolocation" in navigator) {
                navigator.geolocation.getCurrentPosition(
                    (pos) => {
                        const lat = pos.coords.latitude;
                        const lng = pos.coords.longitude;
                        locatorStatus.innerText = "Coordinates verified. Launching certified maps routing...";
                        const mapsUrl = `https://www.google.com/maps/search/${encodeURIComponent(query)}/@${lat},${lng},12z`;
                        window.open(mapsUrl, "_blank");
                        setTimeout(() => { locatorStatus.innerText = ""; }, 3000);
                    },
                    (err) => {
                        locatorStatus.innerText = "Location permission unavailable. Launching regional search...";
                        window.open(`https://www.google.com/maps/search/${encodeURIComponent(query)}`, "_blank");
                        setTimeout(() => { locatorStatus.innerText = ""; }, 3000);
                    }
                );
            } else {
                window.open(`https://www.google.com/maps/search/${encodeURIComponent(query)}`, "_blank");
            }
        });
    }

    // --- Inventory Modal Logic ---
    const btnInventory = document.getElementById("btn-inventory");
    const inventoryModal = document.getElementById("inventory-modal");
    const btnCloseInventory = document.getElementById("btn-close-inventory");
    const inventoryContainer = document.getElementById("inventory-container");

    if (btnInventory && inventoryModal) {
        btnInventory.addEventListener("click", (e) => {
            e.preventDefault();
            inventoryModal.style.display = "flex";
            loadInventoryRecords();
        });
        if (btnCloseInventory) {
            btnCloseInventory.addEventListener("click", () => {
                inventoryModal.style.display = "none";
            });
        }
    }

    function loadInventoryRecords() {
        if (!inventoryContainer) return;
        inventoryContainer.innerHTML = '<p class="muted-text">Fetching database records...</p>';

        fetch('/api/inventory')
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    inventoryContainer.innerHTML = '<p class="muted-text" style="padding: 40px; text-align: center;">No e-waste items logged yet. Perform a scan to save items.</p>';
                    return;
                }

                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px;">';
                data.forEach(item => {
                    const info = item.info || {};
                    const hazardStyle = info.hazard ? 'border-color: var(--hazard);' : '';
                    const hazardBadge = info.hazard ? '<div style="color: var(--hazard); font-size: 0.75rem; font-weight: bold; margin-bottom: 6px;">⚠️ HAZARDOUS RECYCLING REQUIRED</div>' : '';

                    html += `
                        <div class="glass-card" style="padding: 12px; ${hazardStyle}">
                            <img src="${item.image_url}" style="width: 100%; height: 140px; object-fit: cover; border-radius: 4px; margin-bottom: 10px;" />
                            <h4 style="color: var(--primary); font-family: 'Share Tech Mono';">${info.class || 'Unknown'}</h4>
                            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;">${item.timestamp}</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: var(--gold); margin-bottom: 6px;">${info.value || '₹0'}</div>
                            ${hazardBadge}
                            <p style="font-size: 0.75rem; color: var(--text-muted);">${info.summary || ''}</p>
                        </div>
                    `;
                });
                html += '</div>';
                inventoryContainer.innerHTML = html;
            })
            .catch(() => {
                inventoryContainer.innerHTML = '<p style="color: var(--hazard);">Failed to load inventory.</p>';
            });
    }

    // --- Mobile Link Modal ---
    const btnMobileConnect = document.getElementById("btn-mobile-connect");
    const mobileModal = document.getElementById("mobile-modal");
    const qrCode = document.getElementById("qr-code");
    const btnCloseModal = document.getElementById("btn-close-modal");

    if (btnMobileConnect && mobileModal) {
        btnMobileConnect.addEventListener("click", (e) => {
            e.preventDefault();
            const url = window.location.origin + '/mobile';
            if (qrCode) {
                qrCode.src = `https://api.qrserver.com/v1/create-qr-code/?size=240x240&data=${encodeURIComponent(url)}&bgcolor=FFFFFF`;
            }
            mobileModal.style.display = "flex";
        });
        if (btnCloseModal) {
            btnCloseModal.addEventListener("click", () => {
                mobileModal.style.display = "none";
            });
        }
    }

    // --- Desktop Polling for Mobile Upload Sync ---
    setInterval(() => {
        fetch("/api/desktop_sync")
            .then(res => res.json())
            .then(data => {
                if (data.has_new && data.data) {
                    updateDataStream("[ MOBILE SYNC ] Real-time mobile audit received!");
                    renderMultiModelResults(data.data);
                }
            })
            .catch(() => {});
    }, 2500);

    // Initial Trigger on Page Load: Run PCB Sample automatically so user sees the 3 models immediately
    setTimeout(() => {
        runSampleAudit("pcb");
    }, 600);
});
