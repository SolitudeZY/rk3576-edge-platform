const $ = (id) => document.getElementById(id);
const state = { hls: null, hlsUrl: "", polling: false };

function formatBytes(bytes, unit = "GB") {
  const divisor = unit === "GB" ? 1024 ** 3 : 1024 ** 2;
  return `${(Number(bytes || 0) / divisor).toFixed(unit === "GB" ? 1 : 0)} ${unit}`;
}
function number(value, digits = 1) { return Number(value || 0).toFixed(digits); }
function meter(name, value) {
  const safe = Math.max(0, Math.min(100, Number(value || 0)));
  $(`${name}-value`).textContent = `${safe.toFixed(0)}%`;
  $(`${name}-meter`).style.width = `${safe}%`;
}
function setError(message = "") { $("error-message").textContent = message; }

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `请求失败 ${response.status}`);
  return payload;
}

async function refreshVideos(selected = "") {
  const payload = await api("/api/videos");
  const select = $("video-select");
  select.innerHTML = '<option value="">请选择已上传视频</option>';
  payload.videos.forEach((video) => {
    const option = document.createElement("option");
    option.value = video.name;
    option.textContent = `${video.name} · ${formatBytes(video.bytes, "MB")}`;
    option.selected = video.name === selected;
    select.appendChild(option);
  });
}

async function refreshModels(selected = "") {
  const payload = await api("/api/models");
  const select = $("model-select");
  const preferred = selected || select.value || payload.active;
  select.innerHTML = "";
  payload.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = `${model.name} · ${formatBytes(model.bytes, "MB")}`;
    option.selected = model.name === preferred;
    select.appendChild(option);
  });
  if (!payload.models.length) {
    select.innerHTML = '<option value="">板端没有可用模型</option>';
  }
}

function attachHls(url) {
  if (!url || state.hlsUrl === url) return;
  state.hlsUrl = url;
  const video = $("detected-video");
  if (state.hls) state.hls.destroy();
  if (window.Hls && Hls.isSupported()) {
    state.hls = new Hls({ lowLatencyMode: true, liveSyncDurationCount: 2, maxLiveSyncPlaybackRate: 1.5 });
    state.hls.loadSource(url);
    state.hls.attachMedia(video);
    state.hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(() => {}));
    state.hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) { state.hlsUrl = ""; setTimeout(() => attachHls(url), 1500); }
    });
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = url;
    video.play().catch(() => {});
  }
}

function resetPlayer() {
  const video = $("detected-video");
  if (state.hls) state.hls.destroy();
  state.hls = null;
  state.hlsUrl = "";
  video.pause();
  video.removeAttribute("src");
  video.load();
}

function renderStatus(payload) {
  const edge = payload.edge || {};
  const online = Boolean(payload.edge_online && edge.running && edge.publisher_connected);
  $("status-dot").classList.toggle("online", online);
  $("status-label").textContent = online ? "实时链路运行中" : payload.edge_online ? "边缘服务待机" : "板端离线";
  $("video-empty").classList.toggle("hidden", online);
  $("fps").textContent = number(edge.fps);
  $("inference-ms").textContent = `${number(edge.inference_ms)} ms`;
  $("pipeline-ms").textContent = `${number(edge.pipeline_ms)} ms`;
  $("detections").textContent = edge.detections || 0;
  $("frame-counter").textContent = `${edge.frames_processed || 0} 帧`;
  $("preprocess-ms").textContent = `${number(edge.preprocess_ms)} ms`;
  $("postprocess-ms").textContent = `${number(edge.postprocess_ms)} ms`;
  $("capture-latency").textContent = `${number(edge.capture_to_publish_ms)} ms`;
  $("dropped-frames").textContent = edge.frames_dropped || 0;
  meter("cpu", edge.cpu_percent);
  meter("npu", edge.npu_percent);
  meter("memory", edge.memory_percent);
  const rss = Number(edge.npu_memory_bytes || 0);
  $("npu-memory-value").textContent = formatBytes(rss, "MB");
  $("npu-memory-meter").style.width = `${Math.min(100, rss / (1024 ** 2 * 10))}%`;
  $("memory-detail").textContent = `${formatBytes(edge.memory_used_bytes)} / ${formatBytes(edge.memory_total_bytes)}`;
  const cores = edge.npu_cores_percent || [];
  $("core0").textContent = `${number(cores[0], 0)}%`;
  $("core1").textContent = `${number(cores[1], 0)}%`;
  $("npu-frequency").textContent = `${number(Number(edge.npu_frequency_hz || 0) / 1e6, 0)} MHz`;
  if (payload.urls) {
    $("source-url").textContent = payload.urls.source_rtsp;
    $("detected-url").textContent = payload.urls.detected_rtsp;
    attachHls(payload.urls.detected_hls);
  }
  if (edge.last_error && edge.last_error !== "Source disconnected; reconnecting") setError(edge.last_error);
}

async function pollStatus() {
  if (state.polling) return;
  state.polling = true;
  try { renderStatus(await api("/api/status")); } catch (error) { setError(error.message); }
  finally { state.polling = false; }
}

async function upload(file) {
  if (!file) return;
  const progress = $("upload-progress");
  const copy = $("upload-copy");
  progress.classList.add("active"); progress.value = 15; copy.textContent = `上传中 · ${file.name}`;
  try {
    const xhr = new XMLHttpRequest();
    const complete = new Promise((resolve, reject) => {
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) progress.value = 15 + event.loaded / event.total * 80; };
      xhr.onload = () => xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).error));
      xhr.onerror = () => reject(new Error("上传失败"));
    });
    xhr.open("POST", "/api/upload"); xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name)); xhr.send(file);
    const result = await complete; progress.value = 100; copy.textContent = `已上传 · ${file.name}`;
    await refreshVideos(result.name);
  } catch (error) { setError(error.message); copy.textContent = "上传失败"; }
  finally { setTimeout(() => progress.classList.remove("active"), 700); }
}

async function uploadImage(file) {
  if (!file) return;
  const zone = $("image-upload-zone");
  const meta = $("image-result-meta");
  zone.classList.add("busy");
  meta.textContent = `检测中 · ${file.name}`;
  try {
    const response = await fetch("/api/image", {
      method: "POST",
      headers: {
        "X-Filename": encodeURIComponent(file.name),
        "X-Model": encodeURIComponent($("model-select").value),
        "X-Conf": encodeURIComponent($("conf").value),
        "X-Iou": "0.6",
      },
      body: file,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `图片检测失败 ${response.status}`);
    $("image-result").src = `data:image/jpeg;base64,${payload.image_base64}`;
    $("image-result-shell").classList.remove("hidden");
    meta.textContent = `检测 ${payload.detections} 个目标 · 推理 ${number(payload.inference_ms)} ms · 总计 ${number(Number(payload.preprocess_ms) + Number(payload.inference_ms) + Number(payload.postprocess_ms))} ms`;
    setError("");
  } catch (error) {
    meta.textContent = "图片检测失败";
    setError(error.message);
  } finally {
    zone.classList.remove("busy");
  }
}

$("video-file").addEventListener("change", (event) => upload(event.target.files[0]));
$("image-file").addEventListener("change", (event) => uploadImage(event.target.files[0]));
const zone = $("upload-zone");
["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragging"); }));
zone.addEventListener("drop", (event) => upload(event.dataTransfer.files[0]));

$("start-button").addEventListener("click", async () => {
  setError("");
  const video = $("video-select").value;
  if (!video) return setError("请先上传并选择视频");
  try {
    resetPlayer();
    await api("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      video, conf: Number($("conf").value), output_fps: Number($("output-fps").value),
      cores: $("cores").value, encoder: $("encoder").value,
      model: $("model-select").value, warmup: 20
    }) });
    setTimeout(pollStatus, 500);
  } catch (error) { setError(error.message); }
});
$("stop-button").addEventListener("click", async () => { try { resetPlayer(); await api("/api/stop", { method: "POST" }); } catch (error) { setError(error.message); } });
$("fullscreen-button").addEventListener("click", () => $("video-shell").requestFullscreen?.());
$("mute-button").addEventListener("click", () => { const v = $("detected-video"); v.muted = !v.muted; $("mute-button").textContent = v.muted ? "M" : "S"; });

setInterval(() => $("clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }), 1000);
setInterval(pollStatus, 1000);
refreshVideos().catch((error) => setError(error.message));
refreshModels().catch((error) => setError(error.message));
pollStatus();
