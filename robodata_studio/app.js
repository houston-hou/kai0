const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const viewTitles = {
  dashboard: ["Dashboard", "机器人数据、训练和推理的一体化管理入口。"],
  datasets: ["Datasets", "扫描和检查本地 LeRobot 数据集。"],
  episode: ["Episode Viewer", "查看 episode 视频、row 元信息和数值曲线。"],
  editor: ["Data Editor", "Convert HDF5, split atomic actions from keyframes, and clean LeRobot datasets."],
  training: ["Training", "训练配置和命令预览，后续接入任务启动和日志。"],
  inference: ["Inference", "推理服务配置和命令预览，后续接入服务状态。"],
  replay: ["Replay & Evaluation", "离线回放、专家对比和误差评估。"],
  jobs: ["Jobs", "查看平台发起的长任务状态。"],
  settings: ["Settings", "平台路径、端口和默认参数。"],
};
viewTitles.conversion = ["Conversion", "Convert HDF5 folders to LeRobot datasets."];

let datasets = [];
let selectedDataset = null;
let currentEpisodes = [];
let currentSeries = null;
let currentFeature = null;
let currentRow = 0;
let playTimer = null;
let currentEpisodeName = null;
let imageRequestToken = 0;
let replayDim = 0;
let compareDataset = null;
let compareEpisodes = [];
let compareSeries = null;
let compareEpisodeName = null;
let compareFeature = null;
let conversionTasks = [];
let atomicSegments = [];

const atomicPresetPrompts = {
  liquid: [
    "beaker_to_graduated_cylinder | pour solution from the beaker into the graduated cylinder",
    "graduated_cylinder_to_reactor | pour solution from the graduated cylinder into the reactor",
  ].join("\n"),
  solid: [
    "pick_funnel_to_reactor | pick up the funnel and place it on the reactor",
    "pick_weighing_boat_to_balance | pick up the weighing boat and place it on the balance",
    "press_tare_button | press the tare button on the balance",
    "scoop_solid_to_weighing_boat | scoop solid into the weighing boat",
    "pour_solid_to_reactor | pour the solid into the reactor",
  ].join("\n"),
  mix_distill: [
    "return_funnel_to_rack | pick up the funnel and put it back on the funnel rack",
    "place_distillation_rack | place the distillation rack",
    "turn_reactor_knob | turn the reactor knob",
  ].join("\n"),
};

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const raw = await response.text();
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = JSON.parse(raw).error || message;
    } catch {
      if (raw) message = raw;
    }
    throw new Error(message);
  }
  return raw ? JSON.parse(raw) : null;
}

function formatCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "-";
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "-";
}

function renderMetricCards(container, items) {
  container.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong>`;
    container.appendChild(card);
  });
}

function featureRow(series, key, rowIndex) {
  return series?.featureData?.[key]?.[rowIndex] || [];
}

function scalarSeries(series, key) {
  return (series?.featureData?.[key] || []).map((row) => Number(row?.[0]));
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : NaN;
}

function maxValue(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.max(...finite) : NaN;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function savedDatasetPaths() {
  try {
    const value = JSON.parse(localStorage.getItem("robodataDatasetPaths") || "[]");
    return Array.isArray(value) ? value.filter((item) => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function setView(viewId) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === viewId));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === viewId));
  const [title, subtitle] = viewTitles[viewId] || viewTitles.dashboard;
  $("#viewTitle").textContent = title;
  $("#viewSubtitle").textContent = subtitle;
  if (viewId === "jobs") loadJobs();
  if (viewId === "replay") {
    renderReplayControls();
    drawReplayCanvas();
    drawCompareCanvas();
  }
}

function renderDatasetOptions() {
  const select = $("#globalDatasetSelect");
  select.innerHTML = "";
  if (!datasets.length) {
    const option = document.createElement("option");
    option.textContent = "No datasets";
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.label;
    select.appendChild(option);
  });
  if (!selectedDataset || !datasets.some((item) => item.id === selectedDataset.id)) {
    selectedDataset = datasets[0];
  }
  select.value = selectedDataset.id;
  $("#trimOutputInput").placeholder = `${selectedDataset.absoluteRoot}_trimmed`;
  const atomicPrefix = $("#atomicRepoPrefixInput");
  if (atomicPrefix) atomicPrefix.placeholder = `${selectedDataset.label}_atomic`;
  const atomicOutputRoot = $("#atomicOutputRootInput");
  if (atomicOutputRoot && !atomicOutputRoot.value) {
    atomicOutputRoot.placeholder = selectedDataset.absoluteRoot.replace(/[\\/][^\\/]+$/, "");
  }
  const atomicBatchSourceRoot = $("#atomicBatchSourceRootInput");
  if (atomicBatchSourceRoot && !atomicBatchSourceRoot.value) {
    atomicBatchSourceRoot.placeholder = selectedDataset.absoluteRoot.replace(/[\\/][^\\/]+$/, "");
  }
}

function renderDashboard() {
  const totalEpisodes = datasets.reduce((sum, item) => sum + (Number(item.totalEpisodes) || 0), 0);
  const totalFrames = datasets.reduce((sum, item) => sum + (Number(item.totalFrames) || 0), 0);
  const totalVideos = datasets.reduce((sum, item) => sum + (Number(item.videoCount || item.totalVideos) || 0), 0);
  $("#metricDatasets").textContent = formatCount(datasets.length);
  $("#metricEpisodes").textContent = formatCount(totalEpisodes);
  $("#metricFrames").textContent = formatCount(totalFrames);
  $("#metricVideos").textContent = formatCount(totalVideos);

  const list = $("#dashboardDatasetList");
  list.innerHTML = "";
  datasets.slice(0, 8).forEach((dataset) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <strong>${dataset.label}</strong>
      <span>${formatCount(dataset.totalEpisodes)} eps</span>
      <span>${formatCount(dataset.totalFrames)} frames</span>
      <span>${formatCount(dataset.videoCount || dataset.totalVideos)} videos</span>
      <button type="button">Open</button>
    `;
    row.querySelector("button").addEventListener("click", () => {
      selectedDataset = dataset;
      renderDatasetOptions();
      setView("episode");
      loadDatasetDetails();
    });
    list.appendChild(row);
  });
}

function renderDatasetTable() {
  const body = $("#datasetTableBody");
  body.innerHTML = "";
  datasets.forEach((dataset) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><strong>${dataset.label}</strong><br><span>${dataset.root}</span></td>
      <td>${formatCount(dataset.totalEpisodes)}</td>
      <td>${formatCount(dataset.totalFrames)}</td>
      <td>${formatCount(dataset.parquetCount)}</td>
      <td>${formatCount(dataset.videoCount || dataset.totalVideos)}</td>
      <td>${dataset.fps || "-"}</td>
    `;
    row.addEventListener("click", () => {
      selectedDataset = dataset;
      renderDatasetOptions();
      loadDatasetDetails();
    });
    body.appendChild(row);
  });
}

function renderFeatureOptions(keys) {
  const select = $("#featureSelect");
  const previous = currentFeature;
  select.innerHTML = "";
  keys.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key;
    select.appendChild(option);
  });
  if (keys.includes("action")) {
    [
      ["action::left", "action left"],
      ["action::right", "action right"],
    ].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
  }
  const available = Array.from(select.options).map((option) => option.value);
  currentFeature = available.includes(previous)
    ? previous
    : available.includes("action::left")
      ? "action::left"
      : keys[0] || null;
  select.value = currentFeature || "";
}

function renderEpisodeOptions() {
  const select = $("#episodeSelect");
  select.innerHTML = "";
  currentEpisodes.forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.episode_name;
    option.textContent = `${episode.episode_name.replace(".parquet", "")} · ${episode.length || "-"} rows`;
    option.disabled = !episode.exists;
    select.appendChild(option);
  });
}

function updateEpisodeNavigation() {
  const button = $("#nextEpisodeButton");
  if (!button) return;
  const available = currentEpisodes.filter((episode) => episode.exists);
  const currentIndex = available.findIndex((episode) => episode.episode_name === currentEpisodeName);
  button.disabled = !available.length || currentIndex >= available.length - 1;
}

async function loadDatasetDetails() {
  if (!selectedDataset) return;
  const payload = await fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset.id)}/episodes`);
  currentEpisodes = payload.episodes || [];
  renderEpisodeOptions();
  renderFeatureOptions(selectedDataset.numericKeys || []);
  renderReplayControls();
  renderCompareControls();
  const firstAvailable = currentEpisodes.find((episode) => episode.exists);
  if (firstAvailable) {
    await loadEpisode(firstAvailable.episode_name);
  } else {
    currentEpisodeName = null;
    updateEpisodeNavigation();
  }
}

async function loadEpisode(episodeName) {
  if (!selectedDataset || !episodeName) return;
  stopPlayback();
  currentEpisodeName = episodeName;
  updateEpisodeNavigation();
  $("#episodeSelect").value = episodeName;
  const payload = await fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset.id)}/episode/${encodeURIComponent(episodeName)}/series`);
  currentSeries = payload;
  currentRow = 0;
  $("#rowRange").max = Math.max((payload.rowCount || 1) - 1, 0);
  $("#rowRange").value = "0";
  renderFeatureOptions(payload.numericKeys || []);
  await renderMedia(episodeName);
  await renderCurrentImages();
  renderCurrentRow();
  drawFeatureCanvas();
  renderReplayControls();
  renderReplayPanel();
  drawCompareCanvas();
}

async function renderMedia(episodeName) {
  const grid = $("#videoGrid");
  grid.innerHTML = "";
  const payload = await fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset.id)}/episode/${encodeURIComponent(episodeName)}/media`);
  const videos = payload.videos || [];
  $("#mediaStatus").textContent = `${videos.filter((video) => video.exists).length}/${videos.length} available`;
  if (!videos.length) {
    grid.innerHTML = "<p>No video features in this dataset.</p>";
    return;
  }
  videos.forEach((video) => {
    const card = document.createElement("div");
    card.className = "video-card";
    card.innerHTML = `
      <header>${video.key}${video.exists ? "" : " · missing"}</header>
      ${video.exists ? `<video controls muted src="${video.url}"></video>` : "<div class='readout'>Missing video file</div>"}
    `;
    grid.appendChild(card);
  });
}

async function renderCurrentImages() {
  const grid = $("#imageGrid");
  if (!grid || !selectedDataset || !currentEpisodeName || !currentSeries) return;
  const imageKeys = currentSeries.imageKeys || [];
  if (!imageKeys.length) {
    grid.innerHTML = "<p>No image features in this dataset.</p>";
    return;
  }
  const token = ++imageRequestToken;
  grid.innerHTML = imageKeys.map((key) => `<div class="image-card"><header>${key}</header><div class="readout">Loading row ${currentRow}</div></div>`).join("");
  try {
    const payload = await fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset.id)}/episode/${encodeURIComponent(currentEpisodeName)}/frame/${currentRow}/images`);
    if (token !== imageRequestToken) return;
    grid.innerHTML = "";
    imageKeys.forEach((key) => {
      const image = payload.images?.[key];
      const card = document.createElement("div");
      card.className = "image-card";
      card.innerHTML = `
        <header>${key}</header>
        ${image?.src ? `<img src="${image.src}" alt="${key} row ${currentRow}">` : "<div class='readout'>No image payload</div>"}
      `;
      grid.appendChild(card);
    });
  } catch (error) {
    if (token === imageRequestToken) grid.innerHTML = `<p>${error.message}</p>`;
  }
}

function renderCurrentRow() {
  if (!currentSeries) return;
  const row = currentSeries.rowMeta?.[currentRow] || {};
  $("#rowRange").value = String(currentRow);
  $("#rowReadout").textContent = `row ${currentRow} / ${Math.max((currentSeries.rowCount || 1) - 1, 0)} · frame ${row.frame_index ?? "-"} · task ${row.task_index ?? "-"}`;
  const text = Object.fromEntries(Object.entries(currentSeries.textData || {}).map(([key, values]) => {
    const value = values[currentRow] ?? "";
    if (key.endsWith("action_sequence") && value) {
      try {
        return [key, JSON.parse(value)];
      } catch {
        return [key, value];
      }
    }
    return [key, value];
  }));
  $("#rowJson").textContent = JSON.stringify({ ...row, ...text }, null, 2);
  renderReplayPanel();
}

function currentFeatureSeries() {
  if (!currentSeries || !currentFeature) return [];
  if (currentFeature === "action::left" || currentFeature === "action::right") {
    const rows = currentSeries.featureData?.action || [];
    return rows.map((row) => {
      const midpoint = Math.ceil((row?.length || 0) / 2);
      return currentFeature === "action::left" ? row.slice(0, midpoint) : row.slice(midpoint);
    });
  }
  return currentSeries.featureData?.[currentFeature] || [];
}

function drawFeatureCanvas() {
  const canvas = $("#featureCanvas");
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(canvas.clientWidth || 600, 320);
  const height = 260;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fbfa";
  ctx.fillRect(0, 0, width, height);

  const rows = currentFeatureSeries();
  if (!rows.length) {
    ctx.fillStyle = "#66736f";
    ctx.fillText("No numeric feature selected", 18, 28);
    $("#featureStatus").textContent = "-";
    return;
  }

  const dims = Math.min(rows[0]?.length || 1, 7);
  const values = [];
  for (let dim = 0; dim < dims; dim += 1) {
    rows.forEach((row) => values.push(Number(row?.[dim])));
  }
  const finite = values.filter(Number.isFinite);
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = Math.max(max - min, 1e-6);
  const pad = 24;
  const x = (index) => pad + (index / Math.max(rows.length - 1, 1)) * (width - pad * 2);
  const y = (value) => height - pad - ((value - min) / span) * (height - pad * 2);
  const colors = ["#0f766e", "#2563eb", "#dc2626", "#7c3aed", "#b45309", "#0891b2", "#475569"];

  ctx.strokeStyle = "#dbe4e1";
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  for (let dim = 0; dim < dims; dim += 1) {
    ctx.strokeStyle = colors[dim % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const value = Number(row?.[dim]);
      if (!Number.isFinite(value)) return;
      const px = x(index);
      const py = y(value);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  ctx.strokeStyle = "#17211f";
  ctx.lineWidth = 1;
  const cursorX = x(currentRow);
  ctx.beginPath();
  ctx.moveTo(cursorX, pad);
  ctx.lineTo(cursorX, height - pad);
  ctx.stroke();

  $("#featureStatus").textContent = `${currentFeature} · ${rows.length} rows · ${dims} dims shown`;
}

function replayAvailable(series = currentSeries) {
  const keys = Object.keys(series?.featureData || {});
  return ["action", "expert.action", "action.error"].every((key) => keys.includes(key));
}

function actionNames() {
  const width = currentSeries?.featureData?.action?.[0]?.length || 0;
  return Array.from({ length: width }, (_, index) => `dim ${index}`);
}

function renderReplayControls() {
  const episodeSelect = $("#replayEpisodeSelect");
  const dimSelect = $("#replayDimSelect");
  if (!episodeSelect || !dimSelect) return;
  episodeSelect.innerHTML = "";
  currentEpisodes.filter((episode) => episode.exists).forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.episode_name;
    option.textContent = episode.episode_name.replace(".parquet", "");
    episodeSelect.appendChild(option);
  });
  episodeSelect.value = currentEpisodeName || episodeSelect.value;

  const names = actionNames();
  dimSelect.innerHTML = "";
  if (!names.length) {
    const option = document.createElement("option");
    option.textContent = "No action dims";
    dimSelect.appendChild(option);
    dimSelect.disabled = true;
    return;
  }
  dimSelect.disabled = false;
  replayDim = Math.min(replayDim, names.length - 1);
  names.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = name;
    dimSelect.appendChild(option);
  });
  dimSelect.value = String(replayDim);
}

function renderReplayPanel() {
  const stats = $("#replayStats");
  const body = $("#replayDiffBody");
  if (!stats || !body) return;
  if (!replayAvailable()) {
    renderMetricCards(stats, [
      { label: "Replay fields", value: "missing" },
      { label: "Required", value: "action/error" },
    ]);
    body.innerHTML = "";
    drawReplayCanvas();
    return;
  }

  const l1 = scalarSeries(currentSeries, "action.l1_error");
  const l2 = scalarSeries(currentSeries, "action.l2_error");
  const roundtrip = scalarSeries(currentSeries, "client_timing.roundtrip_ms");
  const replan = scalarSeries(currentSeries, "policy.is_replan");
  renderMetricCards(stats, [
    { label: "Rows", value: formatCount(currentSeries.rowCount) },
    { label: "Mean L1", value: formatNumber(mean(l1)) },
    { label: "Max L1", value: formatNumber(maxValue(l1)) },
    { label: "Mean L2", value: formatNumber(mean(l2)) },
    { label: "Roundtrip ms", value: formatNumber(mean(roundtrip)) },
    { label: "Replans", value: formatCount(replan.filter((value) => value > 0.5).length) },
  ]);

  const action = featureRow(currentSeries, "action", currentRow);
  const expert = featureRow(currentSeries, "expert.action", currentRow);
  const error = featureRow(currentSeries, "action.error", currentRow);
  body.innerHTML = "";
  const dims = Math.max(action.length, expert.length, error.length);
  for (let index = 0; index < dims; index += 1) {
    const row = document.createElement("tr");
    if (index === replayDim) row.className = "selected-row";
    row.innerHTML = `
      <td>${index}</td>
      <td>${formatNumber(action[index])}</td>
      <td>${formatNumber(expert[index])}</td>
      <td>${formatNumber(error[index])}</td>
      <td>${formatNumber(Math.abs(Number(error[index])))}</td>
    `;
    row.addEventListener("click", () => {
      replayDim = index;
      renderReplayControls();
      renderReplayPanel();
      drawReplayCanvas();
    });
    body.appendChild(row);
  }
  drawReplayCanvas();
}

function drawLineChart(canvas, seriesList, cursorIndex = 0, message = "No data") {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(canvas.clientWidth || 600, 320);
  const height = Number(canvas.getAttribute("height")) || 300;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fbfa";
  ctx.fillRect(0, 0, width, height);

  const values = seriesList.flatMap((item) => item.values).filter(Number.isFinite);
  if (!values.length) {
    ctx.fillStyle = "#66736f";
    ctx.fillText(message, 18, 28);
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const pad = 26;
  const longest = Math.max(...seriesList.map((item) => item.values.length), 1);
  const x = (index, length = longest) => pad + (index / Math.max(length - 1, 1)) * (width - pad * 2);
  const y = (value) => height - pad - ((value - min) / span) * (height - pad * 2);

  ctx.strokeStyle = "#dbe4e1";
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  seriesList.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.dashed ? 1.2 : 1.8;
    ctx.setLineDash(item.dashed ? [6, 5] : []);
    ctx.beginPath();
    item.values.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const px = x(index, item.values.length);
      const py = y(value);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.strokeStyle = "#17211f";
  const cursorX = x(Math.min(cursorIndex, longest - 1));
  ctx.beginPath();
  ctx.moveTo(cursorX, pad);
  ctx.lineTo(cursorX, height - pad);
  ctx.stroke();
}

function drawReplayCanvas() {
  const canvas = $("#replayCanvas");
  if (!canvas) return;
  if (!replayAvailable()) {
    drawLineChart(canvas, [], currentRow, "Current dataset is not a replay dataset");
    return;
  }
  const action = currentSeries.featureData.action.map((row) => Number(row?.[replayDim]));
  const expert = currentSeries.featureData["expert.action"].map((row) => Number(row?.[replayDim]));
  const error = currentSeries.featureData["action.error"].map((row) => Number(row?.[replayDim]));
  drawLineChart(canvas, [
    { label: "action", values: action, color: "#0f766e" },
    { label: "expert", values: expert, color: "#2563eb", dashed: true },
    { label: "error", values: error, color: "#dc2626" },
  ], currentRow);
}

async function ensureCompareEpisodes(dataset) {
  if (!dataset) return [];
  const payload = await fetchJson(`/api/datasets/${encodeURIComponent(dataset.id)}/episodes`);
  return payload.episodes || [];
}

function renderCompareControls() {
  const datasetSelect = $("#compareDatasetSelect");
  const episodeSelect = $("#compareEpisodeSelect");
  const featureSelect = $("#compareFeatureSelect");
  if (!datasetSelect || !episodeSelect || !featureSelect) return;

  datasetSelect.innerHTML = "";
  datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.label;
    datasetSelect.appendChild(option);
  });
  if (!compareDataset || !datasets.some((dataset) => dataset.id === compareDataset.id)) {
    compareDataset = selectedDataset || datasets[0] || null;
  }
  if (compareDataset) datasetSelect.value = compareDataset.id;

  episodeSelect.innerHTML = "";
  compareEpisodes.filter((episode) => episode.exists).forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.episode_name;
    option.textContent = episode.episode_name.replace(".parquet", "");
    episodeSelect.appendChild(option);
  });

  featureSelect.innerHTML = "";
  const keys = currentSeries?.numericKeys || selectedDataset?.numericKeys || [];
  keys.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key;
    featureSelect.appendChild(option);
  });
  compareFeature = keys.includes(currentFeature) ? currentFeature : keys[0] || null;
  featureSelect.value = compareFeature || "";
}

async function updateCompareDataset(datasetId) {
  compareDataset = datasets.find((dataset) => dataset.id === datasetId) || null;
  compareEpisodes = await ensureCompareEpisodes(compareDataset);
  compareSeries = null;
  compareEpisodeName = null;
  renderCompareControls();
  drawCompareCanvas();
}

async function loadCompareEpisode() {
  if (!selectedDataset || !compareDataset || !compareFeature) return;
  const episodeName = $("#compareEpisodeSelect").value;
  if (!episodeName) return;
  compareSeries = await fetchJson(`/api/datasets/${encodeURIComponent(compareDataset.id)}/episode/${encodeURIComponent(episodeName)}/series`);
  compareEpisodeName = episodeName;
  drawCompareCanvas();
}

function drawCompareCanvas() {
  const canvas = $("#compareCanvas");
  const stats = $("#compareStats");
  if (!canvas || !stats) return;
  const key = compareFeature || currentFeature;
  const primary = currentSeries?.featureData?.[key] || [];
  const secondary = compareSeries?.featureData?.[key] || [];
  if (!primary.length || !secondary.length) {
    renderMetricCards(stats, [
      { label: "Compare", value: "not loaded" },
      { label: "Feature", value: key || "-" },
    ]);
    drawLineChart(canvas, [], currentRow, "Load a compare trajectory");
    return;
  }

  const dim = 0;
  const primaryValues = primary.map((row) => Number(row?.[dim]));
  const secondaryValues = secondary.map((row) => Number(row?.[dim]));
  const ratio = currentSeries.rowCount > 1 ? currentRow / (currentSeries.rowCount - 1) : 0;
  const compareRow = Math.round(ratio * Math.max(secondary.length - 1, 0));
  const currentDelta = Number(primary[currentRow]?.[dim]) - Number(secondary[compareRow]?.[dim]);
  renderMetricCards(stats, [
    { label: "Feature", value: key },
    { label: "Primary rows", value: formatCount(primary.length) },
    { label: "Compare rows", value: formatCount(secondary.length) },
    { label: "Dim", value: String(dim) },
    { label: "Current diff", value: formatNumber(currentDelta) },
  ]);
  drawLineChart(canvas, [
    { label: "primary", values: primaryValues, color: "#0f766e" },
    { label: "compare", values: secondaryValues, color: "#dc2626", dashed: true },
  ], currentRow);
}

function stepRow(offset) {
  if (!currentSeries) return;
  currentRow = Math.min(Math.max(currentRow + offset, 0), Math.max((currentSeries.rowCount || 1) - 1, 0));
  renderCurrentRow();
  renderCurrentImages();
  drawFeatureCanvas();
  drawReplayCanvas();
  drawCompareCanvas();
}

function stopPlayback() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  $("#playButton").textContent = "Play";
}

function togglePlayback() {
  if (playTimer) {
    stopPlayback();
    return;
  }
  $("#playButton").textContent = "Stop";
  playTimer = setInterval(() => {
    if (!currentSeries || currentRow >= currentSeries.rowCount - 1) {
      stopPlayback();
      return;
    }
    stepRow(1);
  }, 100);
}

async function loadCatalog(preferredDatasetId = null) {
  const payload = await fetchJson("/api/datasets");
  datasets = payload.datasets || [];
  if (preferredDatasetId) {
    selectedDataset = datasets.find((dataset) => dataset.id === preferredDatasetId) || selectedDataset;
  }
  renderDatasetOptions();
  renderDashboard();
  renderDatasetTable();
  if (selectedDataset) {
    if (!compareDataset) compareDataset = selectedDataset;
    compareEpisodes = await ensureCompareEpisodes(compareDataset);
    await loadDatasetDetails();
  }
  renderCompareControls();
}

function trimPayload() {
  if (!selectedDataset) throw new Error("No dataset selected");
  return {
    datasetId: selectedDataset.id,
    outputDataset: $("#trimOutputInput").value.trim() || `${selectedDataset.label}_trimmed`,
    actionKey: $("#trimActionKeyInput").value.trim() || "action",
    stateKey: $("#trimStateKeyInput").value.trim() || "observation.state",
    actionIdleThreshold: Number($("#trimActionThresholdInput").value) || 0.01,
    stateIdleThreshold: Number($("#trimStateThresholdInput").value) || 0.002,
    minEdgeIdleFrames: Number($("#trimMinEdgeInput").value) || 5,
    keepEdgeIdleFrames: Number($("#trimKeepEdgeInput").value) || 0,
    workers: Number($("#trimWorkersInput").value) || 1,
    videoKeys: $("#trimVideoKeysInput").value.trim(),
    dryRun: $("#trimDryRunInput").checked,
    trimVideos: $("#trimVideosInput").checked,
    losslessVideos: $("#trimLosslessVideosInput").checked,
    alsoRequireStateIdle: $("#trimStateIdleInput").checked,
    overwrite: $("#trimOverwriteInput").checked,
  };
}

async function startTrimJob() {
  try {
    const payload = await fetchJson("/api/editor/trim-idle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(trimPayload()),
    });
    $("#latestJobOutput").textContent = JSON.stringify(payload.job, null, 2);
    renderTrimReport(payload.job);
    setView("jobs");
    await loadJobs();
  } catch (error) {
    $("#latestJobOutput").textContent = error.message;
  }
}

function updateConversionControls() {
  const splitMode = $("#conversionSplitModeSelect")?.value || "separate";
  const mergedInput = $("#conversionMergedRepoInput");
  if (mergedInput) mergedInput.disabled = splitMode !== "merged";
}

function renderConversionTasks() {
  const list = $("#conversionTaskList");
  const status = $("#conversionTaskStatus");
  if (!list || !status) return;
  list.innerHTML = "";
  if (!conversionTasks.length) {
    status.textContent = "No tasks discovered";
    list.innerHTML = "<p>Enter a raw HDF5 root and run discovery.</p>";
    return;
  }
  status.textContent = `${conversionTasks.length} task folders`;
  conversionTasks.forEach((task) => {
    const row = document.createElement("label");
    row.className = "task-row";
    const instructions = task.instructionFiles?.length ? task.instructionFiles.join(", ") : "no instruction file";
    row.innerHTML = `
      <input type="checkbox" value="${escapeHtml(task.name)}" checked>
      <div>
        <strong>${escapeHtml(task.name)}</strong>
        <span>${formatCount(task.hdf5Count)} hdf5 · info.json ${task.hasInfo ? "yes" : "no"} · ${escapeHtml(instructions)}</span>
        <code>${escapeHtml(task.relativePath || task.path)}</code>
      </div>
    `;
    list.appendChild(row);
  });
}

function selectedConversionTaskNames() {
  return $$("#conversionTaskList input[type='checkbox']:checked").map((input) => input.value);
}

async function discoverHdf5Tasks() {
  const rawRoot = $("#conversionRawRootInput").value.trim();
  try {
    const payload = await fetchJson("/api/conversion/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rawRoot }),
    });
    conversionTasks = payload.tasks || [];
    renderConversionTasks();
    $("#conversionJobOutput").textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    conversionTasks = [];
    renderConversionTasks();
    $("#conversionJobOutput").textContent = error.message;
  }
}

function conversionPayload() {
  const taskNames = selectedConversionTaskNames();
  if (conversionTasks.length && !taskNames.length) {
    throw new Error("Select at least one HDF5 subtask");
  }
  return {
    rawRoot: $("#conversionRawRootInput").value.trim(),
    repoPrefix: $("#conversionRepoPrefixInput").value.trim() || "emchem_atomic",
    conversionMode: $("#conversionSplitModeSelect").value,
    splitMode: $("#conversionSplitModeSelect").value,
    mergedRepoId: $("#conversionMergedRepoInput").value.trim(),
    outputMode: $("#conversionOutputModeSelect").value,
    taskNames,
  };
}

async function startConversionJob() {
  try {
    const payload = await fetchJson("/api/conversion/hdf5-to-lerobot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(conversionPayload()),
    });
    $("#conversionJobOutput").textContent = JSON.stringify(payload.job, null, 2);
    await loadJobs();
  } catch (error) {
    $("#conversionJobOutput").textContent = error.message;
  }
}

function atomicPayloadBase() {
  if (!selectedDataset) throw new Error("No dataset selected");
  return {
    datasetId: selectedDataset.id,
    stateKey: $("#atomicStateKeyInput").value.trim() || "observation.state",
    actionKey: $("#atomicActionKeyInput").value.trim() || "action",
    jointThreshold: Number($("#atomicJointThresholdInput").value) || 0.035,
    minGap: Number($("#atomicMinGapInput").value) || 20,
    searchRadius: Number($("#atomicSearchRadiusInput").value) || 80,
    subtasks: $("#atomicPromptsInput").value.trim(),
  };
}

function renderAtomicCandidates(payload) {
  const list = $("#atomicKeyframeList");
  if (!list) return;
  const candidates = payload.candidates || [];
  if (!candidates.length) {
    list.innerHTML = "<p>No return-home boundaries found. Increase threshold or search radius.</p>";
    return;
  }
  list.innerHTML = candidates.slice(0, 10).map((item) => `
    <div class="keyframe-row">
      <strong>${item.transitionIndex ? `${escapeHtml(item.fromTask)} -> ${escapeHtml(item.toTask)}` : "candidate"}</strong>
      <span>ep ${item.episodeIndex ?? "-"}</span>
      <span>frame ${item.frame}</span>
      <span>home ${(Number(item.homeRatio) * 100).toFixed(1)}%</span>
      <span>speed ${formatNumber(item.stateVelocity)}</span>
      <span>action ${formatNumber(item.actionNorm)}</span>
      <span>score ${formatNumber(item.score)}</span>
    </div>
  `).join("");
}

async function loadNextEpisode() {
  const available = currentEpisodes.filter((episode) => episode.exists);
  if (!available.length) return;
  const currentIndex = available.findIndex((episode) => episode.episode_name === currentEpisodeName);
  const nextIndex = currentIndex >= 0 ? currentIndex + 1 : 0;
  if (nextIndex >= available.length) return;
  await loadEpisode(available[nextIndex].episode_name);
}

function syncAtomicLabels() {
  $("#atomicLabelsInput").value = JSON.stringify({ segments: atomicSegments }, null, 2);
}

function renderAtomicSegmentEditor() {
  const container = $("#atomicSegmentEditor");
  if (!container) return;
  if (!atomicSegments.length) {
    container.innerHTML = "<p>Suggested segments will appear here for frame-level correction.</p>";
    return;
  }
  container.innerHTML = `
    <div class="table-shell">
      <table class="segment-table">
        <thead>
          <tr><th>Episode</th><th>Subtask</th><th>Start</th><th>End (exclusive)</th><th>Frames</th><th>Prompt</th></tr>
        </thead>
        <tbody>
          ${atomicSegments.map((segment, index) => `
            <tr>
              <td>${segment.episode_index}</td>
              <td>${escapeHtml(segment.label)}</td>
              <td><input type="number" min="0" data-segment-index="${index}" data-field="start" value="${segment.start}"></td>
              <td><input type="number" min="1" data-segment-index="${index}" data-field="end" value="${segment.end}"></td>
              <td>${Math.max(Number(segment.end) - Number(segment.start), 0)}</td>
              <td>${escapeHtml(segment.task)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function updateAtomicSegment(event) {
  const input = event.target.closest("input[data-segment-index]");
  if (!input) return;
  const index = Number(input.dataset.segmentIndex);
  const field = input.dataset.field;
  const value = Math.max(Number(input.value) || 0, 0);
  const segment = atomicSegments[index];
  if (!segment || !["start", "end"].includes(field)) return;
  segment[field] = value;
  if (field === "end" && atomicSegments[index + 1]?.episode_index === segment.episode_index) {
    atomicSegments[index + 1].start = value;
  }
  if (field === "start" && atomicSegments[index - 1]?.episode_index === segment.episode_index) {
    atomicSegments[index - 1].end = value;
  }
  syncAtomicLabels();
  renderAtomicSegmentEditor();
}

async function suggestAtomicKeyframes() {
  try {
    const payload = await fetchJson("/api/editor/suggest-keyframes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(atomicPayloadBase()),
    });
    atomicSegments = payload.segments || [];
    syncAtomicLabels();
    renderAtomicCandidates(payload);
    renderAtomicSegmentEditor();
    $("#atomicSplitOutput").textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    $("#atomicSplitOutput").textContent = error.message;
  }
}

async function startAtomicSplitJob() {
  try {
    const labelsJson = $("#atomicLabelsInput").value.trim();
    if (!labelsJson) throw new Error("Manual labels JSON is required");
    const payload = await fetchJson("/api/editor/split-atomic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasetId: selectedDataset?.id,
        labelsJson,
        repoPrefix: $("#atomicRepoPrefixInput").value.trim() || `${selectedDataset.label}_atomic`,
        outputRoot: $("#atomicOutputRootInput").value.trim() || selectedDataset?.absoluteRoot?.replace(/[\\/][^\\/]+$/, ""),
        splitVideos: $("#atomicSplitVideosInput").checked,
        losslessVideos: $("#atomicLosslessVideosInput").checked,
        videoKeys: $("#atomicVideoKeysInput").value.trim(),
      }),
    });
    $("#atomicSplitOutput").textContent = JSON.stringify(payload.job, null, 2);
    await loadJobs();
  } catch (error) {
    $("#atomicSplitOutput").textContent = error.message;
  }
}

function applyAtomicPresetPrompts(force = false) {
  const input = $("#atomicPromptsInput");
  const preset = $("#atomicTaskPresetSelect")?.value || "solid";
  if (!input) return;
  if (force || !input.value.trim()) {
    input.value = atomicPresetPrompts[preset] || atomicPresetPrompts.solid;
  }
}

function atomicBatchPayload() {
  const sourceRoot = $("#atomicBatchSourceRootInput").value.trim();
  if (!sourceRoot) throw new Error("Multi dataset source root is required");
  return {
    sourceRoot,
    outputRoot: $("#atomicOutputRootInput").value.trim() || sourceRoot,
    repoPrefix: $("#atomicRepoPrefixInput").value.trim() || sourceRoot.replace(/[\\/]$/, "").split(/[\\/]/).pop() || "atomic",
    taskPreset: $("#atomicTaskPresetSelect").value,
    subtasks: $("#atomicPromptsInput").value.trim(),
    stateKey: $("#atomicStateKeyInput").value.trim() || "observation.state",
    actionKey: $("#atomicActionKeyInput").value.trim() || "action",
    jointThreshold: Number($("#atomicJointThresholdInput").value) || 0.035,
    minHomeRatio: 0.65,
    fallbackHomeRatio: 0.45,
    edgeHomeRatio: 0.65,
    searchRadius: Number($("#atomicSearchRadiusInput").value) || 80,
    minGap: Number($("#atomicMinGapInput").value) || 20,
    minEdgeHomeFrames: 5,
    keepEdgeHomeFrames: 0,
    minSegmentFrames: 20,
    edgeStateVelocityThreshold: 0.02,
    splitVideos: $("#atomicSplitVideosInput").checked,
    losslessVideos: $("#atomicLosslessVideosInput").checked,
    videoKeys: $("#atomicVideoKeysInput").value.trim(),
    skipFailedEpisodes: true,
  };
}

async function startAtomicBatchSplitJob() {
  try {
    applyAtomicPresetPrompts(false);
    const payload = await fetchJson("/api/editor/batch-split-atomic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(atomicBatchPayload()),
    });
    $("#atomicSplitOutput").textContent = JSON.stringify(payload.job, null, 2);
    await loadJobs();
  } catch (error) {
    $("#atomicSplitOutput").textContent = error.message;
  }
}

async function addDatasetPath() {
  const path = $("#datasetPathInput").value.trim();
  if (!path) return;
  try {
    const payload = await fetchJson("/api/datasets/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const saved = savedDatasetPaths();
    localStorage.setItem("robodataDatasetPaths", JSON.stringify(Array.from(new Set([...saved, path]))));
    $("#datasetPathInput").value = "";
    await loadCatalog(payload.dataset.id);
  } catch (error) {
    $("#datasetPathInput").value = path;
    $("#dashboardDatasetList").textContent = error.message;
  }
}

async function restoreDatasetPaths() {
  const saved = savedDatasetPaths();
  await Promise.all(saved.map((path) => fetchJson("/api/datasets/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).catch(() => null)));
}

async function renderTrimReport(job) {
  const status = $("#trimReportStatus");
  const grid = $("#trimSummaryGrid");
  const output = $("#trimReportOutput");
  if (!status || !grid || !output || !job) return;
  let report = job.report || null;
  if (!report && ["succeeded", "failed"].includes(job.status)) {
    try {
      report = await fetchJson(`/api/jobs/${encodeURIComponent(job.id)}/report`);
    } catch {
      report = null;
    }
  }
  if (!report?.summary) {
    status.textContent = `${job.status} · report pending`;
    renderMetricCards(grid, [
      { label: "Job", value: job.status },
      { label: "Return code", value: job.returnCode ?? "-" },
    ]);
    output.textContent = [
      `command: ${(job.command || []).join(" ")}`,
      "",
      "stdout:",
      job.stdout || "",
      "",
      "stderr:",
      job.stderr || "",
    ].join("\n");
    return;
  }
  const summary = report.summary;
  status.textContent = report.reportDir || "report loaded";
  renderMetricCards(grid, [
    { label: "Episodes", value: formatCount(summary.episodes) },
    { label: "Original", value: formatCount(summary.original_frames) },
    { label: "Kept", value: formatCount(summary.kept_frames) },
    { label: "Trimmed", value: formatCount(summary.trimmed_frames) },
    { label: "Ratio", value: `${formatNumber(Number(summary.trimmed_ratio) * 100)}%` },
    { label: "Videos", value: formatCount(summary.videos_written) },
  ]);
  output.textContent = report.markdown || JSON.stringify(summary, null, 2);
}

async function loadJobs() {
  const payload = await fetchJson("/api/jobs");
  const list = $("#jobsList");
  list.innerHTML = "";
  const jobs = payload.jobs || [];
  if (!jobs.length) {
    list.innerHTML = "<p>No jobs yet.</p>";
    return;
  }
  jobs.forEach((job) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <strong>${job.id}</strong>
      <span>${job.kind}</span>
      <span>${job.status}</span>
      <span>${job.returnCode ?? "-"}</span>
      <button type="button">Details</button>
    `;
    row.querySelector("button").addEventListener("click", () => {
      if (job.kind === "hdf5-convert") {
        $("#conversionJobOutput").textContent = JSON.stringify(job, null, 2);
        setView("editor");
      } else if (job.kind === "split-atomic" || job.kind === "batch-split-atomic") {
        $("#atomicSplitOutput").textContent = JSON.stringify(job, null, 2);
        setView("editor");
      } else {
        $("#latestJobOutput").textContent = JSON.stringify(job, null, 2);
        renderTrimReport(job);
        setView("editor");
      }
    });
    list.appendChild(row);
  });
}

async function previewTraining() {
  const config = encodeURIComponent($("#trainConfigInput").value.trim());
  const steps = encodeURIComponent($("#trainStepsInput").value.trim());
  const dataset = encodeURIComponent(selectedDataset?.id || "");
  const payload = await fetchJson(`/api/training/preview-command?config=${config}&dataset=${dataset}&steps=${steps}`);
  $("#trainCommandOutput").textContent = payload.command;
}

async function previewInference() {
  const config = encodeURIComponent($("#inferConfigInput").value.trim());
  const checkpoint = encodeURIComponent($("#inferCheckpointInput").value.trim());
  const port = encodeURIComponent($("#inferPortInput").value.trim());
  const payload = await fetchJson(`/api/inference/preview-command?config=${config}&checkpoint=${checkpoint}&port=${port}`);
  $("#inferCommandOutput").textContent = payload.command;
}

$$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
$("#refreshButton").addEventListener("click", loadCatalog);
$("#addDatasetButton").addEventListener("click", addDatasetPath);
$("#datasetPathInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") addDatasetPath();
});
$("#globalDatasetSelect").addEventListener("change", async (event) => {
  selectedDataset = datasets.find((dataset) => dataset.id === event.target.value) || null;
  $("#trimOutputInput").placeholder = selectedDataset ? `${selectedDataset.absoluteRoot}_trimmed` : "";
  await loadDatasetDetails();
});
$("#episodeSelect").addEventListener("change", (event) => loadEpisode(event.target.value));
$("#featureSelect").addEventListener("change", (event) => {
  currentFeature = event.target.value;
  drawFeatureCanvas();
});
$("#rowRange").addEventListener("input", (event) => {
  currentRow = Number(event.target.value) || 0;
  renderCurrentRow();
  renderCurrentImages();
  drawFeatureCanvas();
  drawReplayCanvas();
  drawCompareCanvas();
});
$("#prevRowButton").addEventListener("click", () => stepRow(-1));
$("#nextEpisodeButton").addEventListener("click", loadNextEpisode);
$("#nextRowButton").addEventListener("click", () => stepRow(1));
$("#playButton").addEventListener("click", togglePlayback);
$("#startTrimButton").addEventListener("click", startTrimJob);
$("#refreshJobsButton").addEventListener("click", loadJobs);
$("#jobsRefreshButton").addEventListener("click", loadJobs);
$("#discoverHdf5Button").addEventListener("click", discoverHdf5Tasks);
$("#startConversionButton").addEventListener("click", startConversionJob);
$("#refreshConversionJobsButton").addEventListener("click", loadJobs);
$("#conversionSplitModeSelect").addEventListener("change", updateConversionControls);
$("#suggestAtomicButton").addEventListener("click", suggestAtomicKeyframes);
$("#startAtomicSplitButton").addEventListener("click", startAtomicSplitJob);
$("#startAtomicBatchSplitButton").addEventListener("click", startAtomicBatchSplitJob);
$("#atomicTaskPresetSelect").addEventListener("change", () => applyAtomicPresetPrompts(true));
$("#atomicSegmentEditor").addEventListener("change", updateAtomicSegment);
$("#atomicLabelsInput").addEventListener("change", () => {
  try {
    atomicSegments = JSON.parse($("#atomicLabelsInput").value || "{}").segments || [];
    renderAtomicSegmentEditor();
  } catch (error) {
    $("#atomicSplitOutput").textContent = `Invalid labels JSON: ${error.message}`;
  }
});
$("#previewTrainButton").addEventListener("click", previewTraining);
$("#previewInferButton").addEventListener("click", previewInference);
$("#replayEpisodeSelect").addEventListener("change", (event) => loadEpisode(event.target.value));
$("#replayDimSelect").addEventListener("change", (event) => {
  replayDim = Number(event.target.value) || 0;
  renderReplayPanel();
});
$("#compareDatasetSelect").addEventListener("change", (event) => updateCompareDataset(event.target.value));
$("#compareFeatureSelect").addEventListener("change", (event) => {
  compareFeature = event.target.value;
  drawCompareCanvas();
});
$("#loadCompareButton").addEventListener("click", loadCompareEpisode);
$("#clearCompareButton").addEventListener("click", () => {
  compareSeries = null;
  compareEpisodeName = null;
  drawCompareCanvas();
});
window.addEventListener("resize", () => {
  drawFeatureCanvas();
  drawReplayCanvas();
  drawCompareCanvas();
});
window.addEventListener("beforeunload", stopPlayback);

restoreDatasetPaths().then(() => loadCatalog()).catch((error) => {
  $("#dashboardDatasetList").textContent = error.message;
});
updateConversionControls();
renderConversionTasks();
applyAtomicPresetPrompts(false);
renderAtomicSegmentEditor();
