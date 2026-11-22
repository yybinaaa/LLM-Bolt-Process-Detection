"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("judge-form");
const fileInput = byId("image-file");
const imageArea = byId("image-area");
const chooseImage = byId("choose-image");
const imagePreview = byId("image-preview");
const preview = byId("preview");
const imageActions = byId("image-actions");
const imageError = byId("image-error");
const replaceImage = byId("replace-image");
const removeImage = byId("remove-image");
const prompt = byId("prompt");
const example = byId("example");
const submit = byId("submit");
const connection = byId("connection");
const modelSelect = byId("model-select");
const modelStatus = byId("model-status");
const apiKeyControl = byId("api-key-control");
const apiKeyInput = byId("api-key");
const imagePaused = byId("image-paused");
const imageMode = byId("image-mode");
const destination = byId("destination");
const result = byId("result");
const resultStatus = byId("result-status");
const answer = byId("answer");
const answerModel = byId("answer-model");
const copy = byId("copy");

const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const SOURCE_LABELS = {ollama: "本机 · Ollama", vllm: "本机 · vLLM", api: "远程 API"};
const STATUS_LABELS = {key: "需密钥", offline: "服务未启动", inactive: "模型未加载"};

let models = [];
const sessionKeys = new Map();
let refreshing = false;
let selectedFile = null;
let previewUrl = null;
let imageVersion = 0;
let decoding = false;
let busy = false;

function currentModel() {
  return models.find((model) => model.id === modelSelect.value);
}

function canAttachImage() {
  return currentModel()?.vision ?? true;
}

function updateControls() {
  const model = currentModel();
  const canAttach = canAttachImage();
  const modelReady = model?.status === "ready" || (model?.status === "key" && sessionKeys.has(model.key_env));
  submit.disabled = busy || decoding || !modelReady || (canAttach && !selectedFile) || !prompt.value.trim();
  submit.textContent = busy ? "判断中…" : model?.source === "api" ? "发送给 API" : "开始判断";
  modelSelect.disabled = busy || refreshing || !models.length;
  apiKeyInput.disabled = busy;
  chooseImage.hidden = Boolean(selectedFile) || !canAttach;
  chooseImage.disabled = busy || !canAttach;
  replaceImage.disabled = busy || !canAttach;
  removeImage.disabled = busy;
  imagePaused.hidden = canAttach || Boolean(selectedFile);
  imageMode.hidden = canAttach || !selectedFile;
  prompt.readOnly = busy;
  example.disabled = busy || !canAttach;
}

function showImageError(message = "") {
  imageError.textContent = message;
  imageError.hidden = !message;
}

function setResult(message = "提交后，回答会显示在这里。", isError = false) {
  resultStatus.textContent = message;
  resultStatus.classList.toggle("is-error", isError);
  resultStatus.hidden = !message;
  answer.hidden = true;
  answer.textContent = "";
  answerModel.hidden = true;
  copy.hidden = true;
  copy.textContent = "复制回答";
}

async function selectImage(file) {
  if (busy || !file || !canAttachImage()) return;
  showImageError();
  if (!IMAGE_TYPES.includes(file.type)) {
    showImageError("请选择 JPG、PNG 或 WebP 图片。");
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showImageError("图片大小请控制在 20 MB 以内。");
    return;
  }
  const version = ++imageVersion;
  const url = URL.createObjectURL(file);
  decoding = true;
  updateControls();
  let decoded = false;
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    decoded = true;
  } catch {
    if (version === imageVersion) showImageError("无法读取这张图片，请换一张再试。");
  }
  if (!decoded || version !== imageVersion) {
    URL.revokeObjectURL(url);
  } else {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = url;
    selectedFile = file;
    preview.src = url;
    imagePreview.hidden = false;
    imageActions.hidden = false;
    byId("file-name").textContent = file.name;
    byId("file-name").title = file.name;
    setResult();
  }
  if (version === imageVersion) {
    decoding = false;
    updateControls();
  }
}

function clearImage() {
  imageVersion++;
  decoding = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  preview.removeAttribute("src");
  imagePreview.hidden = true;
  imageActions.hidden = true;
  showImageError();
  setResult();
  updateControls();
  (canAttachImage() ? chooseImage : prompt).focus();
}

function showModel() {
  const model = currentModel();
  const needsKey = model?.status === "key";
  apiKeyControl.hidden = !needsKey;
  apiKeyInput.value = needsKey ? sessionKeys.get(model.key_env) ?? "" : "";
  byId("key-help").textContent = needsKey ? `密钥只在当前页面暂用，也可配置 ${model.key_env}。` : "";
  modelStatus.classList.remove("is-error");
  modelStatus.textContent = model ? [SOURCE_LABELS[model.source], model.vision ? "图文" : "文字", STATUS_LABELS[model.status]].filter(Boolean).join(" · ") : "暂无可选模型。";
  destination.hidden = model?.source !== "api";
  if (model?.source === "api") {
    destination.textContent = `提交时${model.vision ? "图片和提示词" : "提示词"}将发送至 ${model.host}。`;
  }
  showImageError();
  updateControls();
}

function populateModels(config) {
  const previous = modelSelect.value;
  models = config.models;
  modelSelect.replaceChildren();
  for (const [source, label] of Object.entries(SOURCE_LABELS)) {
    const options = models.filter((model) => model.source === source);
    if (!options.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const model of options) {
      const label = [model.name, model.vision ? "图文" : "文字", STATUS_LABELS[model.status]].filter(Boolean).join(" · ");
      group.append(new Option(label, model.id));
    }
    modelSelect.append(group);
  }
  const preferred = models.find((model) => model.id === previous)
    ?? models.find((model) => model.id === config.default && model.status === "ready")
    ?? models.find((model) => model.source !== "api" && model.status === "ready")
    ?? models[0];
  if (preferred) modelSelect.value = preferred.id;
  else modelSelect.add(new Option("暂无可选模型", ""));
  showModel();
}

async function refreshModels() {
  if (refreshing) return;
  refreshing = true;
  connection.disabled = true;
  byId("connection-label").textContent = "正在刷新…";
  updateControls();
  try {
    const response = await fetch("/api/models");
    const config = await response.json();
    if (!response.ok) throw new Error(config.error);
    populateModels(config);
  } catch {
    modelStatus.textContent = "模型列表读取失败，请点击「刷新模型」重试。";
    modelStatus.classList.add("is-error");
    if (!models.length) modelSelect.replaceChildren(new Option("未读取模型列表", ""));
  } finally {
    refreshing = false;
    connection.disabled = false;
    byId("connection-label").textContent = "刷新模型";
    updateControls();
  }
}

function encodeImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result.split(",")[1]));
    reader.addEventListener("error", () => reject(new Error("无法读取图片，请重新选择。")));
    reader.readAsDataURL(file);
  });
}

chooseImage.addEventListener("click", () => fileInput.click());
replaceImage.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectImage(fileInput.files[0]);
  fileInput.value = "";
});
removeImage.addEventListener("click", clearImage);

window.addEventListener("dragover", (event) => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});
imageArea.addEventListener("dragover", (event) => {
  if (busy || !canAttachImage() || !event.dataTransfer.types.includes("Files")) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  imageArea.classList.add("dragover");
});
imageArea.addEventListener("dragleave", (event) => {
  if (!imageArea.contains(event.relatedTarget)) imageArea.classList.remove("dragover");
});
imageArea.addEventListener("drop", (event) => {
  event.preventDefault();
  imageArea.classList.remove("dragover");
  if (busy || !canAttachImage()) return;
  if (event.dataTransfer.files.length > 1) {
    showImageError("每次放入一张图片。");
    return;
  }
  selectImage(event.dataTransfer.files[0]);
});
window.addEventListener("paste", (event) => {
  const file = Array.from(event.clipboardData?.files ?? []).find((item) => item.type.startsWith("image/"));
  if (busy || !file || !canAttachImage()) return;
  event.preventDefault();
  selectImage(file);
});

prompt.addEventListener("input", updateControls);
modelSelect.addEventListener("change", showModel);
apiKeyInput.addEventListener("input", () => {
  const model = currentModel();
  const key = apiKeyInput.value.trim();
  if (key) sessionKeys.set(model.key_env, key);
  else sessionKeys.delete(model.key_env);
  updateControls();
});
example.addEventListener("click", () => {
  prompt.value = "请判断图中的扳手是否接触螺栓。先回答“接触”“未接触”或“看不清”，再用一句话说明画面依据。";
  updateControls();
  prompt.focus();
});
form.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !event.isComposing && !submit.disabled) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const model = currentModel();
  busy = true;
  updateControls();
  result.setAttribute("aria-busy", "true");
  setResult(`正在等 ${model.name} 回答…`);
  try {
    const payload = {model: model.id, prompt: prompt.value.trim()};
    if (model.vision) payload.image = await encodeImage(selectedFile);
    if (model.status === "key") payload.api_key = sessionKeys.get(model.key_env);
    const response = await fetch("/api/judge", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "判断失败，请重试。");
    setResult("");
    answerModel.textContent = `${SOURCE_LABELS[data.source]} · ${data.model}`;
    answerModel.hidden = false;
    answer.textContent = data.answer;
    answer.hidden = false;
    copy.hidden = false;
  } catch (error) {
    setResult(error instanceof TypeError ? "本地页面连接中断，请刷新后重试。" : error.message, true);
  } finally {
    busy = false;
    result.removeAttribute("aria-busy");
    updateControls();
  }
});

copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(answer.textContent);
    copy.textContent = "已复制";
  } catch {
    copy.textContent = "请手动选择文字复制";
  }
});
connection.addEventListener("click", refreshModels);
refreshModels();
