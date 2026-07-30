const referenceInput = document.querySelector("#reference-input");
const referenceName = document.querySelector("#reference-name");
const startButton = document.querySelector("#start-button");
const finishButton = document.querySelector("#finish-button");
const video = document.querySelector("#video");
const canvas = document.querySelector("#capture-canvas");
const cameraStatus = document.querySelector("#camera-status");
const challenges = document.querySelector("#challenges");
const frameCount = document.querySelector("#frame-count");
const placeholder = document.querySelector("#video-placeholder");
const resultCard = document.querySelector("#result-card");

let stream;
let sessionId;
let frameTimer;
let inFlight = false;

referenceInput.addEventListener("change", () => {
  const [file] = referenceInput.files;
  referenceName.textContent = file ? file.name : "No portrait selected";
  startButton.disabled = !file;
});

startButton.addEventListener("click", startSession);
finishButton.addEventListener("click", finishSession);

async function startSession() {
  const [reference] = referenceInput.files;
  if (!reference) return;
  setBusy(startButton, true, "Starting…");
  resultCard.classList.add("hidden");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    placeholder.classList.add("hidden");
    setStatus("Camera ready", "ready");

    const data = new FormData();
    data.append("reference", reference);
    const response = await fetch("/api/sessions", { method: "POST", body: data });
    const session = await response.json();
    if (!response.ok) throw new Error(session.detail || "Could not start session");
    sessionId = session.session_id;
    renderChallenges(session.challenges);
    finishButton.disabled = false;
    setStatus("Analysing", "live");
    frameTimer = window.setInterval(captureFrame, 750);
  } catch (error) {
    stopCamera();
    setStatus("Error", "error");
    showError(error.message);
  } finally {
    setBusy(startButton, false, "Start camera session");
  }
}

async function captureFrame() {
  if (!sessionId || inFlight || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  inFlight = true;
  try {
    const blob = await videoBlob();
    const data = new FormData();
    data.append("frame", blob, "webcam.jpg");
    const response = await fetch(`/api/sessions/${sessionId}/frames`, { method: "POST", body: data });
    const progress = await response.json();
    if (!response.ok) throw new Error(progress.detail || "Frame processing failed");
    frameCount.textContent = `${progress.frames_seen} frames analysed`;
    markCompleted(progress.completed_challenges);
    if (progress.warnings.length) setStatus("Suspicious PAD signal", "warning");
  } catch (error) {
    clearInterval(frameTimer);
    setStatus("Frame error", "error");
    showError(error.message);
  } finally {
    inFlight = false;
  }
}

async function finishSession() {
  if (!sessionId) return;
  clearInterval(frameTimer);
  finishButton.disabled = true;
  setBusy(finishButton, true, "Verifying…");
  try {
    const response = await fetch(`/api/sessions/${sessionId}/finish`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not finish session");
    renderResult(result);
    setStatus(result.matched ? "Complete" : "Review result", result.matched ? "complete" : "warning");
  } catch (error) {
    setStatus("Error", "error");
    showError(error.message);
  } finally {
    sessionId = undefined;
    stopCamera();
    setBusy(finishButton, false, "Finish and verify");
  }
}

function videoBlob() {
  const scale = Math.min(1, 640 / video.videoWidth);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not capture webcam frame")), "image/jpeg", 0.82));
}

function renderChallenges(items) {
  challenges.innerHTML = "";
  items.forEach((item, index) => {
    const node = document.createElement("li");
    node.dataset.challenge = item;
    node.innerHTML = `<span>${index + 1}</span>${item.replaceAll("_", " ")}`;
    challenges.append(node);
  });
}

function markCompleted(completed) {
  completed.forEach(item => document.querySelector(`[data-challenge="${item}"]`)?.classList.add("complete"));
}

function renderResult(result) {
  const liveness = result.liveness;
  resultCard.classList.remove("hidden");
  document.querySelector("#result-title").textContent = result.matched ? "Verification passed" : "Verification needs review";
  const badge = document.querySelector("#result-badge");
  badge.textContent = result.matched ? "MATCH" : "NO MATCH";
  badge.className = `result-badge ${result.matched ? "pass" : "fail"}`;
  document.querySelector("#metric-match").textContent = result.matched ? "Yes" : "No";
  document.querySelector("#metric-similarity").textContent = result.similarity == null ? "—" : result.similarity.toFixed(3);
  document.querySelector("#metric-confidence").textContent = `${Math.round(liveness.confidence * 100)}%`;
  const notes = [...liveness.warnings, ...liveness.reasons, ...result.reasons];
  document.querySelector("#result-details").innerHTML = notes.length
    ? `<h3>Signals</h3><ul>${notes.map(note => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
    : "<p>No warning signals reported.</p>";
}

function showError(message) {
  resultCard.classList.remove("hidden");
  document.querySelector("#result-title").textContent = "Session error";
  document.querySelector("#result-details").innerHTML = `<p>${escapeHtml(message)}</p>`;
}

function setStatus(label, kind) {
  cameraStatus.textContent = label;
  cameraStatus.className = `status ${kind}`;
}

function setBusy(button, busy, label) {
  button.disabled = busy || (button === startButton && !referenceInput.files.length);
  button.textContent = label;
}

function stopCamera() {
  clearInterval(frameTimer);
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.srcObject = null;
  placeholder.classList.remove("hidden");
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value;
  return element.innerHTML;
}
