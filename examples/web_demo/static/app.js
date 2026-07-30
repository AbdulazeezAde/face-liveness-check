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
let sessionToken;
let socket;
let frameTimer;
let inFlight = false;
let finishRequest;
let streamReady;
let expectedSocketClose = false;

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
    sessionToken = session.session_token;
    renderChallenges(session.challenges);
    await connectStream(session.stream_path, session.session_token);
    finishButton.disabled = false;
    setStatus("Secure stream active", "live");
    frameTimer = window.setInterval(captureFrame, 750);
  } catch (error) {
    closeSessionConnection();
    stopCamera();
    setStatus("Error", "error");
    showError(error.message);
  } finally {
    setBusy(startButton, false, "Start camera session");
  }
}

function connectStream(streamPath, token) {
  return new Promise((resolve, reject) => {
    streamReady = { resolve, reject };
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${window.location.host}${streamPath}`);
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => socket.send(JSON.stringify({ type: "authenticate", session_token: token })), { once: true });
    socket.addEventListener("error", () => {
      const error = new Error("Could not open the local WebSocket stream");
      streamReady?.reject(error);
      streamReady = undefined;
    }, { once: true });
    socket.addEventListener("message", event => {
      try {
        handleStreamMessage(JSON.parse(event.data));
      } catch {
        handleStreamError(new Error("The local server returned an invalid stream message"));
      }
    });
    socket.addEventListener("close", event => {
      socket = undefined;
      inFlight = false;
      if (!expectedSocketClose && sessionToken) {
        handleStreamError(new Error(`Stream closed (${event.code})`));
      }
    });
  });
}

async function captureFrame() {
  if (!sessionToken || inFlight || socket?.readyState !== WebSocket.OPEN || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  inFlight = true;
  try {
    socket.send(await (await videoBlob()).arrayBuffer());
  } catch (error) {
    inFlight = false;
    handleStreamError(error);
  }
}

async function finishSession() {
  if (!sessionToken || socket?.readyState !== WebSocket.OPEN) return;
  clearInterval(frameTimer);
  finishButton.disabled = true;
  setBusy(finishButton, true, "Verifying…");
  try {
    const result = await requestFinish();
    renderResult(result);
    setStatus(result.matched ? "Complete" : "Review result", result.matched ? "complete" : "warning");
  } catch (error) {
    setStatus("Error", "error");
    showError(error.message);
  } finally {
    expectedSocketClose = true;
    sessionToken = undefined;
    stopCamera();
    setBusy(finishButton, false, "Finish and verify");
    finishButton.disabled = true;
  }
}

function requestFinish() {
  return new Promise((resolve, reject) => {
    finishRequest = { resolve, reject };
    socket.send(JSON.stringify({ type: "finish" }));
  });
}

function handleStreamMessage(message) {
  if (message.type === "authenticate") return;
  if (message.type === "ready") {
    streamReady?.resolve();
    streamReady = undefined;
    return;
  }
  if (message.type === "progress") {
    inFlight = false;
    frameCount.textContent = `${message.frames_seen} frames analysed`;
    markCompleted(message.completed_challenges);
    if (message.warnings.length) setStatus("Suspicious PAD signal", "warning");
    return;
  }
  if (message.type === "warning") {
    inFlight = false;
    setStatus("Frame rate limited", "warning");
    return;
  }
  if (message.type === "result") {
    inFlight = false;
    expectedSocketClose = true;
    finishRequest?.resolve(message.result);
    finishRequest = undefined;
    return;
  }
  if (message.type === "error") {
    const error = new Error(message.detail || "Stream processing failed");
    streamReady?.reject(error);
    streamReady = undefined;
    handleStreamError(error);
  }
}

function handleStreamError(error) {
  clearInterval(frameTimer);
  inFlight = false;
  finishRequest?.reject(error);
  finishRequest = undefined;
  streamReady?.reject(error);
  streamReady = undefined;
  expectedSocketClose = true;
  closeSessionConnection();
  stopCamera();
  finishButton.disabled = true;
  setStatus("Stream error", "error");
  showError(error.message);
}

function closeSessionConnection() {
  clearInterval(frameTimer);
  if (socket?.readyState === WebSocket.OPEN && sessionToken) {
    expectedSocketClose = true;
    socket.send(JSON.stringify({ type: "cancel" }));
  }
  socket?.close();
  socket = undefined;
  sessionToken = undefined;
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
