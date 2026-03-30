const audioInput = document.getElementById("audio-upload");
const audioPlayer = document.getElementById("audio-player");
const fileMeta = document.getElementById("file-meta");
const dropzone = document.getElementById("dropzone");
const analyzeButton = document.getElementById("analyze-button");
const resultPlaceholder = document.getElementById("result-placeholder");
const waveformCanvas = document.getElementById("waveform-canvas");
const spectrogramCanvas = document.getElementById("spectrogram-canvas");
const waveCtx = waveformCanvas.getContext("2d");
const specCtx = spectrogramCanvas.getContext("2d");

let audioContext;
let lastAudioBuffer = null;
let lastObjectUrl = null;
let animationFrameId = null;
let selectedFile = null;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "Unknown size";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "Unknown duration";
  }

  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");

  return `${mins}:${secs}`;
}

function resetCanvas(ctx, canvas, fill) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawEmptyStates() {
  resetCanvas(waveCtx, waveformCanvas, "#102331");
  waveCtx.fillStyle = "rgba(240, 240, 240, 0.75)";
  waveCtx.font = "20px 'IBM Plex Mono', monospace";
  waveCtx.fillText("Upload audio to preview waveform", 32, 46);

  resetCanvas(specCtx, spectrogramCanvas, "#0c1b26");
  specCtx.fillStyle = "rgba(240, 240, 240, 0.75)";
  specCtx.font = "20px 'IBM Plex Mono', monospace";
  specCtx.fillText("Spectrogram preview appears here", 32, 46);
}

function drawWaveform(audioBuffer) {
  resetCanvas(waveCtx, waveformCanvas, "#102331");
  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / waveformCanvas.width);
  const amp = waveformCanvas.height / 2;

  waveCtx.lineWidth = 2;
  waveCtx.strokeStyle = "#f3b17f";
  waveCtx.beginPath();

  for (let i = 0; i < waveformCanvas.width; i += 1) {
    let min = 1;
    let max = -1;

    for (let j = 0; j < step; j += 1) {
      const datum = data[(i * step) + j];
      if (datum < min) {
        min = datum;
      }
      if (datum > max) {
        max = datum;
      }
    }

    waveCtx.moveTo(i, (1 + min) * amp);
    waveCtx.lineTo(i, (1 + max) * amp);
  }

  waveCtx.stroke();
}

function drawStaticSpectrogram(audioBuffer) {
  resetCanvas(specCtx, spectrogramCanvas, "#0c1b26");

  const channelData = audioBuffer.getChannelData(0);
  const columns = 120;
  const rows = 64;
  const sliceSize = Math.max(1, Math.floor(channelData.length / columns));

  for (let x = 0; x < columns; x += 1) {
    let energy = 0;

    for (let i = 0; i < sliceSize; i += 1) {
      const sample = channelData[(x * sliceSize) + i] || 0;
      energy += Math.abs(sample);
    }

    const normalizedEnergy = Math.min(1, energy / sliceSize * 3.4);

    for (let y = 0; y < rows; y += 1) {
      const frequencyBias = 1 - (y / rows);
      const intensity = Math.max(0, normalizedEnergy * (0.45 + frequencyBias * 0.8) - Math.random() * 0.12);
      const hue = 24 + (frequencyBias * 36);
      const lightness = 12 + intensity * 58;

      specCtx.fillStyle = `hsl(${hue} 90% ${lightness}%)`;
      specCtx.fillRect(
        x * (spectrogramCanvas.width / columns),
        y * (spectrogramCanvas.height / rows),
        Math.ceil(spectrogramCanvas.width / columns),
        Math.ceil(spectrogramCanvas.height / rows)
      );
    }
  }
}

function showMetadata(file, durationText) {
  fileMeta.innerHTML = `
    <strong>${file.name}</strong>
    <p>Type: ${file.type || "Unknown"}<br>Size: ${formatBytes(file.size)}<br>Duration: ${durationText}</p>
  `;
}

async function decodeAudioFile(file) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  const buffer = await file.arrayBuffer();
  return audioContext.decodeAudioData(buffer.slice(0));
}

async function handleFile(file) {
  if (!file) {
    return;
  }

  selectedFile = file;

  if (lastObjectUrl) {
    URL.revokeObjectURL(lastObjectUrl);
  }

  resultPlaceholder.innerHTML = `
    <strong>File loaded</strong>
    <p>The browser has prepared preview data. Press Analyze Audio to send the sample to the CNN backend.</p>
  `;

  const objectUrl = URL.createObjectURL(file);
  lastObjectUrl = objectUrl;
  audioPlayer.src = objectUrl;
  audioPlayer.classList.add("fade-in");

  try {
    const audioBuffer = await decodeAudioFile(file);
    lastAudioBuffer = audioBuffer;
    const durationText = formatDuration(audioBuffer.duration);

    showMetadata(file, durationText);
    drawWaveform(audioBuffer);
    drawStaticSpectrogram(audioBuffer);
  } catch (error) {
    lastAudioBuffer = null;
    fileMeta.innerHTML = `
      <strong>${file.name}</strong>
      <p>The browser could not decode this file for preview. You can still wire it to a backend upload flow.</p>
    `;
    drawEmptyStates();
  }
}

function renderPrediction(result) {
  const isSynthetic = result.label !== "Human";
  const verdictLabel = isSynthetic ? "Synthetic voice detected" : "Human voice detected";
  const verdictClass = isSynthetic ? "synthetic" : "human";

  const probabilityLines = Object.entries(result.probabilities)
    .map(([label, score]) => {
      const clampedScore = Math.max(0, Math.min(100, Number(score) || 0));
      return `
        <div class="probability-card">
          <span class="probability-label-row">
            <strong>${label}</strong>
            <span>${clampedScore.toFixed(2)}%</span>
          </span>
          <div class="probability-bar-track">
            <span class="probability-bar-fill" style="width: ${clampedScore}%;"></span>
          </div>
        </div>
      `;
    })
    .join("");

  resultPlaceholder.innerHTML = `
    <span class="result-pill ${verdictClass}">${verdictLabel}</span>
    <strong class="result-title">${result.label}</strong>
    <p>Model confidence: ${result.confidence}%</p>
    <div class="probability-list">${probabilityLines}</div>
  `;
}

async function analyzeAudio() {
  if (!selectedFile || !lastAudioBuffer) {
    resultPlaceholder.innerHTML = `
      <strong>No audio ready</strong>
      <p>Please upload an audio file first so the interface has something to analyze.</p>
    `;
    return;
  }

  cancelAnimationFrame(animationFrameId);
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";

  resultPlaceholder.innerHTML = `
    <strong>Running inference</strong>
    <p>Uploading audio, extracting mel-spectrogram features, and querying the CNN model.</p>
  `;

  try {
    const formData = new FormData();
    formData.append("audio", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Prediction request failed.");
    }

    renderPrediction(result);
  } catch (error) {
    resultPlaceholder.innerHTML = `
      <strong>Analysis failed</strong>
      <p>${error.message}</p>
    `;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze Audio";
  }
}

audioInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  handleFile(file);
});

analyzeButton.addEventListener("click", analyzeAudio);

drawEmptyStates();
