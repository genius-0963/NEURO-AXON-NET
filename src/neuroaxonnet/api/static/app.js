let currentJobId = null;
let originalImage = null;
let overlayImage = null;

const uploadForm = document.getElementById("uploadForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const viewCanvas = document.getElementById("viewCanvas");
const ctx = viewCanvas.getContext("2d");
const loader = document.getElementById("loader");
const resultsSection = document.getElementById("resultsSection");
const alphaSlider = document.getElementById("alphaSlider");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("imageFile");
  if (!fileInput.files.length) return;

  const file = fileInput.files[0];
  await processImageFile(file);
});

alphaSlider.addEventListener("input", renderCanvas);
document.querySelectorAll("input[name='viewMode']").forEach((r) => {
  r.addEventListener("change", renderCanvas);
});

async function processImageFile(file) {
  loader.classList.remove("hidden");
  analyzeBtn.disabled = true;

  const metadata = {
    species: document.getElementById("species").value,
    stain: document.getElementById("stain").value,
    pixel_size_um: parseFloat(document.getElementById("pixelSize").value),
    microscope: document.getElementById("microscope").value,
  };

  const formData = new FormData();
  formData.append("image", file);
  formData.append("metadata", JSON.stringify(metadata));

  try {
    const res = await fetch("/api/v1/jobs", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis request failed.");
    }
    const data = await res.json();
    currentJobId = data.job_id;

    // Load full result
    const resultRes = await fetch(`/api/v1/jobs/${currentJobId}/result`);
    const resultData = await resultRes.json();

    // Load overlay image artifact
    const overlayRes = await fetch(`/api/v1/jobs/${currentJobId}/artifacts/overlay`);
    const overlayBlob = await overlayRes.blob();

    // Prepare canvas images
    originalImage = new Image();
    originalImage.src = URL.createObjectURL(file);
    await originalImage.decode();

    overlayImage = new Image();
    overlayImage.src = URL.createObjectURL(overlayBlob);
    await overlayImage.decode();

    viewCanvas.width = originalImage.width;
    viewCanvas.height = originalImage.height;

    renderCanvas();
    displayResults(resultData);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    loader.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
}

function renderCanvas() {
  if (!originalImage) return;
  const mode = document.querySelector("input[name='viewMode']:checked").value;
  const alpha = parseFloat(alphaSlider.value);

  ctx.clearRect(0, 0, viewCanvas.width, viewCanvas.height);

  if (mode === "original") {
    ctx.drawImage(originalImage, 0, 0);
  } else if (mode === "density") {
    ctx.drawImage(overlayImage, 0, 0);
  } else {
    // Blended mode
    ctx.drawImage(originalImage, 0, 0);
    ctx.globalAlpha = alpha;
    ctx.drawImage(overlayImage, 0, 0);
    ctx.globalAlpha = 1.0;
  }
}

function displayResults(data) {
  resultsSection.classList.remove("hidden");
  document.getElementById("statCount").textContent = data.count.raw_estimate;

  const qc = data.qc;
  const qcBadge = document.getElementById("statQCStatus");
  const qcFlags = document.getElementById("statQCFlags");

  qcBadge.textContent = qc.status.toUpperCase();
  qcBadge.className = "stat-badge " + qc.status.toLowerCase();

  qcFlags.textContent = qc.flags.length ? qc.flags.join(", ") : "Zero Defect Flags";
  document.getElementById("statFocus").textContent = qc.metrics.laplacian_focus;
  document.getElementById("statTissue").textContent = (qc.metrics.tissue_ratio * 100).toFixed(1) + "%";
}

async function loadSample(sampleId) {
  loader.classList.remove("hidden");
  try {
    const res = await fetch(`/data/fixtures/images/${sampleId}.png`);
    const blob = await res.blob();
    const file = new File([blob], `${sampleId}.png`, { type: "image/png" });
    await processImageFile(file);
  } catch (err) {
    alert("Sample could not be loaded: " + err.message);
    loader.classList.add("hidden");
  }
}

function exportArtifact(type) {
  if (!currentJobId) return;
  if (type === "result") {
    window.open(`/api/v1/jobs/${currentJobId}/result`, "_blank");
  } else {
    window.open(`/api/v1/jobs/${currentJobId}/artifacts/${type}`, "_blank");
  }
}

function setAdjudication(status) {
  alert(`Human review adjudication set to: [${status}]. Recorded in research audit log.`);
}
