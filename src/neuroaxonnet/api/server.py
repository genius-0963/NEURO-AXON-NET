"""
FastAPI REST Service and Research Web Server (Phase 05).

Implements Section 7 API Specification:
- POST /api/v1/jobs
- GET  /api/v1/jobs/{job_id}
- GET  /api/v1/jobs/{job_id}/result
- GET  /api/v1/jobs/{job_id}/artifacts/{artifact_name}
- GET  / (Interactive Research Review UI)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from neuroaxonnet.inference.service import NeuroAxonInferenceService

app = FastAPI(
    title="NeuroAxonNet API",
    description="Research-grade Retinal Ganglion Cell Axon Density and Count Service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single Python inference engine
inference_service = NeuroAxonInferenceService()

# In-memory job cache
JOB_STORE: Dict[str, Dict[str, Any]] = {}
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "NeuroAxonNet",
        "version": "0.1.0",
        "disclaimer": "RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE.",
    }


@app.post("/api/v1/jobs")
async def create_job(
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    model_version: Optional[str] = Form("0.1.0"),
):
    """
    Submits a microscopy image for automated axon count density analysis.
    """
    try:
        content = await image.read()
        meta_dict = json.loads(metadata) if metadata else {}
        arr, meta = inference_service.validate_image_bytes(content, meta_dict)

        # Decide whether to run patch or whole-image
        if arr.shape[0] > 400 or arr.shape[1] > 400:
            result = inference_service.predict_full_micrograph(arr, meta)
        else:
            result = inference_service.predict_patch(arr, meta)

        job_id = result["job_id"]
        # Cache raw original and result in store
        JOB_STORE[job_id] = {
            "result": result,
            "original_image": arr,
            "density_map": result.pop("_density_map"),
        }

        return {
            "job_id": job_id,
            "status": result["status"],
            "qc_flags": result["qc"]["flags"],
            "result_url": f"/api/v1/jobs/{job_id}/result",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    job_data = JOB_STORE[job_id]["result"]
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "progress": 1.0,
        "qc_flags": job_data["qc"]["flags"],
    }


@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JOB_STORE[job_id]["result"]


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_name}")
def get_job_artifact(job_id: str, artifact_name: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    job = JOB_STORE[job_id]
    orig = job["original_image"]
    density = job["density_map"]
    res = job["result"]

    if artifact_name in ("overlay", "overlay.png"):
        png_bytes = inference_service.generate_overlay_png(orig, density, alpha=0.5)
        return Response(content=png_bytes, media_type="image/png")

    elif artifact_name in ("csv", "summary.csv"):
        import csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["job_id", "raw_estimate", "unit", "qc_status", "flags", "sha256"])
        writer.writerow([
            job_id,
            res["count"]["raw_estimate"],
            res["count"]["unit"],
            res["qc"]["status"],
            ";".join(res["qc"]["flags"]),
            res["input"]["sha256"],
        ])
        return Response(content=buf.getvalue(), media_type="text/csv")

    elif artifact_name in ("density", "density.npy"):
        buf = io.BytesIO()
        np.save(buf, density)
        return Response(content=buf.getvalue(), media_type="application/octet-stream")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown artifact: {artifact_name}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = TEMPLATES_DIR / "index.html"
    if html_file.is_file():
        return html_file.read_text(encoding="utf-8")
    return "<h1>NeuroAxonNet Platform API is Running.</h1>"
