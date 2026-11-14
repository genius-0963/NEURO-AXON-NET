"""
Unified Command-Line Interface for NeuroAxonNet (Phase 05).

Implements Section 7 CLI specification:
- validate-image INPUT --metadata metadata.json
- predict INPUT --model MODEL_BUNDLE --out OUTPUT_DIR
- evaluate MANIFEST --checkpoint CHECKPOINT --report REPORT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
import click
from PIL import Image
import numpy as np

from neuroaxonnet.inference.service import NeuroAxonInferenceService
from neuroaxonnet.data.manifest import DatasetManifest
from neuroaxonnet.evaluation.metrics import compute_full_agreement_suite


@click.group()
@click.version_option(version="0.1.0")
def main():
    """NeuroAxonNet: Retinal Ganglion Cell Axon Density Estimation CLI."""
    pass


@main.command("validate-image")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--metadata", "meta_path", type=click.Path(exists=True), default=None, help="JSON metadata file.")
def validate_image_cmd(input_path: str, meta_path: Optional[str]):
    """Validates raw image bytes, calculates SHA-256 pre-hash, and checks dimensions."""
    service = NeuroAxonInferenceService()
    p = Path(input_path)
    meta = {}
    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    with open(p, "rb") as f:
        content = f.read()

    try:
        arr, verified_meta = service.validate_image_bytes(content, meta)
        click.echo("=" * 60)
        click.echo("[VALID] Image Ingestion Audit Successful")
        click.echo(f"  File:       {p.name}")
        click.echo(f"  SHA-256:    {verified_meta['sha256']}")
        click.echo(f"  Dimensions: {verified_meta['width']} x {verified_meta['height']} px")
        click.echo(f"  Scale:      {verified_meta['pixel_size_um']} um/pixel")
        click.echo("=" * 60)
    except Exception as e:
        click.echo(f"[ERROR] Image validation failed: {e}", err=True)
        sys.exit(1)


@main.command("predict")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--model", "model_bundle", type=str, default="release/model_bundle", help="Model bundle path.")
@click.option("--out", "out_dir", type=click.Path(), default="artifacts/predictions", help="Output directory.")
def predict_cmd(input_path: str, model_bundle: str, out_dir: str):
    """Predicts continuous axon density map and total normal-appearing axon count."""
    service = NeuroAxonInferenceService()
    p = Path(input_path)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with open(p, "rb") as f:
        content = f.read()

    arr, meta = service.validate_image_bytes(content)

    if arr.shape[0] > 400 or arr.shape[1] > 400:
        result = service.predict_full_micrograph(arr, meta)
    else:
        result = service.predict_patch(arr, meta)

    density_map = result.pop("_density_map")

    # Save artifacts
    res_json_file = out_p / f"{p.stem}_result.json"
    with open(res_json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    overlay_png = service.generate_overlay_png(arr, density_map)
    overlay_file = out_p / f"{p.stem}_overlay.png"
    with open(overlay_file, "wb") as f:
        f.write(overlay_png)

    click.echo("=" * 60)
    click.echo(f"PREDICTION RESULT: {p.name}")
    click.echo(f"  Estimated Count: {result['count']['raw_estimate']} {result['count']['unit']}")
    click.echo(f"  QC Status:       [{result['qc']['status'].upper()}] Flags: {result['qc']['flags']}")
    click.echo(f"  Focus Metric:    {result['qc']['metrics']['laplacian_focus']}")
    click.echo(f"  Result JSON:     {res_json_file}")
    click.echo(f"  Overlay PNG:     {overlay_file}")
    click.echo("=" * 60)


@main.command("evaluate")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option("--checkpoint", "ckpt_path", type=str, default="artifacts/checkpoints", help="Checkpoint directory.")
@click.option("--report", "report_dir", type=click.Path(), default="artifacts/reports", help="Report output path.")
def evaluate_cmd(manifest_path: str, ckpt_path: str, report_dir: str):
    """Evaluates model performance across test manifest."""
    service = NeuroAxonInferenceService()
    manifest = DatasetManifest.load(manifest_path)
    test_patches = manifest.get_split("test")

    out_p = Path(report_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    preds, refs = [], []
    for rec in test_patches:
        p_img = Path(rec.file_path)
        if p_img.is_file():
            with open(p_img, "rb") as f:
                arr, meta = service.validate_image_bytes(f.read())
            res = service.predict_patch(arr, meta)
            preds.append(res["count"]["raw_estimate"])
            refs.append(float(rec.ground_truth_count))

    metrics = compute_full_agreement_suite(preds, refs)
    rep_file = out_p / "cli_evaluation_report.json"
    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    click.echo("=" * 60)
    click.echo(f"EVALUATION COMPLETE on {len(test_patches)} Test Patches:")
    click.echo(f"  MAE:             {metrics['mae']} axons/patch")
    click.echo(f"  RMSE:            {metrics['rmse']}")
    click.echo(f"  Concordance CCC: {metrics['concordance_correlation_ccc']}")
    click.echo(f"  Saved Report:    {rep_file}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
