# Multi-Layer Image Authenticity Detection

A FastAPI service and Gradio UI that run an image through **eight layers** of forensic and learned detectors. A fusion layer combines them into one verdict, **authentic, uncertain, ai_generated, manipulated, or deepfake**, with a confidence score, a per-layer breakdown, and a heatmap of suspicious regions.

## Layers

| Layer | What it checks | Weight |
|---|---|---|
| 0 · Numerical forensics | ELA, DCT, SRM filters, PRNU sensor noise, Laplacian, chromatic aberration, metadata | 0.10 |
| 0.5 · Purification | Re-scores after JPEG, blur, bit-depth reduction, and crops (optional DiffPure); large shifts flag adversarial tampering | flags only |
| 1 · AI-generated | UnivFD, NPR, DRCT, FatFormer | 0.25 |
| 2 · Localization | TruFor, HiFi-IFDL, PIM demosaicing traces, copy-move detection | 0.20 |
| 2.5 · Text | OCR, font consistency, garbled-text and perspective checks | 0.10 (off when no text) |
| 2.6 · Semantic | Lighting, shadows, reflections, vanishing points | 0.10 |
| 3 · Deepfake | Reality Defender API with a local fallback | 0.15 (off when no faces) |
| 4 · Watermark | C2PA, invisible watermarks, SynthID-style g-function | 0.10 |
| 5 · Fusion | Weighted fusion, cross-layer conflict rules, verdict thresholds, merged heatmap | n/a |

Layers run in order. The detectors inside each layer run in parallel. All weights and thresholds live in [`config/default.yaml`](config/default.yaml).

## Run it

```bash
pip install -e ".[dev]"            # add ".[c2pa]" for C2PA checks
python scripts/download_weights.py

uvicorn detection.app:app --port 8000      # REST API
python -m detection.gradio_ui              # web UI on http://localhost:7860
```

Or with Docker (GPU):

```bash
docker compose up        # API on :8000, UI on :7860
```

Set `REALITY_DEFENDER_API_KEY` to use the Reality Defender deepfake layer. Without it, the local fallback runs.

## API

```bash
curl -X POST http://localhost:8000/analyze -F "file=@photo.jpg"
curl http://localhost:8000/health
```

`/analyze` returns `verdict`, `confidence`, `overall_score`, `flags`, a per-layer and per-detector `breakdown`, `processing_time_ms`, and `heatmap_base64`.

## Status

This is a work in progress. The weight URLs in `scripts/download_weights.py` are placeholders, so fill in real checkpoint URLs before the learned detectors (UnivFD, NPR, DRCT, FatFormer) can load.

## Tests

```bash
pytest
```
