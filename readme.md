# PolyGuard

Evidence-based drug interaction analysis with Explainable AI.

---

## Project Structure

This is the exact layout matching your project (based on your file tree):

```
DDI-PREDICTION/                        ← project root (run uvicorn from here)
│
│   # ── Your original flat files (do NOT move these) ──
├── nlp_engine.py
├── polyguard_engine_evidence_based.py
├── evidence_based_weights.py
├── main.py
├── app.py
├── generate_bibliography.py
├── literature_sources.py
├── requirements.txt
│
├── datasets/                          ← CSV files
│
│   # ── New package ──
├── polyguard/
│   ├── __init__.py
│   ├── asgi.py                        ← uvicorn entry point
│   │
│   ├── core/
│   │   ├── __init__.py                ← injects project root into sys.path
│   │   ├── analyser.py                ← pipeline orchestrator (Steps 1-7)
│   │   ├── data_loader.py             ← all CSV I/O
│   │   ├── models.py                  ← Pydantic request/response types
│   │   ├── bibliography.py            ← BibliographyGenerator
│   │   └── xai_explainer.py           ← XAI module
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                     ← FastAPI routes
│   │
│   └── scripts/
│       ├── __init__.py
│       └── run_analysis.py            ← CLI
│
└── examples/
    └── usage.py
```

> **Why the flat files stay at the root:** `polyguard/core/__init__.py`
> adds the project root to `sys.path` at import time, so `nlp_engine`,
> `polyguard_engine_evidence_based`, and `evidence_based_weights` are
> importable from anywhere without moving them.

---

## Quickstart

### Install

```bash
pip install -e ".[dev]"
```

### Run the API server

```bash
uvicorn polyguard.asgi:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### Run the CLI

```bash
# Search brands
polyguard --search Aug

# List ingredients
polyguard --ingredients "Augmentin 625 Duo Tablet"

# Full analysis — no patient data
polyguard --brands "Augmentin 625 Duo Tablet" "Ascoril LS Syrup"

# Full analysis — with patient context
polyguard \
  --brands "Augmentin 625 Duo Tablet" "Azithral 500 Tablet" "Ascoril LS Syrup" \
  --age 72 --gender Female \
  --conditions Hypertension "Diabetes Type 2" "Atrial Fibrillation" COPD \
  --lab eGFR=42 ALT=85 platelet_count=110 INR=3.2 blood_glucose=195 \
  --save report.json
```

### Use as a library

```python
from polyguard.core import PolyGuardAnalyser
from polyguard.core.data_loader import DataLoader

loader   = DataLoader("./datasets").load()
analyser = PolyGuardAnalyser(loader)

result = analyser.analyse(
    brand_names  = ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
    patient_data = {"age": 72, "conditions": ["Hypertension"],
                    "lab_values": {"eGFR": 42}},
    explain      = True,
)

print(result.summary.overall_risk_level)   # e.g. 'MODERATE'
print(result.cascades)                      # list[CascadeAlert]
```

See [`examples/usage.py`](examples/usage.py) for the complete guide.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Liveness probe |
| GET  | `/brands/search?prefix=Aug` | Brand name prefix search |
| GET  | `/brands/{brand}/ingredients` | Ingredient lookup |
| POST | `/analyse` | Full pipeline (Steps 1-7) |
| GET  | `/analyse/quick?ingredients=amoxicillin&ingredients=clarithromycin` | Direct ingredient analysis |
| GET  | `/bibliography` | Evidence base as JSON |
| GET  | `/bibliography/markdown` | Evidence base as Markdown |

### POST /analyse — request body

```json
{
  "brand_names": ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
  "patient_data": {
    "age": 72,
    "gender": "Female",
    "conditions": ["Hypertension", "Diabetes Type 2"],
    "lab_values": { "eGFR": 42, "ALT": 85, "INR": 3.2 }
  },
  "explain": true
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYGUARD_DATA_DIR` | `./datasets` | Path to datasets directory |
| `POLYGUARD_LOG_LEVEL` | `INFO` | Python logging level |

---

## Design Decisions

| Decision | Why |
|----------|-----|
| `DataLoader` is separate from `PolyGuardAnalyser` | Lets you mock/swap data sources in tests without touching engine logic |
| All models are Pydantic | Single source of truth for validation, serialisation, and OpenAPI schema |
| Engine imported lazily in `PolyGuardAnalyser` | NLP model training runs only when `.analyse()` is first called, keeping API startup fast |
| `create_app()` factory | Enables multiple app instances in tests without shared global state |
| `asgi.py` entry point | Standard pattern for uvicorn / gunicorn deployment |