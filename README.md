# CivicLens AI

**Where Should We Fix First?** A local municipal-operations decision-support prototype using synthetic educational records.

> The supplied coordinates and reports are simulated. This is not an official city dataset or a safety assessment. Scores and predictions are screening aids for human review only.

## Data

Keep the supplied files in the project root: `locations.csv`, `incidents.csv`, `public_complaints.csv`, and `README.txt`. The application validates the required columns and the `location_id` relationship at load time. Current source counts are 1,000 locations, 10,000 incidents, and 8,000 complaints. All three join to the location table; record IDs are unique and the current files contain no missing values. Incident records span 2025-01-01 through 2025-12-31.

The training-cleaning report is generated at `reports/data_cleaning_report.json`. Unresolved incidents and invalid/non-positive durations or timestamps are excluded from supervised regression; valid duration outliers are counted and retained. Joins validate many-to-one location context, while location-level outputs aggregate one-to-many incident and complaint rows before merging.

## Architecture

- `frontend/`: React, TypeScript, Vite, Tailwind CSS, Recharts, Leaflet, and Lucide.
- `backend/app/`: FastAPI API, CSV loading/validation, transparent priority scoring, and other model workflows.
- `backend/ml/`: NumPy linear regression with train-fitted categorical encoding/scaling, chronological training, metrics, and JSON persistence.
- `backend/ml/train_random_forest.py`: scikit-learn Random Forest pipeline, temporal CV/tuning, test evaluation, metadata, and inference artifact.
- `models/`: persisted regression/forest artifacts and evaluation metadata.
- `backend/models/`: dedicated Random Forest inference pipeline and feature/model metadata.
- `reports/`: data-quality report, epoch log, and measured regression report.

The primary model uses NumPy batch gradient descent, not a prebuilt Linear Regression estimator. It uses a chronological 80/20 test holdout; the last 20% of the earlier development window is validation. The median baseline and Random Forest comparison use the same untouched test records. The predictor selected for the API is the candidate with lower held-out MAE; the scratch model remains separately available for evaluation.

## Windows Setup

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m pip check
Push-Location frontend
npm ci
Pop-Location
```

If npm is temporarily offline and the pinned packages are cached, use `npm ci --offline`.

## Train and Verify

```powershell
.\.venv\Scripts\python.exe -m backend.ml.train_linear_regression
.\.venv\Scripts\python.exe -m pytest backend\tests -q
Push-Location frontend
npm run build
Pop-Location
```

Training writes:

- `models/incident_resolution_linear.json` — versioned weights and train-fitted preprocessing.
- `models/incident_resolution_forest.joblib` — comparison model from the same chronological training partition.
- `reports/incident_resolution_linear.json` — split, epoch histories, measured metrics, baseline, forest comparison, chosen predictor, and limitations.
- `reports/linear_regression_training.log` — one measured training/validation MSE line per completed epoch.
- `reports/data_cleaning_report.json` — dataset schema, missing/duplicate checks, joins, exclusions, and flagged target outliers.

The current training run used 7,126 resolved records with valid positive duration and timestamp; 2,874 unresolved rows were excluded. Its chronological partitions are 4,560 training, 1,140 validation, and 1,426 test rows. Early stopping retained epoch 17 after 117 epochs because validation loss did not improve for 100 epochs.

Current held-out results (hours for duration errors):

| Candidate | MSE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| NumPy Linear Regression | 452.9027 | 16.4779 | 21.2815 | -0.0061 |
| Training median baseline | 472.1941 | 15.9583 | 21.7300 | -0.0489 |
| Random Forest comparison | 477.0889 | 16.9650 | 21.8424 | -0.0598 |

The scratch model has lower MSE/RMSE than the median but worse MAE and slightly negative R². The API therefore selects the median predictor by the explicitly configured MAE criterion; it also returns the scratch model estimate separately. These weak historical results are not evidence of real-world performance.

### Random Forest Regression

Train the dedicated Random Forest API model separately from starting FastAPI:

```powershell
\.venv\Scripts\python.exe -m backend.train_model
```

This performs a small grid search with 3-fold chronological `TimeSeriesSplit` on the 4,560 training rows only. The untouched test window is the same 1,426 rows used by the saved scratch Linear Regression and median comparisons. Preprocessing lives inside the fitted scikit-learn pipeline. The endpoint uses the fitted feature pipeline and clamps negative predictions to zero hours.

Current RF evaluation:

| Candidate | Held-out MAE | Held-out RMSE | Held-out R² |
|---|---:|---:|---:|
| Tuned Random Forest | 16.6500 h | 21.5636 h | -0.0329 |
| Scratch NumPy Linear Regression | 16.4779 h | 21.2815 h | -0.0061 |
| Training median baseline | 15.9583 h | 21.7300 h | -0.0489 |

Temporal CV selected `max_depth=14`, `min_samples_leaf=8`, and 120 trees, with mean CV MAE 17.8393 hours. The RF did not beat the median on held-out MAE, and its R² is negative; it is not claimed to be superior. Its artifact is retained for the requested model-specific prediction route and comparison.

RF artifacts and report:

- `backend/models/random_forest_pipeline.joblib`
- `backend/models/model_metadata.json`
- `reports/random_forest_metrics.json`

The RF API provides `GET /health`, `GET /model/metrics`, and `POST /predict`. Example JSON shape:

```json
{
	"location_id": "L0001",
	"incident_type": "Collision",
	"severity": "Critical",
	"reported_by": "Citizen",
	"reported_injuries": 2
}
```

The existing Model Lab form posts to `/predict`; `VITE_API_BASE_URL` is read from `frontend/.env.development` or another Vite environment file. A collision with 5+ reported injuries shows a separate human-review warning, independent of the resolution estimate.

## Run the Application

Start the API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
Push-Location frontend
npm run dev -- --host 127.0.0.1
Pop-Location
```

Open <http://127.0.0.1:5173>. API documentation is at <http://127.0.0.1:8000/docs>. Alternatively run `./start.ps1` from the project root to open both services in separate PowerShell windows. OpenStreetMap tiles require an internet connection; the API and application data remain local.

Key model routes: `GET /api/models/status`, `GET /api/models/metrics`, `GET /api/models/training-history`, `POST /api/models/train`, and `POST /api/models/predict`. Model training runs in a background thread with actual per-epoch history; the status and history endpoints expose its current state and terminal errors.

Random Forest routes: `GET /health`, `GET /model/metrics`, and `POST /predict`. Set `CIVICLENS_FRONTEND_ORIGIN` if the frontend is served from a non-default development origin. The browser UI reads its API base from `VITE_API_BASE_URL`.

## Limitations

- One synthetic year does not establish stable seasonality; incident-volume output is a short rolling baseline.
- Metadata cannot describe work complexity, resource availability, weather, or causal context.
- Locations can appear in both time partitions; the split prevents future-to-past leakage but does not estimate performance on unseen locations.
- The score is a configurable operational screening heuristic, not a probability of harm.
- The regression test metrics are weak; the selected median estimate is not a service guarantee.
- Never use the synthetic coordinates or predictions as actual civic or public-safety assessments.