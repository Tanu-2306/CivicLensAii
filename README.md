![Uploading LIVEMAP.png…]()
![Uploading OVERVIEW.png…]()
## Hackathon Submission
**Team Name:** SyntaxError
**Track:** Track 3
### Team Members
| Name | Registration Number |
|---|---|
| Naman Bothra | 24BCA7017 |
| Tanniskka Bhandari | 24BCE7620 |



# CivicLens AI
### Where Should We Fix First?

**An explainable AI-powered decision-support system for smarter civic incident prioritization.**

CivicLens AI helps municipal operations teams understand civic issues, identify high-priority incidents, explore location-level patterns, and estimate incident resolution time through a unified decision-support interface.

**Core idea:** Don't just collect complaints. Turn civic data into actionable priorities.

> **Project status:** Hackathon prototype using synthetic educational datasets. Predictions and simulated coordinates are not official municipal data or real-world safety assessments.

---

## 1. The Problem

Cities deal with recurring civic issues such as potholes, flooding, damaged roads, drainage failures, traffic incidents, and pedestrian safety concerns.

The challenge is not simply receiving reports. It is deciding **what to investigate first when time, staff, and budgets are limited.**

Traditional complaint workflows can make it difficult to:

- Identify high-priority incidents across multiple locations.
- Compare incidents using consistent, explainable criteria.
- Understand recurring civic problems and location-level patterns.
- Estimate how long an incident might take to resolve.
- Allocate limited resources using a common operational view.

Without effective prioritization, teams risk treating every report as equally urgent.

## 2. Our Solution

CivicLens AI brings incident data, transparent priority scoring, geographic context, analytics, and machine-learning predictions into one municipal operations interface.

The system is designed to help teams answer three questions:

1. **What needs attention first?** Rank incidents using configurable operational priority criteria.
2. **Where are problems occurring?** Explore location-linked incident information and civic trends.
3. **What resolution time should we plan for?** Generate an estimated resolution duration from historical training data.

CivicLens AI supports human decision-making rather than replacing municipal officers or emergency-response procedures.



## 3. What Makes CivicLens AI Different?

### Explainable prioritization
Priority scoring uses explicit criteria rather than presenting an unexplained AI score. Officials can review the information behind a recommendation.

### Predictive resolution estimates
A custom NumPy Linear Regression model and a scikit-learn Random Forest comparison estimate resolution time in hours. Both are evaluated against a simple median baseline.

### Location-aware analysis
Incident and complaint records are connected to location context using validated `location_id` relationships, enabling location-level analysis without incorrectly multiplying records during joins.

### Transparent model evaluation
The project reports MAE, MSE, RMSE, and R², including results that show where the current models fall short.

### Practical full-stack implementation
A React and TypeScript frontend communicates with a Python FastAPI backend, connecting the interface to data processing, model evaluation, and prediction workflows.

---

## 4. Key Features

| Feature | Purpose |
|---|---|
| Civic incident analytics | Explore incident categories, severity, and trends |
| Priority scoring | Support consistent, configurable incident triage |
| Location-linked insights | Relate incidents and complaints to locations |
| Resolution-time prediction | Estimate incident duration in hours |
| Model comparison | Compare Linear Regression, Random Forest, and a median baseline |
| Model Lab | Access model status, metrics, training history, and prediction workflows |
| FastAPI endpoints | Expose prediction and model information to the frontend |
| Data validation | Check required columns, unique identifiers, missing values, and join relationships |

*Feature availability depends on the relevant backend and frontend workflow being configured and running.*

---

## 5. How It Works

```text
Civic Incident & Complaint Data
              |
              v
     Data Validation
              |
              v
   Cleaning & Preparation
              |
              v
  Location Context & Features
              |
       +------+------+
       |             |
       v             v
Priority Scoring   ML Training
       |             |
       |       +-----+------+
       |       |            |
       |       v            v
       |    NumPy LR    Random Forest
       |       |            |
       |       +-----+------+
       |             |
       |       Model Evaluation
       |       MAE / RMSE / R²
       |             |
       +------+------+
              |
              v
       FastAPI Backend
              |
              v
   React Operations Interface
              |
              v
 Human Review & Decision Support
```

The resolution-time model is evaluated separately from the priority-scoring workflow. A predicted duration is an estimate, not a service-level guarantee or a measure of incident danger.

---

## 6. Dataset and Data Integrity

CivicLens AI currently uses three synthetic CSV datasets.

| Dataset | Records | Primary purpose |
|---|---:|---|
| `locations.csv` | 1,000 | Location context and geographic attributes |
| `incidents.csv` | 10,000 | Incident information and resolution history |
| `public_complaints.csv` | 8,000 | Public complaint information |

### Data preparation

The project includes checks for:

- Required columns and identifier uniqueness.
- Missing values and duplicate records.
- Valid `location_id` relationships.
- Many-to-one location joins.
- Positive resolution durations and valid timestamps.
- Unresolved incidents excluded from supervised duration regression.
- Target-duration outliers retained and reported for transparency.

For location-level aggregation, incident and complaint records are aggregated before being merged with location data. This avoids unintentionally multiplying rows when combining one-to-many relationships.

### Dataset limitations

All records and coordinates are simulated. The dataset represents a development and demonstration environment, not a validated sample of an actual city's operations.

The current incident records span January–December 2025. Results from one synthetic year cannot establish real-world seasonality or operational effectiveness.

---

## 7. Machine Learning Approach

### Model A — NumPy Linear Regression

A Linear Regression model implemented from scratch using NumPy batch gradient descent.

The training pipeline uses train-fitted categorical encoding and scaling, chronological data partitions, validation-based early stopping, and a held-out test set.

This demonstrates the implementation of a regression training workflow without relying on a prebuilt Linear Regression estimator.

### Model B — Random Forest Regression

A scikit-learn Random Forest pipeline provides a nonlinear comparison model.

The dedicated training workflow uses a small hyperparameter search and three-fold chronological `TimeSeriesSplit` cross-validation on the training partition. Preprocessing is kept inside the fitted pipeline.

### Baseline — Training Median

A simple baseline predicts the median resolution duration calculated from training records.

This baseline is important: a machine-learning model should demonstrate value beyond a straightforward alternative, not merely produce predictions.

### Evaluation strategy

The regression experiment uses a chronological 80/20 development-test split, with a validation partition inside the development window.

| Partition | Records |
|---|---:|
| Training | 4,560 |
| Validation | 1,140 |
| Held-out test | 1,426 |
| Total eligible records | 7,126 |

A further 2,874 unresolved incidents were excluded from supervised regression because a completed resolution duration was unavailable. The reported partition counts describe the eligible resolved records used in this experiment.

### Evaluation metrics

- **MAE:** Average absolute prediction error, in hours.
- **MSE:** Average squared prediction error.
- **RMSE:** Square root of MSE, in hours.
- **R²:** Measures performance relative to predicting the test-set mean; negative values indicate worse performance than that reference.

### Current measured results

The following values are from the current documented training run.

| Candidate | MSE ↓ | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| NumPy Linear Regression | 452.9027 | 16.4779 h | 21.2815 h | -0.0061 |
| Training median baseline | 472.1941 | **15.9583 h** | 21.7300 h | -0.0489 |
| Random Forest comparison | 477.0889 | 16.9650 h | 21.8424 h | -0.0598 |

**What the results tell us**

- NumPy Linear Regression achieves the lowest MSE and RMSE among these candidates.
- The training-median baseline achieves the lowest MAE.
- All three candidates have negative R² on the held-out test set.
- The current experiment does not demonstrate reliable predictive improvement over a simple baseline on every metric.

The API selects the median predictor under the configured MAE-based selection criterion, while retaining the scratch model estimate for comparison. The dedicated Random Forest endpoint is available for model-specific predictions and evaluation.

These results are reported to establish a reproducible baseline for future improvements, not to claim proven real-world impact.

---

## 8. Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI and styling | Tailwind CSS, Lucide |
| Visualizations | Recharts |
| Maps | Leaflet |
| Backend | Python, FastAPI |
| Data processing | Pandas, NumPy |
| Machine learning | NumPy gradient descent, scikit-learn Random Forest |
| Model persistence | JSON artifacts, Joblib |
| Testing | Pytest |
| Version control | Git and GitHub |

---

## 9. System Architecture

```text
CivicLensAii/
|
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   `-- ...
|   |-- ml/
|   |   |-- train_linear_regression.py
|   |   `-- train_random_forest.py
|   |-- models/
|   |   |-- random_forest_pipeline.joblib
|   |   `-- model_metadata.json
|   |-- tests/
|   `-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- ...
|
|-- models/
|   |-- incident_resolution_linear.json
|   `-- incident_resolution_forest.joblib
|
|-- reports/
|   |-- data_cleaning_report.json
|   |-- incident_resolution_linear.json
|   |-- random_forest_metrics.json
|   `-- linear_regression_training.log
|
|-- incidents.csv
|-- locations.csv
|-- public_complaints.csv
|-- README.md
`-- start.ps1
```

*This is a representative structure. Consult the repository for the exact current file layout.*

---

## 10. Getting Started

### Prerequisites

- Git
- Python with a version compatible with `backend/requirements.txt`
- Node.js and npm

### Step 1 — Clone the repository

```bash
git clone https://github.com/Tanu-2306/CivicLensAii.git
cd CivicLensAii
```

### Step 2 — Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the Python executable directly using `.venv\Scripts\python.exe`.

### Step 3 — Install backend dependencies

```powershell
python -m pip install -r backend\requirements.txt
python -m pip check
```

### Step 4 — Install frontend dependencies

```powershell
Push-Location frontend
npm ci
Pop-Location
```

### Step 5 — Train and verify the regression workflow

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m backend.ml.train_linear_regression
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

Build the frontend:

```powershell
Push-Location frontend
npm run build
Pop-Location
```

### Step 6 — Start the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

API documentation:

`http://127.0.0.1:8000/docs`

### Step 7 — Start the frontend

Open another terminal from the repository root:

```powershell
Push-Location frontend
npm run dev -- --host 127.0.0.1
Pop-Location
```

Open the application at:

`http://127.0.0.1:5173`

If required, configure `VITE_API_BASE_URL` in the appropriate Vite environment file to point to the backend.

### Train the dedicated Random Forest model

Run separately from the repository root:

```powershell
.\.venv\Scripts\python.exe -m backend.train_model
```

This workflow trains and evaluates the dedicated Random Forest pipeline. Training and API startup are separate operations.

---

## 11. API Endpoints

### Primary application API

| Endpoint | Purpose |
|---|---|
| `GET /api/models/status` | Inspect model status |
| `GET /api/models/metrics` | Retrieve model metrics |
| `GET /api/models/training-history` | Inspect training history |
| `POST /api/models/train` | Start the supported training workflow |
| `POST /api/models/predict` | Request a regression prediction |

### Dedicated Random Forest API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API health |
| `GET /model/metrics` | Retrieve Random Forest evaluation metadata |
| `POST /predict` | Generate a Random Forest resolution-time estimate |

The dedicated Random Forest prediction endpoint accepts incident information such as location ID, incident type, severity, reporting source, and reported injuries, according to the implemented request schema.

Example request body:

```json
{
  "location_id": "L0001",
  "incident_type": "Collision",
  "severity": "Critical",
  "reported_by": "Citizen",
  "reported_injuries": 2
}
```

This is an example request shape, not a claim that the incident exists in a real city. Refer to the running API documentation for the authoritative schema and response format.

---

## 12. Responsible AI and Limitations

CivicLens AI is designed as a human-in-the-loop decision-support prototype.

- Synthetic coordinates and records must not be interpreted as actual civic conditions.
- Priority scores are operational heuristics, not calibrated probabilities of harm.
- Resolution-time estimates are not service guarantees.
- Current regression results are weak and require further validation.
- The historical experiment does not establish performance on unseen locations.
- The model does not currently account for all operational factors, such as workforce availability, weather, work complexity, or material shortages.
- High-risk incidents require appropriate human assessment and established emergency procedures.

Before deployment with real municipal data, the system would need representative historical records, privacy and security controls, bias and subgroup analysis, stronger temporal and geographic validation, monitoring, and review with municipal stakeholders.

---

## 13. Roadmap

### Near-term priorities

- Improve resolution-time prediction through feature analysis and baseline-driven experimentation.
- Validate model behavior across incident categories and severity levels.
- Add clearer explanations of the factors associated with priority and predicted duration.
- Strengthen automated tests for data loading, joins, inference, and API contracts.
- Improve the demonstration workflow so judges can trace an incident from input to recommendation.

### Future directions

- Integrate verified municipal datasets where access and permissions allow.
- Explore duplicate-report detection and recurring hotspot analysis.
- Evaluate geographic generalization to locations excluded from training.
- Add resource-capacity and operational-constraint analysis.
- Assess prediction drift and monitor model quality after deployment.

These are proposed improvements, not claims that the capabilities have already been implemented.

---

## 14. Hackathon Demonstration

A clear demonstration should follow one incident through the system:

1. **Introduce the problem:** Explain why municipal teams need to prioritize limited resources.
2. **Show the incident data:** Demonstrate the available fields and location context.
3. **Explain the priority:** Show the incident's operational scoring and relevant factors.
4. **Generate a prediction:** Submit an incident through the Model Lab and inspect the returned estimate.
5. **Explain the model results:** Compare Linear Regression and Random Forest with the median baseline.
6. **Close with the value:** Show how a unified interface can support more structured review and decision-making.

For the model-evaluation portion, be explicit that the current test results are weak. Explain how the baseline reveals the need for better features and stronger validation. This demonstrates engineering judgment rather than overstating model performance.

---

## 15. Why CivicLens AI Matters

CivicLens AI explores a practical question for local government: **how can limited municipal resources be directed toward the problems that need attention most?**

By bringing together explainable priority scoring, location-linked civic analytics, and transparent predictive modeling, the project provides a foundation for more structured municipal decision-making.

Its current contribution is a working technical prototype and a measurable starting point for further validation—not yet proof of faster real-world resolution.

**CivicLens AI — Understand the problem. Explain the priority. Support the decision.**

---

## Repository

**GitHub:** https://github.com/Tanu-2306/CivicLensAii

