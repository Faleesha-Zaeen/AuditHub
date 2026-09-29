# AuditHub

**AuditHub is an end-to-end Dataset Quality, Repair, Robustness, and MLOps Platform.**

AuditHub empowers data scientists and ML engineers to profile, validate, audit, repair, and monitor datasets — all within a unified platform. It bridges the gap between data quality assurance and production-grade MLOps, providing tools for ingestion, validation, profiling, health scoring, intelligent repair, mutation testing, robustness evaluation, model training, experiment tracking, and report generation.

---

## Architecture Overview

AuditHub follows a **modular, layered architecture** with clear separation of concerns:

```
                         ┌──────────────────────────┐
                         │    Streamlit Dashboard    │  (Frontend)
                         │   (app/, app/pages/)      │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │      FastAPI Backend      │  (API Layer)
                         │    (Future Implementation)│
                         └────────────┬─────────────┘
                                      │
┌─────────────────────────────────────┼────────────────────────────────────────┐
│                                     │                                        │
│  ┌──────────┐ ┌──────────┐ ┌───────▼───────┐ ┌──────────┐ ┌──────────┐     │
│  │Ingestion │ │Validation│ │   Profiling   │ │  Quality  │ │  Repair  │     │  (src/)
│  └──────────┘ └──────────┘ └───────────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Health  │ │Mutation  │ │  Robustness   │ │ Training │ │Evaluation│     │
│  └──────────┘ └──────────┘ └───────────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────────────────────────────────────────────┐         │
│  │Reporting │ │               Pipeline Orchestrator               │         │
│  └──────────┘ └───────────────────────┬──────────────────────────┘         │
│                                        │                                   │
└────────────────────────────────────────┼───────────────────────────────────┘
                                         │
                  ┌───────────────────────▼───────────────────────┐
                  │               Database (SQLite)               │
                  │         src/database/ base layer              │
                  └───────────────────────────────────────────────┘
                                         │
                  ┌───────────────────────▼───────────────────────┐
                  │        Utils / Constants / Exceptions         │
                  │            src/utils/                         │
                  └───────────────────────────────────────────────┘
```

### Design Principles

- **Modularity** — Each capability is a self-contained module under `src/`
- **Separation of Concerns** — Frontend, backend, data processing, and configuration are isolated
- **Testability** — Every module is designed with testing in mind
- **Configurability** — Centralized configuration via YAML files in `configs/`
- **Extensibility** — New modules can be added without modifying existing code
- **Production Readiness** — Structured logging, custom exceptions, type hints, and docstrings

---

## Folder Structure

```
AuditHub/
├── app/                        # Streamlit frontend application
│   ├── __init__.py
│   ├── pages/                  # Streamlit multi-page app pages
│   │   └── __init__.py
│   ├── components/             # Reusable Streamlit UI components
│   │   └── __init__.py
│   └── assets/                 # Static assets (images, CSS, etc.)
│       └── __init__.py
│
├── configs/                    # Centralized configuration
│   ├── config.yaml             # Main application configuration
│   ├── params.yaml             # Tunable parameters
│   ├── schema.yaml             # Data schema definitions
│   └── logging.yaml            # Logging configuration
│
├── data/                       # Data storage (gitignored)
│   ├── raw/                    # Uploaded raw datasets
│   ├── validated/              # Validated datasets
│   ├── repaired/               # Repaired datasets
│   ├── mutated/                # Mutated datasets for testing
│   ├── processed/              # Feature-engineered datasets
│   └── reports/                # Generated reports
│
├── artifacts/                  # ML artifacts (gitignored)
├── models/                     # Trained models (gitignored)
├── reports/                    # Generated reports (gitignored)
├── mlruns/                     # MLflow experiment data (gitignored)
├── logs/                       # Application logs (gitignored)
├── docs/                       # Documentation
│
├── src/                        # Source code
│   ├── ingestion/              # Dataset upload & ingestion module
│   ├── validation/             # Dataset validation (Great Expectations + custom)
│   ├── profiling/              # Dataset profiling (ydata-profiling)
│   ├── quality/                # Dataset quality auditing
│   ├── repair/                 # Intelligent data repair
│   ├── health/                 # Dataset health score computation
│   ├── mutation_lab/           # Data mutation lab
│   ├── robustness/             # Robustness evaluation
│   ├── training/               # ML training (scikit-learn)
│   ├── evaluation/             # Model evaluation
│   ├── reporting/              # Report generation
│   ├── database/               # Database layer (SQLite)
│   ├── utils/                  # Shared utilities
│   │   ├── logger.py           # Centralized logging system
│   │   ├── constants.py        # Global constants & paths
│   │   ├── exceptions.py       # Custom exception hierarchy
│   │   └── helpers.py          # Reusable helper functions
│   └── pipeline/               # Pipeline orchestration
│
├── tests/                      # Test suite
│
├── .github/
│   └── workflows/              # GitHub Actions CI/CD
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build config, linters, test config
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose orchestration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/audithub.git
cd audithub

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Initialize DVC (optional)
dvc init
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t audithub .
docker run -p 8501:8501 audithub
```

---

## Usage

AuditHub can be driven three ways. All three share the same engines, so a
dataset repaired in the UI and one repaired from the CLI come out identical.

### 1. Dashboard

```bash
streamlit run app/main.py
```

Upload a dataset on the **Upload** page, then work through Summary →
Validation → Profiling → Quality Audit → Health Score → **Repair**. Repairs
apply to the active dataset, so every later page recomputes against the
repaired data and the health score updates to match.

### 2. Command line (headless)

Runs the whole flow and writes every artifact to disk:

```bash
python -m src.pipeline run data/raw/customers.csv --target churn
```

Useful flags: `--numeric-strategy {median,mean,mode}`, `--no-profile`,
`--no-train`, `--no-robustness`, `--report-format json`, `--max-rows N`,
`--json` for machine-readable output.

Exit codes: `0` all stages completed, `1` one or more non-critical stages
failed, `2` the run failed, `3` bad usage. This makes the pipeline safe to use
as a CI gate.

### 3. REST API

```bash
uvicorn src.api.main:app --reload
```

Nine endpoints under `/api/v1/` (`ingest`, `validate`, `profile`, `audit`,
`health`, `repair`, `mutate`, `robustness`, `train`). `POST /api/v1/repair`
returns the repaired rows as JSON, or the cleaned file itself with
`download=true`.

---

## Data Cleaning Guarantees

Files are normalised at load time so that "missing" means the same thing
throughout the platform:

- Disguised nulls (`?`, `-`, `N/A`, `null`, `missing`, blanks) become real
  nulls, so completeness metrics are not silently overstated.
- Numeric text is converted: `$50,000` → `50000`, `(1,200)` → `-1200`,
  `45%` → `0.45`. Values with leading zeros (zip codes) are left as text.
- An integer column that pandas floated only because it contains gaps is
  restored to a nullable integer, so imputing it never yields `32.5`.
- Column names are trimmed, duplicate labels are made unique, leftover index
  columns are dropped, and fully empty rows and columns are removed.

Every action is reported in the UI and recorded in the dataset metadata --
nothing is changed silently.

After **Auto-Repair**, the exported dataset is guaranteed to have no missing
values, no duplicate rows, no residual missing-value markers, correct column
types, and no index column. The target column is never imputed: rows with a
missing label are dropped, because inventing labels fabricates ground truth.
Re-loading an exported file requires no further repair.

---

## Tech Stack

| Category        | Technology                                      |
|-----------------|-------------------------------------------------|
| **Frontend**    | Streamlit, Plotly, Streamlit-AgGrid             |
| **Backend**     | FastAPI, Uvicorn                                |
| **Data**        | Pandas, NumPy, Scikit-learn                     |
| **Validation**  | Great Expectations                              |
| **Profiling**   | ydata-profiling                                 |
| **Monitoring**  | Evidently AI                                    |
| **MLOps**       | MLflow, DVC, Docker                             |
| **Database**    | SQLite (via SQLAlchemy)                         |
| **Testing**     | pytest, pytest-cov                              |
| **Code Quality**| Black, isort, ruff, mypy                        |

---

## Future Modules

| Module          | Description                                               |
|-----------------|-----------------------------------------------------------|
| **Ingestion**   | Upload, parse, and store datasets with format detection   |
| **Validation**  | Great Expectations integration for data contract checks   |
| **Profiling**   | Automated dataset profiling with ydata-profiling          |
| **Quality**     | Data quality auditing with configurable dimension checks  |
| **Repair**      | Intelligent imputation, outlier removal, and data repair  |
| **Health**      | Composite dataset health scoring with weighted dimensions |
| **Mutation**    | Data mutation lab for adversarial testing                 |
| **Robustness**  | Model and data robustness evaluation                      |
| **Training**    | Automated ML training with scikit-learn                   |
| **Evaluation**  | Model performance evaluation and comparison               |
| **Reporting**   | Automated report generation (HTML/PDF)                    |
| **Dashboard**   | Full Streamlit dashboard with interactive visualizations  |
| **API**         | FastAPI backend for programmatic access                   |
| **Pipeline**    | End-to-end pipeline orchestration                         |
| **CI/CD**       | GitHub Actions for testing and deployment                 |

---

## Workflow Diagram

```
                          ┌─────────────┐
                          │  User Upload │
                          │  (CSV/JSON/  │
                          │   Parquet)   │
                          └──────┬──────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │     Ingestion       │
                    │  - Format detection │
                    │  - Chunked loading  │
                    │  - Schema inference │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ Validation │ │ Profiling  │ │  Quality   │
       │   (GE)     │ │  (ydata)   │ │  Auditing  │
       └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │   Health Score  │
                    │  (0-100 grade)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌────────────┐ ┌────────────┐
     │   Repair     │ │  Mutation  │ │ Robustness │
     │ Intelligence│ │    Lab     │ │ Evaluation │
     └──────┬───────┘ └──────┬─────┘ └──────┬─────┘
            │                │              │
            └────────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │   ML Training   │
                    │  (scikit-learn) │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Evaluation    │
                    │  (Metrics)      │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    Reporting    │
                    │  (HTML/PDF)     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Dashboard     │
                    │  (Streamlit)    │
                    └─────────────────┘
```

---

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src tests/

# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Run the full pipeline over a dataset
python -m src.pipeline run data/raw/your.csv --target your_label

# Type check
mypy src/

# Run Streamlit
streamlit run app/main.py

# Run FastAPI
uvicorn src.api.main:app --reload

# Run both at once
python run_all.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please follow the standard fork-and-pull-request workflow. Ensure all tests pass and code is formatted with Black and isort before submitting.
