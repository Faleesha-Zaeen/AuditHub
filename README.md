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

# 5. Run Streamlit (coming soon)
# streamlit run app/main.py
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

# Type check
mypy src/

# Run Streamlit (coming soon)
# streamlit run app/main.py

# Run FastAPI (coming soon)
# uvicorn src.api.main:app --reload
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please follow the standard fork-and-pull-request workflow. Ensure all tests pass and code is formatted with Black and isort before submitting.
