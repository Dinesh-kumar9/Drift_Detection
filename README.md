# 🧠 MLOps Observability Platform

> **An industry-grade MLOps platform** — Upload → Preprocess → Train → Deploy → Monitor with **stage-wise drift detection and root-cause attribution.**

[![Backend CI](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/Dinesh-kumar9/Drift_Detection/actions)
[![Frontend CI](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/Dinesh-kumar9/Drift_Detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🚀 What Makes This Different

Most MLOps tools only flag "drift detected" at the model output. This platform independently monitors **three pipeline stages** and uses a **temporal-magnitude attribution engine** to identify _which_ stage caused the drift.

```
Raw Data   ──► KS Test (scipy)  ──┐
Preprocessed ► KS Test (scipy)  ──►  Attribution Engine  ──►  Ranked Root-Cause
Predictions  ► ADWIN (river)   ──┘      + Slack Alert             Report
```

**Core differentiator:** The attribution score is:
```
attribution_score(stage) = drift_magnitude × (1 / onset_lag_seconds)
```
The stage that drifted **hardest AND earliest** gets ranked #1 — the most likely root cause.

---

## 📋 Feature Matrix

| Capability | Details | Phase |
|---|---|---|
| Dataset ingestion | CSV upload, schema validation, baseline stats | Phase 1 |
| Preprocessing pipeline | Missing value imputation, encoding, scaling — versioned | Phase 1 |
| Multi-model training | 2–3 models in parallel per dataset (RF, XGB, LogReg) | Phase 1 |
| Experiment tracking | Params, metrics, artifacts per run | Phase 1 |
| Model registry | staging → production → archived lifecycle | Phase 1 |
| Model serving | REST API with request/response logging | Phase 1 |
| **Stage 1 drift monitor** | KS test on raw ingested feature distributions | **Phase 2** |
| **Stage 2 drift monitor** | KS test on preprocessed feature distributions | **Phase 2** |
| **Stage 3 drift monitor** | ADWIN on prediction error stream | **Phase 2** |
| **Root-cause attribution** | Temporal-magnitude ranking across 3 stages | **Phase 2** |
| **Slack alerting** | Ranked attribution report attached | **Phase 2** |
| Retraining loop | Cost-benefit gated, human approval required | Phase 3 |
| Audit trail | Full log of every action, promotion, and decision | Phase 3 |
| Auth + RBAC | viewer / engineer / admin roles | Phase 4 |
| SLA tiering | Critical / standard / low monitoring cadence | Phase 4 |
| Multi-tenant dashboard | All models' health on one screen | Phase 4 |

---

## ⚡ Quick Start (Local Dev)

**Requirements:** Docker Desktop (with Compose V2)

```bash
# Clone
git clone https://github.com/Dinesh-kumar9/Drift_Detection.git
cd Drift_Detection

# Copy environment config (edit values if needed)
cp backend/.env.example backend/.env

# Start all services
docker-compose up --build
```

| Service | URL | Credentials |
|---|---|---|
| **Frontend** | http://localhost:3000 | — |
| **Backend API (Swagger)** | http://localhost:8000/docs | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **Flower** (task queue monitor) | http://localhost:5555 | — |

**Stop all services:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up --build --force-recreate
```

---

## 🏗️ Architecture

Full architecture document with all diagrams: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│              Frontend (React 18 + Vite + TS)            │
│  Dataset Upload │ Experiments │ Models │ Observability  │
└────────────────────────┬────────────────────────────────┘
                         │ REST / JSON
┌────────────────────────▼────────────────────────────────┐
│             Backend API Layer (FastAPI)                  │
│  Auth │ Ingestion │ Training │ Registry │ Serving │ Obs  │
└────┬──────────┬──────────────────────────────────────────┘
     │          │
     ▼          ▼
┌─────────┐  ┌──────────────────────────────┐
│Postgres │  │ Async Workers (Celery)        │
│Redis    │  │ training · drift · retrain    │
│MinIO/S3 │  │ + Beat Scheduler (5min drift) │
└─────────┘  └──────────────────────────────┘
```

### Observability Engine

```
S1 Ingestion ──► scipy KS Test   ──┐
S2 Preprocessing► scipy KS Test  ──►  Attribution Engine  ──► Ranked Report
S3 Predictions  ► river ADWIN    ──┘  score = magnitude ×      + Slack Alert
                                       (1/onset_lag)
```

→ Full diagrams in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | React 18 + Vite + TypeScript | Fast dev loop, type-safe API contracts |
| **Styling** | Tailwind CSS v3 | Rapid, consistent dark-mode UI |
| **Charts** | Recharts | React-native time-series drift charts |
| **Server state** | TanStack Query | Caching, deduplication, no Redux |
| **API** | FastAPI (Python 3.11) | Async, auto-generates OpenAPI docs |
| **ORM** | SQLAlchemy async + Alembic | Migrations, type-safe queries |
| **Task queue** | Celery + Redis | Non-blocking training & drift jobs |
| **Scheduler** | Celery Beat → Airflow (Phase 2+) | 5-min scheduled drift checks |
| **Drift (batch)** | `scipy.stats.ks_2samp` | Non-parametric, any distribution shape |
| **Drift (stream)** | `river.drift.ADWIN` | Adaptive windowing for error streams |
| **ML models** | scikit-learn + XGBoost | Classification + regression |
| **Object storage** | MinIO (local) / AWS S3 (prod) | S3-compatible, zero cost for dev |
| **Database** | PostgreSQL 16 (asyncpg) | Metadata, drift events, audit logs |
| **Auth** | JWT → AWS Cognito (Phase 4) | Simple local auth, enterprise-ready swap |
| **IaC** | Terraform (AWS) | VPC, RDS, S3, ECS Fargate, Cognito |
| **CI/CD** | GitHub Actions | Lint → test → staging → approval → prod |

---

## 🗂️ Project Structure

```
Drift_Detection/
│
├── frontend/                        # React 18 + Vite + TypeScript
│   ├── src/
│   │   ├── api/client.ts            # Axios + JWT interceptor
│   │   ├── components/Layout.tsx    # Sidebar + header shell
│   │   ├── pages/
│   │   │   ├── DatasetUpload.tsx         # Phase 1
│   │   │   ├── ExperimentDashboard.tsx   # Phase 1
│   │   │   ├── ModelComparison.tsx       # Phase 1
│   │   │   ├── ObservabilityDashboard.tsx # Phase 2 ⭐
│   │   │   └── RetrainApproval.tsx       # Phase 3
│   │   ├── App.tsx                  # Router + QueryClient
│   │   └── index.css                # Design system (Tailwind)
│   ├── tailwind.config.js
│   └── Dockerfile.dev
│
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI routers (7 modules)
│   │   ├── core/
│   │   │   ├── config.py            # Pydantic settings
│   │   │   ├── database.py          # Async SQLAlchemy engine
│   │   │   └── security.py          # JWT + RBAC
│   │   ├── models/__init__.py       # 9 ORM tables
│   │   ├── services/
│   │   │   └── drift/               # ⭐ Core differentiator
│   │   │       ├── stage_monitors.py    # Phase 2
│   │   │       ├── attribution_engine.py # Phase 2
│   │   │       ├── alert_service.py     # Phase 2
│   │   │       └── cost_benefit_gate.py # Phase 3
│   │   └── workers/celery_app.py    # 3-queue Celery config
│   ├── alembic/                     # DB migrations
│   ├── tests/
│   ├── Dockerfile                   # Multi-stage: api / worker / beat
│   └── requirements.txt
│
├── infra/terraform/                 # AWS IaC (VPC, S3, RDS, ECS, Cognito)
│
├── .github/workflows/
│   ├── backend-ci.yml               # Lint → test (60% cov) → ECR push
│   ├── frontend-ci.yml              # Lint → typecheck → build
│   └── deploy.yml                   # Staging → approval → production
│
├── docs/
│   ├── ARCHITECTURE.md              # Full diagrams (6 Mermaid + ER)
│   └── api-spec.yaml                # OpenAPI 3.1 (14 endpoints)
│
├── docker-compose.yml               # 8-service local dev stack
└── README.md
```

---

## 📡 API Reference

Interactive docs (local): **http://localhost:8000/docs**  
Full spec: [`docs/api-spec.yaml`](docs/api-spec.yaml)

```
POST   /auth/login                           Authenticate → JWT
POST   /datasets                             Upload + validate CSV
GET    /datasets/{id}                        Dataset metadata
POST   /experiments/train                    Kick off multi-model training
GET    /experiments/{id}/compare             Side-by-side metrics comparison
POST   /models/{id}/deploy                   Promote to production
POST   /models/{id}/predict                  Run inference (< 300ms p95)
GET    /observability/{model_id}/drift       Stage-wise drift status   ⭐
GET    /observability/{model_id}/attribution Root-cause attribution report ⭐
POST   /retrain/{model_id}/trigger           Request retraining
POST   /retrain/{job_id}/approve             Human approval gate
GET    /audit-logs                           Filterable audit trail
GET    /admin/models                         Multi-tenant health dashboard
```

---

## 🗄️ Database Schema (9 Tables)

| Table | Purpose |
|---|---|
| `users` | Auth, roles (viewer/engineer/admin) |
| `datasets` | CSV metadata, S3 path, baseline stats |
| `experiment_runs` | Training run params, metrics, artifact paths |
| `model_versions` | Registry states (staging/production/archived) |
| `predictions` | Every inference request logged |
| `drift_events` | Per-stage drift scores + onset timestamps ⭐ |
| `attribution_reports` | Ranked root-cause reports ⭐ |
| `retrain_jobs` | Pending/approved/rejected retrain decisions |
| `audit_logs` | Full immutable audit trail |

---

## 🚦 Build Phases

| Phase | Goal | Weeks | Status |
|---|---|---|---|
| **Phase 0** | Repo scaffold, Docker Compose, CI/CD, Architecture doc | 1 | ✅ **Complete** |
| **Phase 1** | Upload → Preprocess → Train → Compare → Deploy → Predict | 2–5 | 🔲 Planned |
| **Phase 2** | Stage-wise drift detection + root-cause attribution | 6–9 | 🔲 Planned |
| **Phase 3** | Cost-gated, human-approved retraining loop | 10–12 | 🔲 Planned |
| **Phase 4** | Auth/RBAC, SLA tiering, multi-tenant dashboard | 13–15 | 🔲 Planned |

> **Phase 2 is the non-negotiable core differentiator** — without it this is just another MLOps CRUD app.

---

## 🔀 Branch Strategy

```
main           ─── always deployable, protected (require CI + PR)
feature/fr1-*  ─── one branch per functional requirement
feature/fr8-stage-monitors   ← example: maps directly to FR8
```

Merge via squash to keep history readable as a portfolio repo.

---

## 🧪 Running Tests

```bash
# Backend unit tests
cd backend
pip install -r requirements.txt
pytest tests/unit/ --cov=app --cov-report=term-missing -v

# Frontend type check + tests
cd frontend
npm ci
npx tsc --noEmit
npm run test
```

CI enforces **60% backend coverage** (raises to 75% in Phase 4).

---

## ☁️ Production Deployment (AWS)

The `infra/terraform/` directory contains skeleton Terraform for:
- **VPC** — public/private subnets across 2 AZs
- **S3** — versioned, encrypted buckets for datasets + models
- **RDS** — Postgres 16 (multi-AZ in production, managed password rotation)
- **ECS Fargate** — API + worker containers, no server management
- **ECR** — image registry with lifecycle policy
- **Cognito** — user pool with custom role attribute (Phase 4)

```bash
cd infra/terraform
terraform init
terraform plan -var="environment=staging"
terraform apply -var="environment=staging"
```

---

## 🤝 Contributing

1. Branch: `git checkout -b feature/fr{N}-{description}`
2. Make changes → `git push origin feature/fr{N}-{description}`
3. Open PR against `main` — CI must pass before merge
4. Squash-merge on approval

---

## 📄 License

MIT © 2026 Dinesh Kumar
