<div align="center">

# 🧠 MLOps Observability Platform

**An industry-grade MLOps platform with stage-wise drift detection and root-cause attribution.**

[![Backend CI](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/Dinesh-kumar9/Drift_Detection/actions/workflows/frontend-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)

</div>

---

## 🗺️ System Architecture

![MLOps Platform Architecture](docs/architecture.jpg)

> **Core Differentiator:** Most MLOps tools only flag "drift detected" at the model output. This platform independently monitors **3 pipeline stages** and uses a temporal-magnitude engine to pinpoint *which* stage caused it. In evaluation it raised **0% false alarms** and **named the correct stage every time it detected drift** — see [docs/DRIFT_ENGINE.md](docs/DRIFT_ENGINE.md).

```
score(stage) = magnitude × 1/(1 + onset_lag/bucket) × causal_factor
The stage that drifted hardest AND earliest — and isn't just inheriting an
upstream shift — is ranked #1 root cause
```

---

## 🔄 How It Works — End to End

```mermaid
flowchart TD
    A([👤 User]) -->|Upload CSV| B[📥 Ingestion Service\nSchema validation + baseline stats]
    B -->|Store| C[(☁️ MinIO / S3)]
    B -->|Metadata| D[(🐘 Postgres)]

    A -->|Trigger Training| E[🏋️ Training Service\nCelery async job]
    E -->|2–3 models in parallel| F[RandomForest · XGBoost · LogReg]
    F -->|Artifacts| C
    F -->|Metrics| D

    A -->|Compare + Deploy| G[📦 Model Registry\nstaging → production]

    G -->|Serve predictions| H[🚀 Inference API\n p95 < 300ms]
    H -->|Log every request| D

    I[⏰ Celery Beat\nevery 5 min] --> J

    subgraph OBS [⭐ Observability Engine]
        J[S1 Ingestion\nKS · χ² · null rate] --> ATT
        K[S2 Preprocessing\nKS · pipeline consistency] --> ATT
        L[S3 Predictions\nclass mix · confidence · error rate] --> ATT
        ATT[🧠 Attribution Engine\nmagnitude × onset × causal factor]
        ATT -->|ranked report| M[🔔 Slack Alert]
        ATT -->|drift confirmed| N[💰 Cost-Benefit Gate]
        N -->|approved| O[✅ Human Approval Console]
        O -->|approved| P[🚀 New Model Version]
    end

    H -.->|feed| J & K & L

    style OBS fill:#0f2a1a,stroke:#06b6d4,stroke-width:2px,color:#fff
    style ATT fill:#134e4a,stroke:#06b6d4,color:#fff
    style M fill:#1e3a5f,stroke:#6366f1,color:#fff
```

---

## 📡 Observability Engine — Deep Dive

```mermaid
flowchart LR
    subgraph INPUT["Live Production Data"]
        D1[Raw Ingestion\nData Stream]
        D2[Preprocessed\nFeatures]
        D3[Model\nPredictions]
    end

    subgraph MONITORS["Independent Stage Monitors"]
        M1["S1 Monitor\nscipy.stats.ks_2samp\n─────────────\ndrift_score · onset_ts"]
        M2["S2 Monitor\nscipy.stats.ks_2samp\n─────────────\ndrift_score · onset_ts"]
        M3["S3 Monitor\nKS · χ² · error-rate test\n─────────────\ndrift_score · onset_ts"]
    end

    subgraph ATTRIBUTION["Attribution Engine"]
        SCORE["score = magnitude × 1/(1+lag/bucket) × causal factor\n\nStage that drifted hardest & earliest\n(and isn't inherited) = Root Cause #1"]
    end

    subgraph ACTION["Response Layer"]
        REPORT[📊 Ranked Report\nstage · score · confidence]
        SLACK[💬 Slack Alert\nattachment: full report]
        GATE[💰 Cost-Benefit Gate\nROI check before retrain]
        APPROVE[👤 Approval Console\nhuman gate]
        DEPLOY[🚀 Deploy New Version\nfull audit trail]
    end

    D1 -->|batch KS| M1
    D2 -->|batch KS| M2
    D3 -->|batch + labels| M3
    M1 & M2 & M3 --> SCORE
    SCORE --> REPORT --> SLACK
    SCORE --> GATE --> APPROVE --> DEPLOY
```

---

## ⚡ Quick Start

**Requirement:** Docker Desktop

```bash
git clone https://github.com/Dinesh-kumar9/Drift_Detection.git
cd Drift_Detection

cp backend/.env.example backend/.env   # configure secrets

docker compose up --build
```

On start the API container applies all Alembic migrations and (in `ENVIRONMENT=local`) seeds demo users:

| Service | URL | Credentials |
|---|---|---|
| **Frontend** | http://localhost:3000 | `engineer@example.com` / `engineer1234` (also `admin@…`/`admin1234`, `viewer@…`/`viewer1234`) |
| **Backend Swagger** | http://localhost:8000/docs | same accounts via `POST /auth/login` |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **Flower** (task monitor) | http://localhost:5555 | — |

**Try the observability engine:** open *Observability*, pick a fault (e.g. "S2 · Preprocessing bug") and click
*Inject & check* — the attribution panel names the stage you broke.

Run the whole lifecycle (upload → train → compare → deploy → predict → feedback) against the running stack:

```bash
cd backend && python scripts/demo_lifecycle.py
```

> Upgrading from a pre-Phase-1.5 checkout? Tables used to be created with `create_all`; reset the
> local database once with `docker compose down -v` so Alembic can own the schema.

### Running tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/unit                                    # pure-Python, no services needed

# integration tests need a throwaway Postgres (tables are truncated!)
docker run -d --name mlops_test_pg -e POSTGRES_USER=mlops -e POSTGRES_PASSWORD=mlops \
  -e POSTGRES_DB=mlops_test -p 5433:5432 postgres:16-alpine
TEST_DATABASE_URL=postgresql+asyncpg://mlops:mlops@localhost:5433/mlops_test pytest tests/
```

---

## 🏗️ Local Stack — All 8 Services

```mermaid
flowchart LR
    subgraph LOCAL["docker-compose up"]
        FE["🌐 Frontend\nVite :3000"]
        API["⚡ FastAPI\n:8000"]
        WK["⚙️ Celery Worker\ntraining·drift·retrain"]
        BT["⏰ Celery Beat\nscheduler"]
        FL["🌸 Flower\n:5555"]
        PG[("🐘 Postgres\n:5432")]
        RD[("⚡ Redis\n:6379")]
        MN[("☁️ MinIO\n:9000/:9001")]
    end

    FE -->|REST| API
    API --> PG & RD & MN
    API -->|enqueue| WK
    WK --> PG & MN
    BT -->|trigger| WK
    WK --> FL
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite + TypeScript | Fast dev loop, type-safe API contracts |
| **Styling** | Tailwind CSS v3 | Dark-mode design system |
| **Charts** | Recharts | Drift trend time-series |
| **Server state** | TanStack Query | Caching, auto-refresh |
| **API** | FastAPI (Python 3.11) | Async, auto OpenAPI docs |
| **ORM** | SQLAlchemy async + Alembic | DB migrations |
| **Task queue** | Celery + Redis | Non-blocking training/drift jobs |
| **Drift batch** | `scipy.stats.ks_2samp` | Feature distribution comparison |
| **Error drift** | one-sided two-proportion test | Error rate vs test-set baseline (needs ground truth) |
| **ML** | scikit-learn + XGBoost | Classification + regression |
| **Storage** | MinIO (local) / AWS S3 (prod) | Datasets + model artifacts |
| **Database** | PostgreSQL 16 | Metadata, drift events, audit |
| **IaC** | Terraform (AWS) | VPC, RDS, S3, ECS, Cognito |
| **CI/CD** | GitHub Actions | Lint → test → staging → prod |

---

## 🗄️ Data Model (11 Tables)

```mermaid
erDiagram
    datasets ||--o{ experiment_groups : "trains"
    experiment_groups ||--o{ experiment_runs : "one per model"
    registered_models ||--o{ model_versions : "lineage"
    experiment_runs ||--o{ model_versions : "produces"
    model_versions ||--o{ predictions : "generates"
    model_versions ||--o{ drift_events : "monitored by"
    model_versions ||--o{ retrain_jobs : "triggers"

    datasets {
        uuid    id
        varchar name
        varchar s3_path
        jsonb   baseline_stats
    }
    registered_models {
        uuid    id
        varchar name
        varchar target_column
        varchar task_type
    }
    experiment_groups {
        uuid    id
        varchar target_column
        varchar status
        text    error
    }
    experiment_runs {
        uuid    id
        varchar model_type
        jsonb   metrics
        varchar status
    }
    model_versions {
        uuid    id
        varchar version_tag
        varchar status
        varchar sla_tier
    }
    predictions {
        uuid      id
        jsonb     input_snapshot
        jsonb     features_snapshot
        jsonb     output
        jsonb     actual
        float     latency_ms
        timestamp predicted_at
    }
    drift_events {
        uuid      id
        varchar   stage
        float     drift_score
        varchar   detector_type
        timestamp onset_timestamp
    }
    attribution_reports {
        uuid  id
        jsonb ranked_stages
        float confidence
    }
    retrain_jobs {
        uuid    id
        float   cost_estimate
        varchar status
        uuid    approved_by
    }
    audit_logs {
        uuid    id
        varchar action
        varchar actor_email
        jsonb   metadata
    }
```


---

## 🚦 CI/CD Pipeline

```mermaid
flowchart LR
    PUSH[📝 git push] --> SPLIT{Changed?}
    SPLIT -->|backend/**| BCI[Backend CI\nblack·flake8·pytest\n≥60% coverage]
    SPLIT -->|frontend/**| FCI[Frontend CI\neslint·tsc·vitest\nvite build]

    BCI -->|main only| ECR[Push to ECR]
    FCI -->|main only| S3F[Upload to S3]

    ECR & S3F --> STG[🚀 Deploy Staging\nECS + CloudFront]
    STG --> INT[Integration Tests\nvs. live staging]
    INT -->|✅ pass| GATE[⏸ Manual Approval\nGitHub Environments]
    INT -->|❌ fail| ROLL[🔙 Auto-Rollback\n+ Slack Alert]
    GATE -->|👤 approved| PROD[🚀 Deploy Production]
```

---

## 📋 Build Phases

| Phase | Goal | Status |
|---|---|---|
| **Phase 0** — Foundation | Scaffold, Docker Compose, CI/CD, Architecture | ✅ **Complete** |
| **Phase 1** — Core Lifecycle | Upload → Train → Compare → Deploy → Predict | ✅ **Complete** (hardened in 1.5) |
| **Phase 1.5** — Make it real | Model bundles, registry lineages, 3-stage prediction logging, auth UI, integration tests | ✅ **Complete** |
| **Phase 2** — Observability ⭐ | Stage drift detection + root-cause attribution | ✅ **Complete** |
| **Phase 3** — Retraining Loop | Cost-gated, human-approved closed-loop retrain | 🔲 Planned |
| **Phase 4** — Hardening | Auth/RBAC, SLA tiers, multi-tenant dashboard | 🔲 Planned |

> ⚠️ **Phase 2 is non-negotiable** — it is the core differentiator. Without it, this is just another MLOps CRUD app.

---

## 📡 API Surface

Interactive docs: **http://localhost:8000/docs** | Spec: [`docs/api-spec.yaml`](docs/api-spec.yaml)

```
POST   /auth/login                           Authenticate → JWT
POST   /datasets                             Upload + schema-validate CSV
POST   /experiments/train                    Kick off multi-model training (async)
GET    /experiments/{id}/status              Poll training progress (per model, with errors)
GET    /experiments/{id}/compare             Side-by-side metrics comparison
GET    /models/registered                    Model lineages + their production versions
POST   /models/{id}/deploy                   Promote to production (per lineage)
POST   /models/{id}/predict                  Inference (p95 < 300ms) — logs S1/S2/S3
POST   /predictions/{id}/feedback            Attach ground truth to a prediction
GET    /observability/overview           ⭐  Every production model's drift health
GET    /observability/{id}/drift         ⭐  Stage-wise drift status + score trend
GET    /observability/{id}/attribution   ⭐  Root-cause attribution reports
POST   /observability/{id}/check         ⭐  Run a drift check now
POST   /observability/{id}/simulate      ⭐  Inject a fault at a stage (demo/testing)
POST   /retrain/{id}/trigger                 Request retraining
POST   /retrain/{job_id}/approve             Human approval gate
GET    /audit-logs                           Filterable audit trail
```

---

## 📁 Repository Structure

```
Drift_Detection/
├── frontend/                   # React 18 + Vite + TypeScript
│   ├── src/
│   │   ├── api/client.ts       # Axios + JWT interceptor
│   │   ├── components/
│   │   │   └── Layout.tsx      # Sidebar + dark-mode shell
│   │   └── pages/              # 5 pages (Phase 1–3)
│   └── tailwind.config.js      # Brand design system
│
├── backend/
│   ├── app/
│   │   ├── core/               # Config · DB · Security
│   │   ├── models/             # 9 SQLAlchemy ORM tables
│   │   ├── api/                # 7 FastAPI routers
│   │   ├── services/drift/  ⭐ # stage_monitors · attribution · alert
│   │   └── workers/            # Celery (3 queues + Beat)
│   ├── alembic/                # Async DB migrations
│   └── Dockerfile              # Multi-stage: api / worker / beat
│
├── infra/terraform/            # AWS IaC (VPC·S3·RDS·ECS·Cognito)
├── .github/workflows/          # backend-ci · frontend-ci · deploy
├── docs/
│   ├── architecture.jpg     ← this diagram
│   ├── ARCHITECTURE.md         # Full Mermaid diagrams
│   └── api-spec.yaml           # OpenAPI 3.1
└── docker-compose.yml          # 8-service local stack
```

---

## 🤝 Contributing

```bash
git checkout -b feature/fr{N}-description   # one branch per requirement
# e.g. feature/fr8-stage-monitors  ← maps directly to FR8

git push origin feature/fr8-stage-monitors
# open PR → CI must pass → squash-merge
```

---

<div align="center">

MIT © 2026 Dinesh Kumar &nbsp;|&nbsp; [Architecture Docs](docs/ARCHITECTURE.md) &nbsp;|&nbsp; [API Spec](docs/api-spec.yaml)

</div>
