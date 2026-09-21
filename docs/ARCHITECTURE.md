# Architecture — Multi-Model MLOps Observability Platform

> **Version:** 0.1.0 | **Last Updated:** 2026-09 | **Phase:** 0 (Foundation)

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [High-Level Component Diagram](#2-high-level-component-diagram)
3. [Observability Engine Detail](#3-observability-engine-detail)
4. [Request Flow — Training to Deployment](#4-request-flow--training-to-deployment)
5. [Drift Detection & Attribution Flow](#5-drift-detection--attribution-flow)
6. [CI/CD Pipeline](#6-cicd-pipeline)
7. [Database Schema](#7-database-schema)
8. [Repository Structure](#8-repository-structure)
9. [Local Development Stack](#9-local-development-stack)
10. [AWS Production Stack](#10-aws-production-stack)
11. [Tech Stack Decisions](#11-tech-stack-decisions)

---

## 1. System Overview

The platform covers the full ML model lifecycle across **five functional domains**:

| Domain | Core Capability | Key FR |
|---|---|---|
| **Ingestion & Storage** | CSV upload, schema validation, versioned datasets | FR1 |
| **Training Pipeline** | Parallel multi-model training, experiment tracking | FR2–FR5 |
| **Serving** | Containerised REST inference with request logging | FR6–FR7 |
| **Observability** | Stage-wise drift detection + root-cause attribution | FR8–FR11 |
| **Retraining Loop** | Cost-gated, human-approved closed-loop retraining | FR12–FR17 |

**Core differentiator:** Most MLOps tools flag "drift detected" at the output only. This platform independently monitors **three stages** (raw ingestion → preprocessed features → model predictions) and uses a **temporal-magnitude attribution engine** to rank which stage is the most likely root cause.

---

## 2. High-Level Component Diagram

```mermaid
flowchart TB
    subgraph Client["Frontend (React 18 + Vite + TypeScript)"]
        direction LR
        UI1[📤 Dataset Upload]
        UI2[🧪 Experiment Dashboard]
        UI3[📊 Model Comparison]
        UI4[📡 Observability Dashboard]
        UI5[✅ Retrain Approval Console]
    end

    subgraph API["Backend API Layer (FastAPI / Python 3.11)"]
        direction TB
        AUTH["🔐 Auth Service\n(JWT → Cognito Phase 4)"]
        ING["📥 Ingestion Service\n(schema validation, S3 upload)"]
        PRE["⚙️ Preprocessing Service\n(versioned pipelines)"]
        TRAIN["🏋️ Training Service\n(Celery async jobs)"]
        REG["📦 Model Registry\n(staging/production/archived)"]
        SERVE["🚀 Serving/Inference\n(predict + log)"]
        OBS["📡 Observability Service\n(KS + ADWIN monitors)"]
        RETRAIN["🔄 Retrain Orchestrator\n(cost-gate + approval)"]
    end

    subgraph Data["Data & Storage Layer"]
        S3[("☁️ S3 / MinIO\ndatasets · models · artifacts")]
        RDS[("🐘 Postgres\nmetadata · drift events · audit")]
        REDIS[("⚡ Redis\nCelery broker · result cache")]
    end

    subgraph Workers["Async Workers"]
        CELERY["⚙️ Celery Workers\ntraining · drift · retrain queues"]
        BEAT["⏰ Celery Beat\nscheduled drift checks (5 min)"]
    end

    subgraph Monitor["Monitoring"]
        FLOWER["🌸 Flower\nCelery task monitoring UI"]
        CW["☁️ CloudWatch\ninfra-level logs (staging/prod)"]
    end

    Client -->|"REST / JSON"| API
    API --> Data
    API --> Workers
    Workers --> Data
    OBS --> CELERY
    CELERY --> BEAT
    BEAT --> OBS
    RETRAIN --> REG
    Workers --> FLOWER
    API --> CW
```

---

## 3. Observability Engine Detail

The observability engine is the platform's **core differentiator** — three independent stage monitors feed a temporal-magnitude attribution engine that produces a ranked root-cause report.

```mermaid
flowchart LR
    subgraph Stages["Independent Stage Monitors"]
        direction TB
        S1["S1 — Raw Ingestion Data\nBaseline: training distribution"]
        S2["S2 — Preprocessed Features\nBaseline: preprocessed training data"]
        S3["S3 — Model Predictions\nBaseline: prediction error distribution"]

        S1 -->|"scipy.stats.ks_2samp\n(batch KS test per feature)"| M1["Stage 1 Monitor\ndrift_score · onset_ts"]
        S2 -->|"scipy.stats.ks_2samp\n(batch KS test per feature)"| M2["Stage 2 Monitor\ndrift_score · onset_ts"]
        S3 -->|"river.drift.ADWIN\n(streaming error rate)"| M3["Stage 3 Monitor\ndrift_score · onset_ts"]
    end

    subgraph Attribution["Attribution Engine"]
        ATT["🧠 Root Cause Ranker\nscore = drift_magnitude × 1/onset_lag\nranked [{stage, score, confidence}]"]
    end

    subgraph Action["Response"]
        ALERT["🔔 Alert Service\nSNS → Slack webhook\nranked report attached"]
        GATE["💰 Cost-Benefit Gate\nestimate ROI of retrain"]
        RETRAIN["🔄 Retrain Job\ncreated with 'pending' status"]
        LOG["📋 Log & Suppress\nnot worth retraining"]
    end

    M1 --> ATT
    M2 --> ATT
    M3 --> ATT

    ATT -->|"confidence > threshold"| ALERT
    ATT -->|"drift confirmed"| GATE
    GATE -->|"ROI approved"| RETRAIN
    GATE -->|"ROI rejected"| LOG
    RETRAIN -->|"human approval required"| APPROVAL["👤 Approval Console"]
    APPROVAL -->|"approved"| DEPLOY["🚀 Deploy New Version"]
```

### Attribution Algorithm

```
attribution_score(stage) =
    drift_magnitude(stage) × (1 / onset_lag_seconds(stage))

Where:
  drift_magnitude  = normalised drift score [0.0–1.0]
  onset_lag        = seconds between first threshold crossing and current detection
  
Stages are ranked descending by attribution_score.
The stage that drifted hardest AND earliest ranks #1 (most likely root cause).
```

---

## 4. Request Flow — Training to Deployment

```mermaid
sequenceDiagram
    participant U as 👤 User (Frontend)
    participant API as ⚡ FastAPI
    participant W as 🔧 Celery Worker
    participant S3 as ☁️ MinIO / S3
    participant DB as 🐘 Postgres

    U->>API: POST /datasets (upload CSV)
    API->>S3: Store raw dataset
    API->>DB: Log dataset metadata + baseline stats
    API-->>U: 201 {dataset_id, schema}

    U->>API: POST /experiments/train {dataset_id}
    API->>DB: Create experiment_run (status: queued)
    API->>W: Enqueue training job (Celery)
    API-->>U: 202 {run_id, status: queued}

    W->>S3: Fetch raw dataset
    W->>W: Preprocess (impute → encode → scale)
    W->>S3: Store preprocessed snapshot
    W->>W: Train RandomForest + XGBoost + LogReg (parallel)
    W->>S3: Store 3 model artifacts (.joblib)
    W->>DB: Log metrics per run (accuracy, F1, latency)
    W->>DB: Update experiment_run (status: completed)

    U->>API: GET /experiments/{id}/compare
    API->>DB: Fetch metrics for all 3 models
    API-->>U: Comparison table + charts data

    U->>API: POST /models/{id}/deploy
    API->>DB: Update model_version (status: production)
    API->>DB: Write audit_log (action: model.deploy)
    API-->>U: 200 {deployment_url}

    U->>API: POST /models/{id}/predict {features}
    API->>S3: Load model artifact (cached)
    API->>API: Run inference (< 300ms p95)
    API->>DB: Log prediction (input_snapshot, output, latency_ms)
    API-->>U: {prediction, confidence}
```

---

## 5. Drift Detection & Attribution Flow

```mermaid
sequenceDiagram
    participant BEAT as ⏰ Celery Beat
    participant DRIFT as 📡 Drift Service
    participant DB as 🐘 Postgres
    participant ATT as 🧠 Attribution Engine
    participant ALERT as 🔔 Alert Service
    participant SLACK as 💬 Slack

    BEAT->>DRIFT: Trigger drift check (every 5 min)
    
    DRIFT->>DB: Fetch recent predictions (last window)
    DRIFT->>DB: Fetch training baseline stats

    par Stage 1 Monitor
        DRIFT->>DRIFT: KS test on raw feature distributions
        DRIFT->>DB: Write drift_event (S1, score, onset_ts)
    and Stage 2 Monitor
        DRIFT->>DRIFT: KS test on preprocessed features
        DRIFT->>DB: Write drift_event (S2, score, onset_ts)
    and Stage 3 Monitor
        DRIFT->>DRIFT: ADWIN on prediction error stream
        DRIFT->>DB: Write drift_event (S3, score, onset_ts)
    end

    DRIFT->>ATT: Send all 3 drift events
    ATT->>ATT: Compute attribution scores\n(magnitude × 1/onset_lag)
    ATT->>DB: Write attribution_report (ranked_stages)

    alt Any stage exceeds threshold
        ATT->>ALERT: Trigger alert with ranked report
        ALERT->>SLACK: POST ranked attribution report
        ATT->>DB: Create retrain_job (status: pending)
    else All stages below threshold
        ATT->>DB: Log check (no drift)
    end
```

---

## 6. CI/CD Pipeline

```mermaid
flowchart LR
    A["📝 Push to branch"] --> B{"Which path\nchanged?"}

    B -->|"backend/**"| C["Backend CI"]
    B -->|"frontend/**"| D["Frontend CI"]

    subgraph BackendCI["Backend CI (Python 3.11)"]
        C --> C1["black --check\nflake8 · isort"]
        C1 --> C2["pytest --cov=app\n≥ 60% coverage gate"]
        C2 --> C3["docker build\n--target api"]
        C3 --> C4["Push to ECR\n(main only)"]
    end

    subgraph FrontendCI["Frontend CI (Node 20)"]
        D --> D1["eslint · tsc --noEmit"]
        D1 --> D2["vitest --run"]
        D2 --> D3["vite build"]
        D3 --> D4["Upload artifact\n(main only)"]
    end

    C4 --> E{"Branch?"}
    D4 --> E

    E -->|"main"| F["🚀 Deploy to Staging\n(ECS + S3/CloudFront)"]
    E -->|"feature/*"| G["🛑 Stop\n(no deploy)"]

    F --> H["Integration tests\nvs. staging URL"]
    H -->|"✅ pass"| I["⏸️ Manual Approval\n(GitHub Environments)"]
    H -->|"❌ fail"| K["🔙 Auto-rollback\n+ Slack alert"]
    I -->|"👤 approved"| J["🚀 Deploy to Production"]
```

### Branch Strategy

| Branch | Purpose | Merge policy |
|---|---|---|
| `main` | Always deployable | Require PR + passing CI |
| `feature/fr{N}-{slug}` | One branch per requirement | Squash-merge to `main` |

**Example:** `feature/fr8-stage-monitors` maps directly to FR8.

---

## 7. Database Schema

```mermaid
erDiagram
    users {
        uuid id PK
        varchar email
        varchar hashed_password
        enum role "viewer|engineer|admin"
        timestamp created_at
    }

    datasets {
        uuid id PK
        varchar name
        varchar s3_path
        jsonb schema_json
        jsonb baseline_stats
        timestamp uploaded_at
        uuid owner_id FK
    }

    experiment_runs {
        uuid id PK
        uuid dataset_id FK
        varchar model_type
        jsonb params
        jsonb metrics
        varchar artifact_s3_path
        jsonb preprocessing_config
        enum status "queued|running|completed|failed"
        timestamp created_at
    }

    model_versions {
        uuid id PK
        uuid run_id FK
        varchar version_tag
        enum status "staging|production|archived"
        enum sla_tier "critical|standard|low"
        timestamp deployed_at
    }

    predictions {
        uuid id PK
        uuid model_version_id FK
        jsonb input_snapshot
        jsonb output
        float latency_ms
        timestamp predicted_at
    }

    drift_events {
        uuid id PK
        uuid model_version_id FK
        enum stage "S1_ingestion|S2_preprocessing|S3_output"
        float drift_score
        varchar detector_type
        varchar feature_name
        timestamp onset_timestamp
        timestamp detected_at
    }

    attribution_reports {
        uuid id PK
        uuid[] drift_event_ids
        jsonb ranked_stages
        varchar root_cause_stage
        float confidence
        timestamp generated_at
    }

    retrain_jobs {
        uuid id PK
        uuid model_version_id FK
        jsonb trigger_reason
        float cost_estimate
        float expected_gain
        uuid approved_by
        enum status "pending|approved|rejected|running|completed|failed"
        timestamp created_at
    }

    audit_logs {
        uuid id PK
        uuid actor_id
        varchar actor_email
        varchar action
        varchar target_type
        uuid target_id
        jsonb metadata
        timestamp created_at
    }

    users ||--o{ datasets : "owns"
    datasets ||--o{ experiment_runs : "trains"
    experiment_runs ||--o{ model_versions : "produces"
    model_versions ||--o{ predictions : "generates"
    model_versions ||--o{ drift_events : "monitored_by"
    model_versions ||--o{ retrain_jobs : "triggers"
```

---

## 8. Repository Structure

```
mlops-observability-platform/
│
├── frontend/                        # React 18 + Vite + TypeScript
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts            # Axios instance + JWT interceptor
│   │   ├── components/
│   │   │   ├── Layout.tsx           # Sidebar + header shell
│   │   │   ├── DriftChart.tsx       # Phase 2: Recharts drift trend
│   │   │   └── AttributionPanel.tsx # Phase 2: Root-cause bar chart
│   │   ├── pages/
│   │   │   ├── DatasetUpload.tsx    # Phase 1
│   │   │   ├── ExperimentDashboard.tsx  # Phase 1
│   │   │   ├── ModelComparison.tsx  # Phase 1
│   │   │   ├── ObservabilityDashboard.tsx  # Phase 2
│   │   │   └── RetrainApproval.tsx  # Phase 3
│   │   ├── App.tsx                  # Router + QueryClient
│   │   └── index.css                # Tailwind + design system
│   ├── Dockerfile.dev
│   └── tailwind.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI routers
│   │   │   ├── auth.py
│   │   │   ├── datasets.py          # Phase 1
│   │   │   ├── experiments.py       # Phase 1
│   │   │   ├── models.py            # Phase 1
│   │   │   ├── observability.py     # Phase 2
│   │   │   ├── retrain.py           # Phase 3
│   │   │   └── audit.py             # Phase 3
│   │   ├── core/
│   │   │   ├── config.py            # Pydantic settings
│   │   │   ├── database.py          # Async SQLAlchemy engine
│   │   │   └── security.py          # JWT + RBAC
│   │   ├── models/
│   │   │   └── __init__.py          # All 9 ORM models
│   │   ├── schemas/                 # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── ingestion_service.py     # Phase 1
│   │   │   ├── preprocessing_service.py # Phase 1
│   │   │   ├── training_service.py      # Phase 1
│   │   │   ├── registry_service.py      # Phase 1
│   │   │   ├── serving_service.py       # Phase 1
│   │   │   └── drift/
│   │   │       ├── stage_monitors.py    # Phase 2 ← core differentiator
│   │   │       ├── attribution_engine.py # Phase 2
│   │   │       ├── alert_service.py     # Phase 2
│   │   │       └── cost_benefit_gate.py # Phase 3
│   │   ├── workers/
│   │   │   ├── celery_app.py
│   │   │   ├── training_tasks.py    # Phase 1
│   │   │   ├── drift_tasks.py       # Phase 2
│   │   │   └── retrain_tasks.py     # Phase 3
│   │   └── main.py
│   ├── alembic/                     # DB migrations
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── Dockerfile
│   ├── alembic.ini
│   └── requirements.txt
│
├── infra/
│   └── terraform/
│       ├── vpc.tf
│       ├── s3.tf
│       ├── rds.tf
│       ├── ecs.tf
│       ├── cognito.tf
│       └── variables.tf
│
├── .github/
│   └── workflows/
│       ├── backend-ci.yml           # Lint → test → ECR push
│       ├── frontend-ci.yml          # Lint → typecheck → build
│       └── deploy.yml               # Staging → integration → approval → production
│
├── docs/
│   ├── ARCHITECTURE.md              # ← this file
│   └── api-spec.yaml                # OpenAPI spec
│
├── docker-compose.yml               # Local dev: all 8 services
└── README.md
```

---

## 9. Local Development Stack

All services run via a single `docker-compose up --build`:

| Service | Port | Purpose |
|---|---|---|
| `postgres` | 5432 | PostgreSQL 16 — metadata, drift events, audit logs |
| `redis` | 6379 | Celery broker + result backend |
| `minio` | 9000 | S3-compatible object storage (datasets, models) |
| `minio` console | 9001 | MinIO web UI |
| `backend` (API) | 8000 | FastAPI — Swagger docs at `/docs` |
| `worker` | — | Celery training/drift/retrain workers |
| `beat` | — | Celery Beat — scheduled drift checks |
| `flower` | 5555 | Celery task monitoring UI |
| `frontend` | 3000 | Vite dev server (hot reload) |

```bash
# Quick start
git clone <repo>
cd mlops-observability-platform
cp backend/.env.example backend/.env
docker-compose up --build

# Access points
# Frontend:       http://localhost:3000
# Backend API:    http://localhost:8000/docs
# MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)
# Flower:         http://localhost:5555
```

---

## 10. AWS Production Stack

| Component | AWS Service | Notes |
|---|---|---|
| Object Storage | S3 | Versioned, AES-256 encrypted |
| Database | RDS Postgres 16 | Multi-AZ in production, managed password rotation |
| Cache / Queue | ElastiCache (Redis) | Celery broker |
| Compute | ECS Fargate | API + worker containers, no server management |
| Container Registry | ECR | Image lifecycle policy: keep last 10 |
| Auth | Cognito | Phase 4 — JWT swapped for Cognito tokens |
| CDN | CloudFront | Frontend static bundle |
| Secrets | Secrets Manager | DB credentials, API keys — never in code |
| Alerting | SNS → Slack | Drift attribution reports |
| IaC | Terraform | `infra/terraform/` — define now, provision in Phase 1 |

---

## 11. Tech Stack Decisions

| Layer | Choice | Rationale |
|---|---|---|
| API Framework | FastAPI (Python 3.11) | Async, auto-OpenAPI, typed |
| ORM | SQLAlchemy async + Alembic | Migrations, type-safe queries |
| Task Queue | Celery + Redis | Non-blocking training jobs |
| Scheduler | Celery Beat → Airflow (Phase 2+) | Start simple, DAG complexity when needed |
| Drift (batch) | `scipy.stats.ks_2samp` | Non-parametric, works on any distribution |
| Drift (stream) | `river.drift.ADWIN` | Adaptive windowing, ideal for error rate streams |
| Frontend | React 18 + Vite + TypeScript | Fast dev loop, type safety for API contracts |
| UI | Tailwind CSS v3 | PRD-specified, rapid consistent styling |
| Charts | Recharts | React-native, good for time-series |
| Server state | TanStack Query | Caching, deduplication, no Redux overhead |
| Local S3 | MinIO | S3-compatible API, zero AWS cost for dev |
| Auth (Phase 0–3) | JWT (local) | Simpler; swap for Cognito in Phase 4 |
| IaC | Terraform | Reproducible, version-controlled AWS infra |
| CI/CD | GitHub Actions | PRD-specified |
