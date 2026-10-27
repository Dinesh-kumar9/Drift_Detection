# Stage-wise Drift Detection & Root-Cause Attribution

How the observability engine decides **that** a deployed model is drifting and
**where in the pipeline** the problem started, and how well it does that.

Code: `backend/app/services/drift/` · Evaluation: `backend/scripts/evaluate_attribution.py`

---

## 1. The three monitored stages

Every prediction is logged with all three stages, so each can be compared with
what the model saw during training:

| Stage | What is logged | Reference ("normal") | Tests |
|---|---|---|---|
| **S1 ingestion** | raw request features | up to 2,000 training rows | KS per numeric column, χ² + total-variation distance per categorical column, null-rate shift per column |
| **S2 preprocessing** | features after the fitted pipeline | the same rows, transformed | KS / χ² per model feature **+ pipeline consistency** (below) |
| **S3 output** | prediction, confidence, ground truth (via feedback) | model predictions on the held-out test split | predicted-class χ² / prediction KS, confidence KS, **error-rate increase** (one-sided) |

The S3 reference uses the test split, not training rows: predictions on rows
the model has already seen are unrealistically confident and accurate.

**Pipeline consistency (S2).** The engine re-runs the training-time transform
on the logged raw inputs and compares the result with the features the model
actually received. If the two disagree, the preprocessing step itself
changed. That evidence separates "the data changed" from "the pipeline broke",
which a distribution test alone cannot do.

## 2. When is a stage "drifted"?

Each test yields an effect size in [0, 1] and a p-value. A test signals drift
only if **both** hold:

- the p-value is below α / k (Bonferroni across the stage's k tests, α = 0.05), and
- the effect size is ≥ 0.1.

With thousands of predictions, negligible shifts become statistically
significant. The effect-size floor is what keeps the false-alarm rate at 0%
(see results). A stage's score is its largest effect size.

## 3. Onset — when did it start?

The check window (default 60 min, ≤ 5,000 predictions) is split into 6
equal-count time buckets, and each bucket is scored against the reference.
The onset is the start of the **latest drift episode**: walking back from the
most recent drifted bucket, an episode continues across at most one clean
bucket. A single noisy bucket early in the window therefore does not set the
onset.

## 4. Attribution — which stage caused it?

```
score(stage) = magnitude × temporal_weight × causal_factor

temporal_weight = 1 / (1 + onset_lag / bucket_width)     earliest stage → 1.0
confidence      = score / Σ scores                        ties → upstream stage
```

The temporal weight is a bounded version of "1 / onset lag", which divides by
zero for the earliest stage.

**Why a causal factor is needed.** All three stages are captured in the same
request, so a real upstream change shows up at S1, S2 and S3 *at the same
moment*. The KS statistic is also unchanged by scaling, so S1 and S2 report
*the same magnitude*. Magnitude and onset alone therefore can't separate a
stage from what it passes downstream. The causal factor discounts downstream
drift that is fully explained upstream:

| Evidence | Interpretation | Factor |
|---|---|---|
| S1 drifted | upstream data changed | 1.0 |
| S2 drifted + pipeline divergence | preprocessing fault | 1.0 |
| S2 drifted, S1 drifted, no divergence | inherited from S1 | 0.3 |
| S3 error rate rose, inputs stable | concept drift | 1.0 |
| S3 error rate rose, inputs also shifted | partly inherited | 0.6 |
| S3 shifted only because inputs shifted | inherited | 0.3 |

When two genuine faults overlap (for example an upstream shift *and* a
preprocessing bug), neither is discounted. They share the confidence, and the
one that started earlier ranks first.

## 5. Evaluation

`python scripts/evaluate_attribution.py --trials 20` trains a model on
synthetic fraud data, generates checks of 600 predictions over 30 minutes, and
injects a known fault into the last half (`app/services/drift/fault_injection.py`).
Each row below is 20 independent trials.

| Fault | Severity | Detected | Correct root cause | Median onset error (min) |
|---|---|---|---|---|
| none (healthy) | — | 0% false alarms | — | — |
| S1 raw-input shift (`age`) | 0.25σ | 0% | 0% | — |
| | 0.5σ | 10% | 10% | 0.0 |
| | 1σ | 100% | 100% | 0.0 |
| | 2σ | 100% | 100% | 0.0 |
| | 3σ | 100% | 100% | 0.0 |
| S1 column arrives empty (`channel`) | 5% of traffic | 0% | 0% | — |
| | 10% | 100% | 100% | 2.0 |
| | 20% | 100% | 100% | 4.0 |
| | 35% | 100% | 100% | 4.5 |
| | 50% | 100% | 100% | 0.0 |
| S2 preprocessing bug (`amount`) | 0.25σ | 100% | 100% | 0.0 |
| | 0.5σ – 3σ | 100% | 100% | 0.0 |
| S3 concept drift (labels flipped) | 5% of labels | 0% | 0% | — |
| | 10% | 15% | 15% | 2.0 |
| | 20% | 100% | 100% | 4.0 |
| | 35% | 100% | 100% | 0.5 |
| | 50% | 100% | 100% | 0.0 |

**Reading the results**

- **No false alarms** on healthy traffic.
- **Whenever drift was detected, the named root cause was correct**: the two
  columns match in every row. Missed detections come from the effect-size
  floor, never from attribution mistakes.
- **Preprocessing faults are caught at any size**, because pipeline
  consistency is a direct check, not a statistical estimate.
- **Detection limits:** raw-input shifts from about 1σ, empty-column floods from
  about 10% of traffic, and concept drift from about 20% of labels. Subtler faults
  are diluted by the healthy half of the window, and lowering the effect-size
  floor would trade them for false alarms.
- **Onset** lands within one bucket (5 min) of the true fault start.

**Limitations**

- The data is synthetic. Real data has correlated features and seasonality,
  which this evaluation does not cover.
- Concept-drift detection depends on ground truth arriving through
  `POST /predictions/{id}/feedback`.
- Onset resolution is one bucket (window / 6), so faults that start within the
  same bucket get the same onset.

## 6. Operating it

| Setting | Default | Meaning |
|---|---|---|
| `DRIFT_KS_P_VALUE_THRESHOLD` | 0.05 | family-wise α per stage |
| `DRIFT_MIN_EFFECT` | 0.1 | minimum effect size to call drift |
| `DRIFT_WINDOW_MINUTES` / `DRIFT_MAX_SAMPLES` | 60 / 5000 | traffic examined per check |
| `DRIFT_MIN_SAMPLES` | 50 | checks are skipped below this |
| `DRIFT_INTERVAL_{CRITICAL,STANDARD,LOW}` | 60 / 300 / 1800 s | check cadence by SLA tier |
| `ALERT_COOLDOWN_MINUTES` | 30 | repeat-alert suppression per model and root cause |
| `SLACK_WEBHOOK_URL` | unset | alerts are logged when unset |
| `ENABLE_FAULT_INJECTION` | true | turn **off** outside demo/staging |

API: `GET /observability/overview`, `GET /observability/{id}/drift`,
`GET /observability/{id}/attribution`, `POST /observability/{id}/check`,
`POST /observability/{id}/simulate`.
