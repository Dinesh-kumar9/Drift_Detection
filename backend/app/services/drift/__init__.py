"""
Stage-wise drift observability (Phase 2).

    baselines.py          reference samples per stage, captured at training time
    detectors.py          per-feature statistical tests (KS, chi-square/TVD, proportions)
    stage_monitors.py     S1 ingestion / S2 preprocessing / S3 output monitors
    onset.py              when did the current drift episode start?
    attribution_engine.py rank stages by magnitude × onset × causal consistency
    fault_injection.py    synthetic traffic with a fault at a chosen stage (demo + tests)
    service.py            run one check end-to-end and persist it
    alerting.py           Slack notification with the ranked report
"""

STAGES = ("S1_ingestion", "S2_preprocessing", "S3_output")
