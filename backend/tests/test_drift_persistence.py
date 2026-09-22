from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.drift_persistence import save_drift_event


@pytest.mark.asyncio
async def test_save_drift_event():
    db = MagicMock()
    db.flush = AsyncMock()

    drift_result = {
        "stage": "S1_ingestion",
        "feature_name": "temperature",
        "drift_detected": True,
        "drift_score": 1.0,
        "threshold": 0.05,
    }

    event = await save_drift_event(
        db=db,
        model_version_id="test-model-id",
        drift_result=drift_result,
    )

    assert event.model_version_id == "test-model-id"
    assert event.stage == "S1_ingestion"
    assert event.drift_score == 1.0
    assert event.detector_type == "ks_test"
    assert event.feature_name == "temperature"
    assert event.threshold_used == 0.05

    db.add.assert_called_once_with(event)
    db.flush.assert_awaited_once()
