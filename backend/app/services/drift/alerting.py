"""
Drift alerts (FR11): Slack incoming-webhook message with the ranked attribution report.

Without SLACK_WEBHOOK_URL the alert is written to the log instead, so local
demos still show what would have been sent. Repeat suppression (cooldown) is
handled by the caller using attribution_reports.alert_sent_at.
"""

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.services.drift.attribution_engine import LABELS, AttributionResult

logger = logging.getLogger(__name__)


def build_message(model_label: str, sla_tier: str, result: AttributionResult, dashboard_url: str | None) -> dict:
    lines = [
        f"*:rotating_light: Drift detected — {model_label}* (SLA: {sla_tier})",
        result.explanation,
        "",
        "*Ranked root causes*",
    ]
    for i, r in enumerate(result.ranked, start=1):
        onset = f", onset {r['onset'][11:19]} UTC" if r.get("onset") else ""
        lines.append(
            f"{i}. {LABELS[r['stage']]} — confidence {r['confidence']:.0%} "
            f"(magnitude {r['magnitude']:.2f}{onset}; top feature: {r.get('top_feature') or 'n/a'})"
        )
    if dashboard_url:
        lines += ["", f"<{dashboard_url}|Open observability dashboard>"]
    return {"text": "\n".join(lines)}


async def send_alert(message: dict[str, Any]) -> str:
    """Deliver the alert. Returns the channel used: 'slack', 'log', or 'failed'."""
    if not settings.SLACK_WEBHOOK_URL:
        logger.warning("DRIFT ALERT (no SLACK_WEBHOOK_URL configured):\n%s", message["text"])
        return "log"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.post(settings.SLACK_WEBHOOK_URL, json=message)
            res.raise_for_status()
        return "slack"
    except httpx.HTTPError:
        logger.exception("Slack alert delivery failed")
        return "failed"
