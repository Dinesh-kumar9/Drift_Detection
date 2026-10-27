"""Shared helpers for integration tests."""

import io

import pandas as pd
from httpx import AsyncClient


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


async def upload(client: AsyncClient, headers: dict, df: pd.DataFrame, name: str) -> dict:
    res = await client.post(
        "/datasets/",
        headers=headers,
        files={"file": (f"{name}.csv", csv_bytes(df), "text/csv")},
        data={"name": name},
    )
    assert res.status_code == 201, res.text
    return res.json()
