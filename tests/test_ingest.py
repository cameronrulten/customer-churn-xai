from __future__ import annotations

import io
import types
from pathlib import Path

import pandas as pd
import pytest

# Import the module under test
from src.data.ingest import (
    validate_csv_shape,
    load_config,
    try_urls,
    stream_csv_bytes,  # we'll monkeypatch this in one test
)

def test_validate_csv_shape_ok():
    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "Churn": ["Yes", "No"],
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    out = validate_csv_shape(csv_bytes, expected_rows=2, expected_cols=2)
    assert out.equals(df)

def test_validate_csv_shape_bad_shape():
    df = pd.DataFrame({"customerID": ["0001"], "Churn": ["Yes"]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    with pytest.raises(ValueError):
        validate_csv_shape(csv_bytes, expected_rows=2, expected_cols=2)

def test_load_config_minimal(tmp_path: Path):
    cfg_text = """
    dataset:
      name: telco
      filename: telco.csv
      expected_rows: 2
      expected_cols: 2
      urls: ["https://example.com/a.csv"]
    paths:
      raw_dir: data/raw
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)
    assert cfg["dataset"]["name"] == "telco"

def test_try_urls_uses_first_success(monkeypatch):
    # Prepare two dummy CSVs; first succeeds.
    good_df = pd.DataFrame({"customerID": ["1", "2"], "Churn": ["Yes", "No"]})
    good_csv = good_df.to_csv(index=False).encode()

    def fake_stream(url):
        # emulate first URL returning good CSV
        return good_csv

    monkeypatch.setattr("src.data.ingest.stream_csv_bytes", fake_stream)
    df = try_urls(["http://u1", "http://u2"], expected_rows=2, expected_cols=2)
    assert df.equals(good_df)

def test_try_urls_all_fail(monkeypatch):
    def fake_stream(url):
        raise RuntimeError("download failed")

    monkeypatch.setattr("src.data.ingest.stream_csv_bytes", fake_stream)
    with pytest.raises(RuntimeError):
        _ = try_urls(["http://u1", "http://u2"], expected_rows=2, expected_cols=2)
