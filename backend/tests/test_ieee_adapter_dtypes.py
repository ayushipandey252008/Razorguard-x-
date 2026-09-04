"""IEEE-CIS CSV loader dtypes. Reproduces Found/NotFound identity sentinels.

Does not use the synthetic fixture as IEEE-CIS results. Does not set IEEE_MAX_ROWS.
Does not modify source CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ml.ieee.adapter import _read_csv, column_dtype_map, load_tables
from app.ml.ieee.constants import (
    DEFAULT_DATA_DIR,
    ID_FILENAME,
    IDENTITY_CATEGORICAL_COLUMNS,
    IDENTITY_NUMERIC_COLUMNS,
    TXN_FILENAME,
)

REAL_IDENTITY = DEFAULT_DATA_DIR / ID_FILENAME
REAL_TRANSACTION = DEFAULT_DATA_DIR / TXN_FILENAME


def test_dtype_map_does_not_force_identity_categoricals_to_float32():
    cols = [
        "TransactionID",
        "id_01",
        "id_02",
        "id_12",
        "id_16",
        "id_27",
        "id_29",
        "id_30",
        "id_31",
        "DeviceType",
        "DeviceInfo",
        "C1",
        "D1",
        "V1",
        "TransactionAmt",
    ]
    mapping = column_dtype_map(cols)
    for col in IDENTITY_NUMERIC_COLUMNS:
        if col in mapping:
            assert mapping[col] == "float32", col
    for col in ("id_12", "id_16", "id_27", "id_29", "id_30", "id_31", "DeviceType", "DeviceInfo"):
        assert mapping[col] == "object", col
        assert mapping[col] != "float32"
    assert mapping["DeviceType"] != "float32"
    assert mapping["D1"] == "float32"
    assert mapping["C1"] == "float32"
    assert mapping["V1"] == "float32"
    # Prefix bug: Device* must not match D1-style numeric coding.
    assert "DeviceType" in IDENTITY_CATEGORICAL_COLUMNS
    assert "id_12" in IDENTITY_CATEGORICAL_COLUMNS
    assert "id_01" in IDENTITY_NUMERIC_COLUMNS


def test_read_csv_preserves_notfound_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("IEEE_MAX_ROWS", raising=False)
    ident_path = tmp_path / ID_FILENAME
    txn_path = tmp_path / TXN_FILENAME
    ident_path.write_text(
        "TransactionID,id_01,id_02,id_12,id_16,id_29,id_30,DeviceType,DeviceInfo\n"
        "1,-5.0,1000,NotFound,NotFound,NotFound,iOS,mobile,SM-G930\n"
        "2,0.0,2000,Found,Found,Found,Android,desktop,Windows\n"
        "3,2.5,,Found,,,chrome,mobile,\n"
    )
    txn_path.write_text(
        "TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,C1,D1\n"
        "1,0,10,12.5,W,1,0\n"
        "2,1,20,40.0,C,2,1\n"
        "3,0,30,8.0,H,1,\n"
    )
    ident = _read_csv(ident_path, None)
    assert ident.loc[0, "id_12"] == "NotFound"
    assert ident.loc[1, "id_12"] == "Found"
    assert ident.loc[0, "id_16"] == "NotFound"
    assert ident.loc[0, "id_29"] == "NotFound"
    assert ident["id_12"].dtype == object
    assert ident.loc[0, "DeviceType"] == "mobile"
    assert pd.api.types.is_numeric_dtype(ident["id_01"])
    assert float(ident.loc[0, "id_01"]) == -5.0

    txn, loaded_ident, meta = load_tables(tmp_path, max_rows=None)
    assert meta["max_rows"] is None
    assert loaded_ident["id_12"].tolist()[0] == "NotFound"
    assert loaded_ident["id_29"].tolist()[0] == "NotFound"
    # Sentinel is a category token, not coerced to a number or dropped.
    assert not pd.api.types.is_numeric_dtype(loaded_ident["id_12"])
    coerced = pd.to_numeric(loaded_ident["id_12"], errors="coerce")
    assert coerced.isna().all()
    assert txn["TransactionID"].tolist() == [1, 2, 3]


@pytest.mark.skipif(not REAL_IDENTITY.is_file(), reason="real IEEE identity CSV not present")
def test_real_identity_csv_loads_without_float_coercion():
    """Read-only check against the downloaded file. Does not rewrite it or train."""
    ident = _read_csv(REAL_IDENTITY, None)
    assert "id_12" in ident.columns
    assert not pd.api.types.is_numeric_dtype(ident["id_12"])
    assert ident["DeviceType"].dtype == object or ident["DeviceType"].dtype.name in {"object", "string"}
    if ident["id_12"].astype(str).eq("NotFound").any():
        assert (ident["id_12"] == "NotFound").any()
        numeric = pd.to_numeric(ident.loc[ident["id_12"] == "NotFound", "id_12"], errors="coerce")
        assert numeric.isna().all()
    assert pd.api.types.is_numeric_dtype(ident["id_01"])
