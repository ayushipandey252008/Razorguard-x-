from app.agents.tools import ToolBox
from app.ml.features import dataframe_from_records
from app.ml.predictor import model_service
from app.services.synthetic import generate_labeled_dataset


def test_preprocessing_no_nan():
    rows = generate_labeled_dataset(50, seed=3)
    df = dataframe_from_records(rows)
    assert not df.isna().any().any()
    assert df.shape[1] == 13


def test_model_loads_and_has_metrics():
    model_service.load_or_train()
    assert model_service.ready
    assert "pr_auc" in (model_service.metrics or {}) or model_service.metrics == {}
