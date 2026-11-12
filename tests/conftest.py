import pytest

from parking_mlops import config
from parking_mlops.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once into a temp folder so tests never depend on committed artifacts."""
    out = tmp_path_factory.mktemp("models")
    model_path, metrics_path = out / "model.joblib", out / "metrics.json"
    metrics = train(config.DATA_PATH, model_path, metrics_path)
    return {"metrics": metrics, "model_path": model_path}
