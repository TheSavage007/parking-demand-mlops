def test_model_beats_seasonal_baseline(trained):
    m = trained["metrics"]
    assert m["quality_gate_passed"]
    assert m["model"]["mae"] < m["seasonal_baseline"]["mae"]


def test_artifact_written(trained):
    assert trained["model_path"].exists()
