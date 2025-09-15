import pathlib

import yaml


def test_config_has_tickers():
    cfg = yaml.safe_load(pathlib.Path("config/base.yaml").read_text())
    assert "tickers" in cfg and "train" in cfg["tickers"] and len(cfg["tickers"]["train"]) >= 40
