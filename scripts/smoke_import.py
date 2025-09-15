import importlib
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

base = ROOT / "config/base.yaml"
cfg = yaml.safe_load(base.read_text())
assert "tickers" in cfg and "train" in cfg["tickers"], "Tickers missing in config."

# third-party libs
for mod in [
    "pandas",
    "numpy",
    "pyarrow",
    "gcsfs",
    "sklearn",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "yaml",
]:
    importlib.import_module(mod)

# our package imports
pk = "volume_price_trade"
mods = [
    f"{pk}.data.gcs_loader",
    f"{pk}.data.calendar",
    f"{pk}.features.volume_profile",
    f"{pk}.features.vpa",
    f"{pk}.features.ict",
    f"{pk}.features.time_of_day",
    f"{pk}.labels.targets",
    f"{pk}.ml.dataset",
    f"{pk}.ml.cv",
    f"{pk}.backtest.engine",
]
for m in mods:
    importlib.import_module(m)

print("Smoke import OK.")
