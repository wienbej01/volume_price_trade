def test_imports():
    import importlib
    base = "volume_price_trade"
    mods = [
        f"{base}.data.gcs_loader", f"{base}.data.calendar",
        f"{base}.features.volume_profile", f"{base}.features.vpa", f"{base}.features.ict",
        f"{base}.features.time_of_day", f"{base}.labels.targets", f"{base}.ml.dataset",
        f"{base}.ml.cv", f"{base}.backtest.engine"
    ]
    for m in mods:
        importlib.import_module(m)
