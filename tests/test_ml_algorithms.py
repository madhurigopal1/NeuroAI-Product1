# Simple test to run example algorithms and ensure they return metrics
import importlib
import pkgutil
import ml_algorithms


def test_some_algorithms_run():
    mods = [m.name for m in pkgutil.iter_modules(ml_algorithms.__path__)]
    for name in mods[:3]:
        mod = importlib.import_module(f"ml_algorithms.{name}")
        if hasattr(mod, 'train_and_eval'):
            res = mod.train_and_eval()
            assert isinstance(res, dict)
