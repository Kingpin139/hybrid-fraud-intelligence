import importlib.util
from pathlib import Path


def test_inference_demo_runs(capsys):
    path = Path(__file__).resolve().parents[1] / "examples" / "inference_demo.py"
    spec = importlib.util.spec_from_file_location("inference_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    assert "Demo completed successfully" in capsys.readouterr().out
