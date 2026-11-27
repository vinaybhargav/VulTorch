import subprocess
import sys
import os
import pytest

EXAMPLES = [
    "01_quickstart_mlp.py",
    "02_cnn_classifier.py",
    "03_transformer_amp.py",
    "04_mamba_and_snake.py",
    "05_multigpu_compile.py",
    "06_quantized_inference.py",
]


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_example_runs(example_name):
    script_path = os.path.join(os.path.dirname(__file__), "..", "examples", example_name)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    res = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Example {example_name} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
