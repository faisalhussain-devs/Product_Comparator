import subprocess
import pytest
from pathlib import Path
import sys

GEMINI_SCRIPT = Path("codes/labelling_data.py")
TRAINER_SCRIPT = Path("codes/denoising_trainer.py")

@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """
    End-to-end test:
    1. Run Gemini labeling script.
    2. Run Trainer script (uses Gemini output).
    3. Assert trainer produces expected artifacts (model + checkpoints).
    """


    result_gemini = subprocess.run(
        [sys.executable, str(GEMINI_SCRIPT)],
        capture_output=True,
        text=True
    )

    if result_gemini.returncode != 0:
        raise RuntimeError(
            f"Gemini pipeline failed.\n\nSTDOUT:\n{result_gemini.stdout}\n\nSTDERR:\n{result_gemini.stderr}"
        )

    result_trainer = subprocess.run(
        [sys.executable, str(TRAINER_SCRIPT)],
        capture_output=True,
        text=True
    )
    if result_trainer.returncode != 0:
        raise RuntimeError(
            f"Trainer script failed.\n\nSTDOUT:\n{result_trainer.stdout}\n\nSTDERR:\n{result_trainer.stderr}"
        )

    model_path = Path("codes/result/final_model")
    assert model_path.exists(), "Expected trained model not found!"

    checkpoints = list(Path("codes/results/denoise_bert").glob("checkpoint-*"))
    assert len(checkpoints) > 0, "Expected checkpoints not found!"



