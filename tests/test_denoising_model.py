import subprocess
import pytest
from pathlib import Path

GEMINI_NOTEBOOK = Path("notebooks/gemini_pipeline.ipynb")
TRAINER_NOTEBOOK = Path("notebooks/trainer_notebook.ipynb")
EXECUTED_GEMINI = Path("notebooks/gemini_pipeline_executed.ipynb")
EXECUTED_TRAINER = Path("notebooks/trainer_notebook_executed.ipynb")

@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """
    End-to-end test:
    1. Run Gemini labeling notebook.
    2. Run Trainer notebook (uses Gemini output).
    3. Assert trainer produces expected artifacts (model + checkpoints).
    """

    result_gemini = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", str(EXECUTED_GEMINI),
            str(GEMINI_NOTEBOOK)
        ],
        capture_output=True,
        text=True
    )
    if result_gemini.returncode != 0:
        raise RuntimeError(
            f"Gemini notebook failed.\n\nSTDOUT:\n{result_gemini.stdout}\n\nSTDERR:\n{result_gemini.stderr}"
        )

    result_trainer = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", str(EXECUTED_TRAINER),
            str(TRAINER_NOTEBOOK)
        ],
        capture_output=True,
        text=True
    )
    if result_trainer.returncode != 0:
        raise RuntimeError(
            f"Trainer notebook failed.\n\nSTDOUT:\n{result_trainer.stdout}\n\nSTDERR:\n{result_trainer.stderr}"
        )

    model_path = Path("notebooks/result/final_model")
    assert model_path.exists(), "Expected trained model not found!"

    checkpoints = list(Path("notebooks/results/denoise_bert").glob("checkpoint-*"))
    assert len(checkpoints) > 0, "Expected checkpoints not found!"


