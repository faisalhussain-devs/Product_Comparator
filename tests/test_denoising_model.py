import subprocess
import pytest
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/label_data.ipynb")
EXECUTED_NOTEBOOK_PATH = Path("notebooks/label_data_test_executed.ipynb")
TRAINER_NOTEBOOK = Path("notebooks/denoising.ipynb")
EXECUTED_TRAINER_NOTEBOOK = ("notebooks/test_denoising.ipynb")

@pytest.mark.integration
def test_run_gemini_pipeline_notebook():
    """
    Run the complete Gemini pipeline notebook end-to-end.
    Saves the executed notebook to a separate file for inspection if needed.
    """
    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", str(EXECUTED_NOTEBOOK_PATH),
            str(NOTEBOOK_PATH)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0, "Gemini pipeline notebook failed to execute."


@pytest.mark.integration
def test_run_trainer_notebook():
    """
    Integration test: Executes the trainer_usefulness notebook end-to-end
    to ensure the training pipeline runs without errors.

    This will:
      1. Run all cells in the notebook.
      2. Save an executed copy with outputs.
      3. Fail the test if execution errors occur.
    """
    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", str(EXECUTED_TRAINER_NOTEBOOK),
            str(TRAINER_NOTEBOOK)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0, "Trainer notebook execution failed"



# tests/test_end_to_end.py
import pytest
import subprocess
from pathlib import Path

# Paths
GEMINI_NOTEBOOK = Path("notebooks/gemini_pipeline.ipynb")
TRAINER_NOTEBOOK = Path("notebooks/trainer_notebook.ipynb")
EXECUTED_GEMINI = Path("notebooks/gemini_pipeline_executed.ipynb")
EXECUTED_TRAINER = Path("notebooks/trainer_notebook_executed.ipynb")

@pytest.mark.slow
@pytest.mark.endtoend
def test_full_pipeline(tmp_path):
    """
    End-to-end test:
    1. Run Gemini labeling notebook.
    2. Run Trainer notebook (uses Gemini output).
    3. Assert trainer produces expected artifacts (e.g., model or metrics).
    """

    # 1. Run Gemini labeling notebook
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
    assert result_gemini.returncode == 0, f"Gemini notebook failed:\n{result_gemini.stderr}"

    # 2. Run Trainer notebook
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
    assert result_trainer.returncode == 0, f"Trainer notebook failed:\n{result_trainer.stderr}"

    # 3. Check trainer output artifact (example: model file or metrics file)
    model_path = Path("outputs/final_model")
    assert model_path.exists(), "Expected trained model not found!"

    metrics_path = Path("outputs/training_metrics.json")
    assert metrics_path.exists(), "Expected metrics file not found!"

