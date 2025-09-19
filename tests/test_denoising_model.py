# tests/test_gemini_pipeline_real.py
import os
import json
import pytest
from gemini_pipeline import main, OUTPUT_FILE

@pytest.mark.integration
def test_gemini_pipeline_real():
    """
    End-to-end test for Gemini labeling pipeline.
    Runs the real Gemini API on DB reviews and checks that output is generated correctly.
    """
    # Run pipeline (real DB + real Gemini API)
    main()

    # Verify output file exists
    assert os.path.exists(OUTPUT_FILE)

    # Check at least one line of JSON output
    with open(OUTPUT_FILE, "r") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) > 0, "No results written to output file"
    for item in lines:
        assert "id" in item
        assert "text" in item
        assert isinstance(item.get("usefulness_score"), (float, int)), "Invalid usefulness_score"
        assert isinstance(item.get("confidence_score"), (float, int)), "Invalid confidence_score"



# tests/test_gemini_pipeline_real.py
import os
import json
import pytest
from gemini_pipeline import main, OUTPUT_FILE

@pytest.mark.integration
def test_gemini_pipeline_real():
    """
    End-to-end test for Gemini labeling pipeline.
    Runs the real Gemini API on DB reviews and checks that output is generated correctly.
    """
    # Run pipeline (real DB + real Gemini API)
    main()

    # Verify output file exists
    assert os.path.exists(OUTPUT_FILE)

    # Check at least one line of JSON output
    with open(OUTPUT_FILE, "r") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) > 0, "No results written to output file"
    for item in lines:
        assert "id" in item
        assert "text" in item
        assert isinstance(item.get("usefulness_score"), (float, int)), "Invalid usefulness_score"
        assert isinstance(item.get("confidence_score"), (float, int)), "Invalid confidence_score"
