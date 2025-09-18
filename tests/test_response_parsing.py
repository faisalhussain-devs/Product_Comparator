import numpy as np
import pytest
from label_data import bootstrap_mean_ci

def test_bootstrap_mean_ci_standard():
    rng = np.random.default_rng(42)
    scores = [1, 2, 3, 4, 5]
    mean, lo, hi, width = bootstrap_mean_ci(scores, n_boot=100, ci=0.8, rng=rng)
    assert mean == pytest.approx(np.mean(scores))
    assert lo < mean < hi
    assert width == pytest.approx(hi - lo)

def test_bootstrap_mean_ci_few_samples():
    scores = [1, 2]
    mean, lo, hi, width = bootstrap_mean_ci(scores, min_samples=3)
    assert mean == pytest.approx(np.mean(scores))
    assert lo is None
    assert hi is None
    assert width is None

def test_bootstrap_mean_ci_empty():
    mean, lo, hi, width = bootstrap_mean_ci([])
    assert mean is None
    assert lo is None
    assert hi is None
    assert width is None

def test_bootstrap_mean_ci_single_value_corrected():
    scores = [5]
    mean, lo, hi, width = bootstrap_mean_ci(scores, min_samples=1)
    assert mean == 5
    assert lo == 5
    assert hi == 5
    assert width == 0

def test_bootstrap_mean_ci_reproducibility():
    rng = np.random.default_rng(123)
    scores = [1, 2, 3, 4, 5]
    mean1, lo1, hi1, width1 = bootstrap_mean_ci(scores, n_boot=50, rng=rng)   
    rng = np.random.default_rng(123)
    mean2, lo2, hi2, width2 = bootstrap_mean_ci(scores, n_boot=50, rng=rng)
    assert mean1 == mean2
    assert lo1 == lo2
    assert hi1 == hi2
    assert width1 == width2


