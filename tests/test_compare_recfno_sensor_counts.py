import numpy as np
import pytest

from scripts.compare_recfno_sensor_counts import paired_stats


def test_paired_difference_and_direction():
    fixed = np.array([1., 2., 3., 4.])
    result = paired_stats(fixed, fixed + 0.5)
    assert result["change_percent"] == pytest.approx(20)
    assert result["paired_mean_difference"] == pytest.approx(0.5)
    assert result["paired_difference_ci95_low"] == pytest.approx(0.5)
    assert result["paired_difference_ci95_high"] == pytest.approx(0.5)


def test_paired_statistics_rejects_missing_and_nonfinite_errors():
    with pytest.raises(ValueError):
        paired_stats(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        paired_stats(np.ones(2), np.array([1., np.nan]))
