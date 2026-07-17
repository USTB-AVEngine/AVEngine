from __future__ import annotations

import json
import math

import pytest

from avengine.m3.calibration import (
    BroadbandEDTCalibrationInputError,
    BroadbandEDTConvergenceError,
    BroadbandEDTEvaluationError,
    BroadbandEDTMonotonicityError,
    BroadbandEDTUnreachableError,
    calibrate_broadband_edt_seconds,
)


def _calibrate(default_evaluate, **overrides):
    arguments = {
        "target_seconds": 0.9,
        "tolerance_seconds": 1.0e-4,
        "absorption_min": 0.1,
        "absorption_max": 0.9,
        "max_iterations": 32,
        "evaluate": default_evaluate,
    }
    arguments.update(overrides)
    return calibrate_broadband_edt_seconds(**arguments)


def test_bounded_search_converges_and_result_is_json_safe() -> None:
    result = _calibrate(lambda absorption: 2.1 - 1.8 * absorption)

    assert result.status == "pass"
    assert result.metric == "edt_seconds"
    assert result.scope == "broadband"
    assert result.target_seconds == 0.9
    assert result.absolute_error_seconds <= result.tolerance_seconds
    assert result.error_seconds == pytest.approx(
        result.achieved_seconds - result.target_seconds
    )
    assert result.final_absorption == pytest.approx(2.0 / 3.0, abs=1.0e-4)
    assert 1 <= result.iterations <= 32
    assert [step.stage for step in result.trace[:2]] == [
        "absorption_min",
        "absorption_max",
    ]
    assert len(result.trace) == result.iterations + 2

    encoded = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["status"] == "pass"
    assert decoded["trace"][-1]["absorption"] == result.final_absorption


def test_repeat_measurements_are_averaged_and_spread_is_recorded() -> None:
    result = _calibrate(
        lambda absorption: [
            2.1 - 1.8 * absorption - 1.0e-5,
            2.1 - 1.8 * absorption,
            2.1 - 1.8 * absorption + 1.0e-5,
        ]
    )

    final = result.trace[-1]
    assert final.achieved_seconds == pytest.approx(2.1 - 1.8 * final.absorption)
    assert final.repeat_spread_seconds == pytest.approx(2.0e-5)
    assert final.measurements_seconds == pytest.approx(
        (
            final.achieved_seconds - 1.0e-5,
            final.achieved_seconds,
            final.achieved_seconds + 1.0e-5,
        )
    )


def test_endpoint_within_tolerance_returns_without_bisection() -> None:
    result = _calibrate(
        lambda absorption: 2.1 - 1.8 * absorption,
        target_seconds=1.92,
    )

    assert result.iterations == 0
    assert result.final_absorption == 0.1
    assert len(result.trace) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_seconds": 0.0}, "target_seconds must be positive"),
        ({"target_seconds": math.inf}, "target_seconds must be a finite"),
        ({"target_seconds": 10**4000}, "target_seconds must be a finite"),
        ({"tolerance_seconds": 0.0}, "tolerance_seconds must be positive"),
        ({"absorption_min": -0.1}, "absorption_min must be in"),
        ({"absorption_max": 1.1}, "absorption_max must be in"),
        (
            {"absorption_min": 0.5, "absorption_max": 0.5},
            "strictly less",
        ),
        ({"max_iterations": 0}, "positive integer"),
        ({"max_iterations": True}, "positive integer"),
        ({"evaluate": None}, "evaluate must be callable"),
    ],
)
def test_invalid_requests_fail_closed(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(BroadbandEDTCalibrationInputError, match=message):
        _calibrate(lambda absorption: 2.1 - 1.8 * absorption, **overrides)


@pytest.mark.parametrize(
    "bad_result",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
        10**4000,
        True,
        [],
        [0.8, math.nan],
        [1.0e308, 1.0e308],
        ["0.8"],
        "0.8",
    ],
)
def test_invalid_callback_measurements_fail_closed(bad_result: object) -> None:
    with pytest.raises(BroadbandEDTEvaluationError):
        _calibrate(lambda absorption: bad_result)


def test_callback_exception_is_wrapped_with_evaluation_context() -> None:
    def broken(_absorption: float) -> float:
        raise RuntimeError("native runner unavailable")

    with pytest.raises(BroadbandEDTEvaluationError, match="native runner unavailable"):
        _calibrate(broken)


def test_lazy_measurement_iterable_exception_is_wrapped() -> None:
    def measurements():
        yield 1.0
        raise RuntimeError("lazy native readback failed")

    with pytest.raises(
        BroadbandEDTEvaluationError, match="iterable failed during iteration"
    ):
        _calibrate(lambda _absorption: measurements())


def test_repeat_spread_cannot_exceed_requested_accuracy() -> None:
    with pytest.raises(BroadbandEDTEvaluationError, match="spread exceeds"):
        _calibrate(
            lambda absorption: [
                2.1 - 1.8 * absorption,
                2.1 - 1.8 * absorption + 2.0e-4,
            ]
        )


@pytest.mark.parametrize("target", [0.1, 2.5])
def test_endpoint_range_must_reach_target(target: float) -> None:
    with pytest.raises(BroadbandEDTUnreachableError, match="endpoint range"):
        _calibrate(
            lambda absorption: 2.1 - 1.8 * absorption,
            target_seconds=target,
        )


def test_endpoint_direction_violation_is_rejected() -> None:
    with pytest.raises(BroadbandEDTMonotonicityError, match="increases"):
        _calibrate(lambda absorption: 0.4 + absorption)


def test_interior_non_monotonic_measurement_is_rejected() -> None:
    def non_monotonic(absorption: float) -> float:
        if absorption == pytest.approx(0.5):
            return 2.0
        return 2.1 - 1.8 * absorption

    with pytest.raises(BroadbandEDTMonotonicityError, match="interior EDT"):
        _calibrate(non_monotonic)


def test_iteration_budget_exhaustion_is_explicit() -> None:
    with pytest.raises(BroadbandEDTConvergenceError, match="did not reach"):
        _calibrate(
            lambda absorption: 2.1 - 1.8 * absorption,
            target_seconds=0.83,
            tolerance_seconds=1.0e-12,
            max_iterations=1,
        )
