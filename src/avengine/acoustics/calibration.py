"""Fail-closed scalar calibration for a broadband EDT target.

This module deliberately calibrates the measured ``edt_seconds`` metric.  It
does not expose, estimate, or claim a material-level RT60.  The caller owns the
acoustic simulation and supplies an evaluation callback that maps one uniform
absorption scalar to either one broadband EDT measurement or a sequence of
repeat measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterable


EDTEvaluation = Callable[[float], Real | Iterable[Real]]


class BroadbandEDTCalibrationError(ValueError):
    """Base class for a calibration that cannot make a valid claim."""


class BroadbandEDTCalibrationInputError(BroadbandEDTCalibrationError):
    """The requested calibration bounds or tolerances are invalid."""


class BroadbandEDTEvaluationError(BroadbandEDTCalibrationError):
    """The evaluation callback failed or returned unusable measurements."""


class BroadbandEDTUnreachableError(BroadbandEDTCalibrationError):
    """The target is outside the EDT range established by the two bounds."""


class BroadbandEDTMonotonicityError(BroadbandEDTCalibrationError):
    """Observed EDT violates higher-absorption-means-shorter-EDT ordering."""


class BroadbandEDTConvergenceError(BroadbandEDTCalibrationError):
    """The bounded search exhausted its iteration budget without a pass."""


@dataclass(frozen=True)
class BroadbandEDTCalibrationStep:
    """One callback evaluation in a broadband EDT calibration trace."""

    evaluation_index: int
    stage: str
    iteration: int
    absorption: float
    measurements_seconds: tuple[float, ...]
    achieved_seconds: float
    error_seconds: float
    absolute_error_seconds: float
    repeat_spread_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe trace record."""

        return {
            "evaluation_index": self.evaluation_index,
            "stage": self.stage,
            "iteration": self.iteration,
            "absorption": self.absorption,
            "measurements_seconds": list(self.measurements_seconds),
            "achieved_seconds": self.achieved_seconds,
            "error_seconds": self.error_seconds,
            "absolute_error_seconds": self.absolute_error_seconds,
            "repeat_spread_seconds": self.repeat_spread_seconds,
        }


@dataclass(frozen=True)
class BroadbandEDTCalibrationResult:
    """A successful, tolerance-bound broadband EDT calibration."""

    target_seconds: float
    tolerance_seconds: float
    achieved_seconds: float
    error_seconds: float
    absolute_error_seconds: float
    final_absorption: float
    iterations: int
    trace: tuple[BroadbandEDTCalibrationStep, ...]
    metric: str = "edt_seconds"
    scope: str = "broadband"
    status: str = "pass"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe calibration result and its complete trace."""

        return {
            "metric": self.metric,
            "scope": self.scope,
            "status": self.status,
            "target_seconds": self.target_seconds,
            "tolerance_seconds": self.tolerance_seconds,
            "achieved_seconds": self.achieved_seconds,
            "error_seconds": self.error_seconds,
            "absolute_error_seconds": self.absolute_error_seconds,
            "final_absorption": self.final_absorption,
            "iterations": self.iterations,
            "trace": [step.to_dict() for step in self.trace],
        }


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BroadbandEDTCalibrationInputError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise BroadbandEDTCalibrationInputError(
            f"{name} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise BroadbandEDTCalibrationInputError(f"{name} must be a finite number")
    return result


def _monotonic_epsilon(*values: float) -> float:
    """Return only a round-off allowance, not an acoustic error tolerance."""

    return max(1.0e-12, max(abs(value) for value in values) * 1.0e-12)


def calibrate_broadband_edt_seconds(
    *,
    target_seconds: float,
    tolerance_seconds: float,
    absorption_min: float,
    absorption_max: float,
    max_iterations: int,
    evaluate: EDTEvaluation,
) -> BroadbandEDTCalibrationResult:
    """Calibrate one uniform absorption scalar to a broadband EDT target.

    The search assumes the physical direction used by the M3 material canary:
    increasing absorption must not increase broadband EDT.  Both absorption
    endpoints are evaluated before bisection, which establishes reachability
    from actual callback results instead of an analytic room formula.

    ``tolerance_seconds`` serves two fail-closed purposes: the final mean EDT
    must be within it of the target, and the max-minus-min spread of callback
    repeats at every evaluated absorption must not exceed it.  A repeat spread
    larger than the requested calibration accuracy cannot support a pass.

    Args:
        target_seconds: Positive finite broadband EDT target.
        tolerance_seconds: Positive finite absolute target/repeat tolerance.
        absorption_min: Inclusive lower scalar bound in ``[0, 1]``.
        absorption_max: Inclusive upper scalar bound in ``[0, 1]``.
        max_iterations: Positive maximum number of bisection evaluations.
        evaluate: Callback returning one positive finite EDT measurement or a
            non-empty iterable of positive finite repeat measurements.

    Raises:
        BroadbandEDTCalibrationInputError: for an invalid request.
        BroadbandEDTEvaluationError: for callback failure, invalid values, or
            repeat spread exceeding ``tolerance_seconds``.
        BroadbandEDTUnreachableError: when endpoint measurements cannot reach
            the requested target within tolerance.
        BroadbandEDTMonotonicityError: when evaluated values violate the
            required absorption/EDT direction.
        BroadbandEDTConvergenceError: when bisection cannot pass within the
            requested iteration budget.
    """

    target = _finite_number(target_seconds, name="target_seconds")
    tolerance = _finite_number(tolerance_seconds, name="tolerance_seconds")
    lower_absorption = _finite_number(absorption_min, name="absorption_min")
    upper_absorption = _finite_number(absorption_max, name="absorption_max")
    if target <= 0.0:
        raise BroadbandEDTCalibrationInputError("target_seconds must be positive")
    if tolerance <= 0.0:
        raise BroadbandEDTCalibrationInputError("tolerance_seconds must be positive")
    if not 0.0 <= lower_absorption <= 1.0:
        raise BroadbandEDTCalibrationInputError("absorption_min must be in [0, 1]")
    if not 0.0 <= upper_absorption <= 1.0:
        raise BroadbandEDTCalibrationInputError("absorption_max must be in [0, 1]")
    if lower_absorption >= upper_absorption:
        raise BroadbandEDTCalibrationInputError(
            "absorption_min must be strictly less than absorption_max"
        )
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise BroadbandEDTCalibrationInputError(
            "max_iterations must be a positive integer"
        )
    if not callable(evaluate):
        raise BroadbandEDTCalibrationInputError("evaluate must be callable")

    trace: list[BroadbandEDTCalibrationStep] = []

    def evaluate_once(
        absorption: float, *, stage: str, iteration: int
    ) -> BroadbandEDTCalibrationStep:
        try:
            raw_measurements = evaluate(absorption)
        except Exception as exc:
            raise BroadbandEDTEvaluationError(
                f"evaluation callback failed at absorption={absorption:.17g}: {exc}"
            ) from exc

        if isinstance(raw_measurements, bool):
            raise BroadbandEDTEvaluationError(
                "evaluation callback must return positive finite EDT seconds"
            )
        if isinstance(raw_measurements, Real):
            candidates: Iterable[object] = (raw_measurements,)
        elif isinstance(raw_measurements, (str, bytes)):
            raise BroadbandEDTEvaluationError(
                "evaluation callback must return a number or measurements iterable"
            )
        else:
            try:
                candidates = iter(raw_measurements)
            except TypeError as exc:
                raise BroadbandEDTEvaluationError(
                    "evaluation callback must return a number or measurements iterable"
                ) from exc

        measurements: list[float] = []
        try:
            for index, candidate in enumerate(candidates):
                if isinstance(candidate, bool) or not isinstance(candidate, Real):
                    raise BroadbandEDTEvaluationError(
                        f"measurement[{index}] must be positive finite EDT seconds"
                    )
                try:
                    measurement = float(candidate)
                except (OverflowError, ValueError) as exc:
                    raise BroadbandEDTEvaluationError(
                        f"measurement[{index}] must be positive finite EDT seconds"
                    ) from exc
                if not math.isfinite(measurement) or measurement <= 0.0:
                    raise BroadbandEDTEvaluationError(
                        f"measurement[{index}] must be positive finite EDT seconds"
                    )
                measurements.append(measurement)
        except BroadbandEDTEvaluationError:
            raise
        except Exception as exc:
            raise BroadbandEDTEvaluationError(
                "evaluation measurements iterable failed during iteration"
            ) from exc
        if not measurements:
            raise BroadbandEDTEvaluationError(
                "evaluation callback returned no EDT measurements"
            )

        spread = max(measurements) - min(measurements)
        if spread > tolerance:
            raise BroadbandEDTEvaluationError(
                "repeat EDT spread exceeds tolerance_seconds at "
                f"absorption={absorption:.17g}: {spread:.17g} > {tolerance:.17g}"
            )
        try:
            achieved = math.fsum(measurements) / len(measurements)
        except OverflowError as exc:
            raise BroadbandEDTEvaluationError(
                "mean EDT measurement is not finite"
            ) from exc
        if not math.isfinite(achieved) or achieved <= 0.0:
            raise BroadbandEDTEvaluationError(
                "mean EDT measurement must be positive and finite"
            )
        error = achieved - target
        step = BroadbandEDTCalibrationStep(
            evaluation_index=len(trace) + 1,
            stage=stage,
            iteration=iteration,
            absorption=absorption,
            measurements_seconds=tuple(measurements),
            achieved_seconds=achieved,
            error_seconds=error,
            absolute_error_seconds=abs(error),
            repeat_spread_seconds=spread,
        )
        trace.append(step)
        return step

    minimum_step = evaluate_once(
        lower_absorption, stage="absorption_min", iteration=0
    )
    maximum_step = evaluate_once(
        upper_absorption, stage="absorption_max", iteration=0
    )
    endpoint_epsilon = _monotonic_epsilon(
        minimum_step.achieved_seconds, maximum_step.achieved_seconds
    )
    if (
        minimum_step.achieved_seconds + endpoint_epsilon
        < maximum_step.achieved_seconds
    ):
        raise BroadbandEDTMonotonicityError(
            "endpoint EDT increases with absorption: "
            f"{minimum_step.achieved_seconds:.17g} at absorption_min, "
            f"{maximum_step.achieved_seconds:.17g} at absorption_max"
        )

    highest_reachable = minimum_step.achieved_seconds
    lowest_reachable = maximum_step.achieved_seconds
    target_above_range = (
        target > highest_reachable and target - highest_reachable > tolerance
    )
    target_below_range = (
        target < lowest_reachable and lowest_reachable - target > tolerance
    )
    if target_above_range or target_below_range:
        raise BroadbandEDTUnreachableError(
            f"target_seconds={target:.17g} is outside the measured endpoint range "
            f"[{lowest_reachable:.17g}, {highest_reachable:.17g}] within "
            f"tolerance_seconds={tolerance:.17g}"
        )

    best_endpoint = min(
        (minimum_step, maximum_step), key=lambda step: step.absolute_error_seconds
    )
    if best_endpoint.absolute_error_seconds <= tolerance:
        return _successful_result(
            target=target,
            tolerance=tolerance,
            final_step=best_endpoint,
            iterations=0,
            trace=trace,
        )

    lower_step = minimum_step
    upper_step = maximum_step
    for iteration in range(1, max_iterations + 1):
        midpoint = (lower_step.absorption + upper_step.absorption) / 2.0
        if midpoint <= lower_step.absorption or midpoint >= upper_step.absorption:
            raise BroadbandEDTConvergenceError(
                "absorption interval can no longer be subdivided before reaching "
                "tolerance_seconds"
            )
        midpoint_step = evaluate_once(
            midpoint, stage="bisection", iteration=iteration
        )
        monotonic_epsilon = _monotonic_epsilon(
            lower_step.achieved_seconds,
            midpoint_step.achieved_seconds,
            upper_step.achieved_seconds,
        )
        if (
            midpoint_step.achieved_seconds
            > lower_step.achieved_seconds + monotonic_epsilon
            or midpoint_step.achieved_seconds
            < upper_step.achieved_seconds - monotonic_epsilon
        ):
            raise BroadbandEDTMonotonicityError(
                "interior EDT violates the current monotonic bracket at "
                f"absorption={midpoint:.17g}"
            )

        if midpoint_step.absolute_error_seconds <= tolerance:
            return _successful_result(
                target=target,
                tolerance=tolerance,
                final_step=midpoint_step,
                iterations=iteration,
                trace=trace,
            )

        if midpoint_step.achieved_seconds > target:
            # EDT is too long, so increase absorption.
            lower_step = midpoint_step
        else:
            # EDT is too short, so decrease absorption.
            upper_step = midpoint_step

    best_step = min(trace, key=lambda step: step.absolute_error_seconds)
    raise BroadbandEDTConvergenceError(
        f"calibration did not reach tolerance_seconds={tolerance:.17g} after "
        f"{max_iterations} iterations; best absolute error was "
        f"{best_step.absolute_error_seconds:.17g} at "
        f"absorption={best_step.absorption:.17g}"
    )


def _successful_result(
    *,
    target: float,
    tolerance: float,
    final_step: BroadbandEDTCalibrationStep,
    iterations: int,
    trace: list[BroadbandEDTCalibrationStep],
) -> BroadbandEDTCalibrationResult:
    result = BroadbandEDTCalibrationResult(
        target_seconds=target,
        tolerance_seconds=tolerance,
        achieved_seconds=final_step.achieved_seconds,
        error_seconds=final_step.error_seconds,
        absolute_error_seconds=final_step.absolute_error_seconds,
        final_absorption=final_step.absorption,
        iterations=iterations,
        trace=tuple(trace),
    )
    # Keep the return boundary fail-closed if this dataclass is changed later.
    if result.absolute_error_seconds > result.tolerance_seconds:
        raise BroadbandEDTConvergenceError(
            "internal calibration result exceeds tolerance_seconds"
        )
    return result


__all__ = [
    "BroadbandEDTCalibrationError",
    "BroadbandEDTCalibrationInputError",
    "BroadbandEDTCalibrationResult",
    "BroadbandEDTCalibrationStep",
    "BroadbandEDTConvergenceError",
    "BroadbandEDTEvaluationError",
    "BroadbandEDTMonotonicityError",
    "BroadbandEDTUnreachableError",
    "EDTEvaluation",
    "calibrate_broadband_edt_seconds",
]
