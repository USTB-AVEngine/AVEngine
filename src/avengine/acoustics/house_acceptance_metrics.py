"""CPU metrics for house admission; thresholds describe simulation quality, not STARSS."""
from __future__ import annotations

import math
import numpy as np
from scipy import signal

OCTAVES_HZ = (125, 250, 500, 1000, 2000, 4000)


def energy_db(value: float, reference: float = 1.0) -> float:
    return float(10 * np.log10(max(float(value), 1e-30) / max(float(reference), 1e-30)))


def decay_fit(wave: np.ndarray, sample_rate: int, lower_db: float):
    """Schroeder integration, least squares over -5..lower_db; RT60 extrapolation."""
    x = np.asarray(wave, dtype=float)
    e = np.cumsum(np.square(x)[::-1])[::-1]
    if not len(e) or e[0] <= 1e-24:
        return {"status": "no_energy", "rt60_s": None}, np.zeros(len(x))
    edc = 10 * np.log10(np.maximum(e / e[0], 1e-30))
    idx = np.flatnonzero((edc <= -5) & (edc >= lower_db))
    if len(idx) < max(32, int(.01 * sample_rate)):
        return {"status": "insufficient_decay_span", "rt60_s": None}, edc
    t = idx / sample_rate
    slope, intercept = np.polyfit(t, edc[idx], 1)
    fitted = slope * t + intercept
    den = float(np.sum((edc[idx] - edc[idx].mean()) ** 2))
    r2 = 1 - float(np.sum((edc[idx] - fitted) ** 2)) / max(den, 1e-30)
    valid = slope < 0 and r2 >= .95
    return {"status": "valid" if valid else "nonlinear_decay",
            "rt60_s": float(-60 / slope) if slope < 0 else None,
            "r_squared": float(r2), "fit_span_s": [float(t[0]), float(t[-1])],
            "decay_interval_db": [-5, lower_db],
            "slope_db_per_s": float(slope)}, edc


def rir_metrics(ir, sample_rate, distance_m, max_ir_seconds):
    a = np.asarray(ir, dtype=float)
    if a.ndim != 2 or a.shape[0] != 4 or not np.isfinite(a).all():
        raise ValueError("RIR must be finite [W,Y,Z,X] x time")
    w = a[0]
    if not len(w) or np.max(np.abs(w)) <= 1e-12:
        return {"status": "no_energy", "octaves": [], "rt60_mid_s": None}, {}
    onset = int(np.flatnonzero(np.abs(w) >= .1 * np.max(np.abs(w)))[0])
    expected = distance_m / 343
    # The first measured arrival owns the direct gate. A stronger reflection
    # inside the old +/-2/3 ms geometric search must not replace that arrival.
    # Keep the existing 10%-of-peak onset criterion; choose the
    # first qualifying local peak independently of the expected travel time.
    # This also leaves a genuinely late arrival visible to the delay check.
    arrival_peaks, _ = signal.find_peaks(np.abs(w), height=.1 * np.max(np.abs(w)))
    arrival_peaks = arrival_peaks[arrival_peaks >= onset]
    direct_peak = int(arrival_peaks[0]) if len(arrival_peaks) else onset
    # Share the already fixed 0.5 ms first-arrival FOA window. The former
    # 1.25 ms energy window also integrated early reflections whose Monte
    # Carlo energy could change the distance verdict with an identical pulse.
    # This is a direct-component measurement, not total early room energy.
    direct_start = onset
    direct_end = min(len(w), direct_start + max(1, int(.0005 * sample_rate)))
    direct_energy = float(np.square(w[direct_start:direct_end]).sum())
    total = float(np.square(w).sum())
    late = float(np.square(w[direct_end:]).sum())
    first = min(onset, direct_start)
    tail = w[first:]
    block = max(1, int(.02 * sample_rate))
    bin_e = np.array([np.square(tail[k:k + block]).mean()
                      for k in range(0, len(tail), block)])
    terminal_db = energy_db(bin_e[-1], float(bin_e.max()))
    tail_energy_db = energy_db(float(np.square(tail[-max(1, int(.05 * sample_rate)):]).sum()),
                               float(np.square(tail).sum()))
    at_limit = len(w) >= max_ir_seconds * sample_rate - 1
    # A backwards integral necessarily reaches -infinity at an artificial cut.
    # Terminal forward energy is therefore assessed independently.
    truncated = bool(terminal_db > -40 and (at_limit or tail_energy_db > -35))
    early_end = min(len(w), direct_peak + int(.05 * sample_rate))
    early = float(np.square(w[first:early_end]).sum())
    c50 = energy_db(early, float(np.square(w[early_end:]).sum()))
    reflection_signal = np.abs(w[direct_end:early_end])
    peaks, _ = signal.find_peaks(reflection_signal,
                                height=max(np.max(np.abs(w)) * .02, 1e-15),
                                distance=max(1, int(.001 * sample_rate)))
    reflections = sorted(({"delay_after_direct_ms": float((direct_end + int(k) - direct_peak) / sample_rate * 1000),
                            "amplitude_relative_direct_db": float(20 * np.log10(max(abs(w[direct_end + k]), 1e-30) /
                                                                               max(abs(w[direct_peak]), 1e-30)))}
                           for k in peaks), key=lambda d: d["amplitude_relative_direct_db"], reverse=True)[:8]
    bands, curves = [], {}
    for fc in OCTAVES_HZ:
        sos = signal.butter(4, [fc / math.sqrt(2), fc * math.sqrt(2)],
                            btype="bandpass", fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, tail)
        t20, edc = decay_fit(filtered, sample_rate, -25)
        t30, _ = decay_fit(filtered, sample_rate, -35)
        be = np.array([np.square(filtered[k:k + block]).mean()
                       for k in range(0, len(filtered), block)])
        end_db = energy_db(be[-1], float(be.max())) if len(be) else 0
        for fit in (t20, t30):
            fit["terminal_energy_db"] = end_db
            if fit["status"] == "valid" and (truncated or end_db > -40):
                fit["status"] = "tail_or_noise_floor_unresolved"
            if fit["status"] == "valid" and fit["fit_span_s"][1] >= .9 * len(filtered) / sample_rate:
                fit["status"] = "fit_touches_tail"
        bands.append({"center_hz": fc, "t20": t20, "t30": t30,
                      "t20_t30_relative_difference": (abs(t20["rt60_s"] - t30["rt60_s"]) / t20["rt60_s"]
                                                    if t20["rt60_s"] and t30["rt60_s"] else None)})
        curves[fc] = edc
    mids = [b["t30"]["rt60_s"] if b["t30"]["status"] == "valid" else b["t20"]["rt60_s"]
            for b in bands if b["center_hz"] in (500, 1000, 2000)
            and (b["t20"]["status"] == "valid" or b["t30"]["status"] == "valid")]
    return {"status": "valid" if len(mids) >= 2 else "insufficient_valid_midbands",
            "sample_count": len(w), "duration_s": len(w) / sample_rate,
            "onset_s": onset / sample_rate, "geometric_direct_time_s": expected,
            "direct_peak_s": direct_peak / sample_rate,
            "direct_delay_error_ms": (direct_peak / sample_rate - expected) * 1000,
            "direct_gate_samples": [direct_start, direct_end],
            "direct_energy": direct_energy, "total_energy": total,
            "drr_db": energy_db(direct_energy, late), "c50_db": c50,
            "early_reflections": reflections, "terminal_energy_db": terminal_db,
            "last_50ms_energy_fraction_db": tail_energy_db, "at_ir_limit": at_limit,
            "tail_truncated": truncated, "octaves": bands,
            "rt60_mid_s": float(np.median(mids)) if len(mids) >= 2 else None}, curves


def foa_direction(ir, listener, source, camera_forward, sample_rate=16000):
    """Reuse v6 raw ACN [W,Y,Z,X] intensity and [forward;-right;up] rotation."""
    a = np.asarray(ir, dtype=float)
    v = np.asarray(source, dtype=float) - np.asarray(listener, dtype=float)
    v /= np.linalg.norm(v)
    forward = np.asarray(camera_forward, dtype=float)
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    matrix = np.stack((forward, -right, up))
    onset = int(np.argmax(np.abs(a[0]) > .1 * np.max(np.abs(a[0]))))
    windows = []
    for ms in (.25, .5, 1, 2, 4, 12, 400):
        end = min(a.shape[1], onset + max(1, int(ms * sample_rate / 1000)))
        raw = np.mean(a[0, onset:end] * a[[3, 1, 2], onset:end], axis=1)
        norm = np.linalg.norm(raw)
        if norm <= 1e-24:
            windows.append({"window_ms": ms, "angle_deg": None}); continue
        unit = raw / norm
        cam = matrix @ unit
        geometric = matrix @ v
        windows.append({"window_ms": ms,
                        "angle_deg": float(np.degrees(np.arccos(np.clip(unit @ v, -1, 1)))),
                        "foa_world_direction": unit.tolist(),
                        "foa_azimuth_left_positive_deg": float(np.degrees(np.arctan2(cam[1], cam[0]))),
                        "geometry_azimuth_left_positive_deg": float(np.degrees(np.arctan2(geometric[1], geometric[0])))})
    return {"camera_forward": forward.tolist(), "camera_right": right.tolist(),
            "camera_up": up.tolist(), "world_to_camera": matrix.tolist(),
            "windows": windows, "v6_method": "raw W*[X,Y,Z], [forward;-right;up] @ world",
            "fixed_check_window_ms": .5,
            "check": windows[1], "geometry_world_direction": v.tolist()}


def distance_attenuation(distances, direct_energies):
    d, e = np.asarray(distances, dtype=float), np.asarray(direct_energies, dtype=float)
    valid = (d > 0) & (e > 0) & np.isfinite(d) & np.isfinite(e)
    d, e = d[valid], e[valid]
    if len(d) < 3 or d.max() / d.min() < 1.8:
        return {"status": "insufficient_distance_span", "pressure_exponent": None}
    slope, intercept = np.polyfit(np.log10(d), 10 * np.log10(e), 1)
    predicted = slope * np.log10(d) + intercept
    r2 = 1 - float(np.square(10 * np.log10(e) - predicted).sum()) / max(float(np.square(10 * np.log10(e) - np.mean(10 * np.log10(e))).sum()), 1e-30)
    return {"status": "valid", "pressure_exponent": float(-slope / 20),
            "energy_slope_db_per_distance_decade": float(slope), "r_squared": float(r2),
            "distance_range_m": [float(d.min()), float(d.max())]}
