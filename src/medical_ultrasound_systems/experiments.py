"""Experiment orchestration for synthetic robustness benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np

from .beamforming import delay_and_sum_plane_wave
from .coherence import conventional_coherence_image
from .geometry import LinearArrayGeometry
from .perturbations import perturb_rf_channel_data
from .phantom import PointScatterer, PointScattererPhantom
from .qbeamforming import quaternionic_alignment_image, quaternionic_intensity_image
from .simulation import simulate_pulse_echo_rf


@dataclass
class PeakResult:
    """Peak localization result for one reconstruction method."""

    method: str
    peak_x_m: float
    peak_z_m: float
    target_x_m: float
    target_z_m: float
    localization_error_m: float
    peak_value: float


@dataclass
class ExperimentResult:
    """Experiment-level benchmark output with peaks and runtimes."""

    name: str
    parameters: dict
    peaks: list[PeakResult]
    runtime_s: dict
    metadata: dict | None = None


def _derive_child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max, endpoint=True))


def _default_x_grid_m() -> np.ndarray:
    return np.linspace(-0.01, 0.01, 56)


def _default_z_grid_m() -> np.ndarray:
    return np.linspace(0.015, 0.055, 72)


def _build_scatterers(
    target_x_m: float,
    target_z_m: float,
    perturbation: dict,
) -> list[PointScatterer]:
    scatterers = [PointScatterer(x_m=target_x_m, z_m=target_z_m, amplitude=1.0)]
    for item in perturbation.get("additional_scatterers", []):
        if isinstance(item, dict):
            scatterers.append(
                PointScatterer(
                    x_m=float(item["x_m"]),
                    z_m=float(item["z_m"]),
                    amplitude=float(item.get("amplitude", 1.0)),
                )
            )
        else:
            x_m, z_m, amplitude = item
            scatterers.append(
                PointScatterer(
                    x_m=float(x_m),
                    z_m=float(z_m),
                    amplitude=float(amplitude),
                )
            )
    return scatterers


def find_peak(image: np.ndarray, x_grid_m: np.ndarray, z_grid_m: np.ndarray) -> tuple[float, float, float]:
    """Return peak x, peak z, and peak value from a 2D image."""
    image_arr = np.asarray(image, dtype=float)
    x_grid_arr = np.asarray(x_grid_m, dtype=float)
    z_grid_arr = np.asarray(z_grid_m, dtype=float)
    if image_arr.ndim != 2:
        raise ValueError("image must be 2D.")
    if x_grid_arr.ndim != 1 or z_grid_arr.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D.")
    if image_arr.shape != (z_grid_arr.size, x_grid_arr.size):
        raise ValueError("image shape must match (len(z_grid_m), len(x_grid_m)).")
    if image_arr.size == 0:
        raise ValueError("image must be non-empty.")

    peak_idx = np.unravel_index(np.argmax(np.abs(image_arr)), image_arr.shape)
    z_idx, x_idx = int(peak_idx[0]), int(peak_idx[1])
    peak_value = float(np.abs(image_arr[z_idx, x_idx]))
    return float(x_grid_arr[x_idx]), float(z_grid_arr[z_idx]), peak_value


def localization_error_m(
    peak_x_m: float,
    peak_z_m: float,
    target_x_m: float,
    target_z_m: float,
) -> float:
    """Return Euclidean peak localization error in meters."""
    return float(np.hypot(float(peak_x_m) - float(target_x_m), float(peak_z_m) - float(target_z_m)))


def single_point_comparison(
    target_x_m: float = 0.002,
    target_z_m: float = 0.032,
    n_elements: int = 32,
    pitch_m: float = 0.0003,
    x_grid_m: np.ndarray | None = None,
    z_grid_m: np.ndarray | None = None,
    perturbation: dict | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """Run one synthetic conventional-vs-quaternionic comparison experiment."""
    x_grid_arr = np.asarray(_default_x_grid_m() if x_grid_m is None else x_grid_m, dtype=float)
    z_grid_arr = np.asarray(_default_z_grid_m() if z_grid_m is None else z_grid_m, dtype=float)
    perturbation = {} if perturbation is None else dict(perturbation)

    rng = np.random.default_rng(seed)
    runtime_s: dict[str, float] = {}

    geometry = LinearArrayGeometry(n_elements=int(n_elements), pitch_m=float(pitch_m), center_frequency_hz=5e6)
    scatterers = _build_scatterers(
        target_x_m=float(target_x_m),
        target_z_m=float(target_z_m),
        perturbation=perturbation,
    )
    phantom = PointScattererPhantom(
        scatterers=scatterers,
        metadata={"generator": "single_point_comparison", "seed": seed},
    )

    t0 = time.perf_counter()
    rf = simulate_pulse_echo_rf(
        geometry=geometry,
        phantom=phantom,
        sample_rate_hz=40e6,
        center_frequency_hz=5e6,
        duration_s=80e-6,
    )
    runtime_s["simulate_pulse_echo_rf"] = float(time.perf_counter() - t0)

    perturbation_params = {
        "snr_db": perturbation.get("snr_db"),
        "dropout_fraction": float(perturbation.get("dropout_fraction", 0.0)),
        "gain_std": float(perturbation.get("gain_std", 0.0)),
        "max_jitter_samples": int(perturbation.get("max_jitter_samples", 0)),
    }

    t0 = time.perf_counter()
    rf_perturbed = perturb_rf_channel_data(
        rf=rf,
        snr_db=perturbation_params["snr_db"],
        dropout_fraction=perturbation_params["dropout_fraction"],
        gain_std=perturbation_params["gain_std"],
        max_jitter_samples=perturbation_params["max_jitter_samples"],
        seed=_derive_child_seed(rng),
    )
    runtime_s["perturb_rf_channel_data"] = float(time.perf_counter() - t0)

    method_funcs = {
        "delay_and_sum_plane_wave": delay_and_sum_plane_wave,
        "conventional_coherence_image": conventional_coherence_image,
        "quaternionic_alignment_image": quaternionic_alignment_image,
        "quaternionic_intensity_image": quaternionic_intensity_image,
    }

    peaks: list[PeakResult] = []
    for method_name, method_fn in method_funcs.items():
        t0 = time.perf_counter()
        image = method_fn(rf_perturbed, x_grid_m=x_grid_arr, z_grid_m=z_grid_arr)
        runtime_s[method_name] = float(time.perf_counter() - t0)
        peak_x_m, peak_z_m, peak_value = find_peak(
            image=image,
            x_grid_m=x_grid_arr,
            z_grid_m=z_grid_arr,
        )
        peaks.append(
            PeakResult(
                method=method_name,
                peak_x_m=peak_x_m,
                peak_z_m=peak_z_m,
                target_x_m=float(target_x_m),
                target_z_m=float(target_z_m),
                localization_error_m=localization_error_m(
                    peak_x_m=peak_x_m,
                    peak_z_m=peak_z_m,
                    target_x_m=float(target_x_m),
                    target_z_m=float(target_z_m),
                ),
                peak_value=peak_value,
            )
        )

    parameters = {
        "target_x_m": float(target_x_m),
        "target_z_m": float(target_z_m),
        "n_elements": int(n_elements),
        "pitch_m": float(pitch_m),
        "x_grid_size": int(x_grid_arr.size),
        "z_grid_size": int(z_grid_arr.size),
        **perturbation_params,
    }
    if "additional_scatterers" in perturbation:
        parameters["additional_scatterers"] = perturbation["additional_scatterers"]

    return ExperimentResult(
        name="single_point_comparison",
        parameters=parameters,
        peaks=peaks,
        runtime_s=runtime_s,
        metadata={
            "seed": seed,
            "notes": "Synthetic benchmark candidate; research metric only and requires validation.",
        },
    )


def noise_sweep(
    snr_values_db: list[float],
    n_trials: int = 3,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """Run repeated synthetic trials over additive-noise SNR settings."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    rng = np.random.default_rng(seed)
    results: list[ExperimentResult] = []
    for snr_db in snr_values_db:
        for trial_idx in range(n_trials):
            result = single_point_comparison(
                perturbation={"snr_db": float(snr_db)},
                seed=_derive_child_seed(rng),
            )
            result.name = "noise_sweep"
            result.parameters["snr_db"] = float(snr_db)
            result.parameters["trial_index"] = int(trial_idx)
            results.append(result)
    return results


def dropout_sweep(
    dropout_values: list[float],
    n_trials: int = 3,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """Run repeated synthetic trials over channel-dropout settings."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    rng = np.random.default_rng(seed)
    results: list[ExperimentResult] = []
    for dropout_fraction in dropout_values:
        for trial_idx in range(n_trials):
            result = single_point_comparison(
                perturbation={"dropout_fraction": float(dropout_fraction)},
                seed=_derive_child_seed(rng),
            )
            result.name = "dropout_sweep"
            result.parameters["dropout_fraction"] = float(dropout_fraction)
            result.parameters["trial_index"] = int(trial_idx)
            results.append(result)
    return results


def gain_jitter_sweep(
    gain_std_values: list[float],
    max_jitter_samples_values: list[int],
    n_trials: int = 3,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """Run repeated synthetic trials over gain-variation and timing-jitter settings."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    rng = np.random.default_rng(seed)
    results: list[ExperimentResult] = []
    for gain_std in gain_std_values:
        for max_jitter_samples in max_jitter_samples_values:
            for trial_idx in range(n_trials):
                result = single_point_comparison(
                    perturbation={
                        "gain_std": float(gain_std),
                        "max_jitter_samples": int(max_jitter_samples),
                    },
                    seed=_derive_child_seed(rng),
                )
                result.name = "gain_jitter_sweep"
                result.parameters["gain_std"] = float(gain_std)
                result.parameters["max_jitter_samples"] = int(max_jitter_samples)
                result.parameters["trial_index"] = int(trial_idx)
                results.append(result)
    return results


def experiment_result_to_dict(result: ExperimentResult) -> dict:
    """Convert an experiment result to a JSON-serializable dictionary."""
    return {
        "name": result.name,
        "parameters": result.parameters,
        "runtime_s": result.runtime_s,
        "metadata": result.metadata,
        "peaks": [
            {
                "method": peak.method,
                "peak_x_m": peak.peak_x_m,
                "peak_z_m": peak.peak_z_m,
                "target_x_m": peak.target_x_m,
                "target_z_m": peak.target_z_m,
                "localization_error_m": peak.localization_error_m,
                "peak_value": peak.peak_value,
            }
            for peak in result.peaks
        ],
    }


def _flatten_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def experiment_results_to_rows(results: list[ExperimentResult]) -> list[dict]:
    """Convert experiment results to flat CSV-friendly row records."""
    rows: list[dict] = []
    for result in results:
        flat_params = {f"param_{key}": _flatten_value(value) for key, value in result.parameters.items()}
        for peak in result.peaks:
            row = {
                "experiment_name": result.name,
                "method": peak.method,
                "target_x_m": peak.target_x_m,
                "target_z_m": peak.target_z_m,
                "peak_x_m": peak.peak_x_m,
                "peak_z_m": peak.peak_z_m,
                "localization_error_m": peak.localization_error_m,
                "peak_value": peak.peak_value,
                "runtime_s": result.runtime_s.get(peak.method),
            }
            row.update(flat_params)
            rows.append(row)
    return rows
