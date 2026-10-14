"""Reproducible, wholly synthetic Huanghua PV/weather data; no model training.

Run with Python 3.10+, numpy and pandas. --validate-only checks existing CSVs,
including deterministic regeneration and compatibility with pv_forecast.
All meteorological parameters are simulation assumptions, not observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
NUMERIC = [
    "pv_power_kw", "SWGDN", "DHI", "DNI", "temperature_c", "rain",
    "cloud_cover", "solar_elevation_deg",
]
COLUMNS = [
    "timestamp", "pv_power_kw", "SWGDN", "DHI", "DNI", "temperature_c",
    "rain", "cloud_cover", "weather_type", "solar_elevation_deg",
    "station_id", "is_simulated",
]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    require(config["schema_version"] == "huanghua-pv-simulation-1", "Unsupported config version")
    require(config["is_simulated"] is True, "Dataset must be explicitly simulated")
    require(config["interval_minutes"] == 15, "This generator requires 15-minute intervals")
    require(config["timezone"] == "Asia/Shanghai", "This regional generator uses Asia/Shanghai")
    require(config["station"]["capacity_kw"] > 0, "Capacity must be positive")
    w = config["weather"]
    probabilities = np.asarray(w["monthly_state_probabilities"], dtype=float)
    require(probabilities.shape == (12, 4), "Expected 12 monthly probability vectors")
    require(np.all(probabilities >= 0) and np.allclose(probabilities.sum(axis=1), 1), "Invalid weather probabilities")
    require(w["states"] == ["clear", "partly_cloudy", "overcast", "rain"], "Unexpected state ordering")
    for key in ["regime_persistence", "cloud_relaxation", "daily_temperature_anomaly_ar",
                "temperature_noise_ar", "rain_wet_continuation_probability", "rain_wet_start_probability"]:
        require(0 <= w[key] < 1, f"Invalid probability/coefficient: {key}")
    require(0 <= config["pv"]["system_loss_fraction"] < 1, "Invalid loss fraction")
    require(0 <= config["pv"]["relative_noise_ar"] < 1, "Invalid power noise coefficient")
    return config


def year_index(config: dict) -> pd.DatetimeIndex:
    year = config["year"]
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min",
                         inclusive="left", tz=config["timezone"])


def solar_geometry(index: pd.DatetimeIndex, latitude: float, longitude: float):
    """NOAA fractional-year approximation; ENU vector and geometric elevation.

    No terrain horizon or refraction correction. Azimuth is clockwise from
    north. Equation-of-time coefficients are part of this versioned algorithm.
    """
    hours = np.asarray(index.hour + index.minute / 60, dtype=float)
    days = np.where(index.is_leap_year, 366.0, 365.0)
    gamma = 2 * np.pi / days * (np.asarray(index.dayofyear) - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                      - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    offset_hours = index[0].utcoffset().total_seconds() / 3600
    true_minutes = (hours * 60 + eqtime + 4 * longitude - 60 * offset_hours) % 1440
    hour_angle = np.radians(true_minutes / 4 - 180)
    lat = np.radians(latitude)
    east = -np.cos(decl) * np.sin(hour_angle)
    north = np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(hour_angle)
    up = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    elevation = np.degrees(np.arcsin(np.clip(up, -1, 1)))
    return east, north, up, elevation


def ar_noise(rng, size: int, coefficient: float, stationary_std: float) -> np.ndarray:
    values = np.empty(size)
    value = rng.normal(0, stationary_std)
    innovation = stationary_std * np.sqrt(1 - coefficient ** 2)
    for i in range(size):
        value = coefficient * value + rng.normal(0, innovation)
        values[i] = value
    return values


def seasonal_monthly(index, values):
    """Interpolate monthly midpoint assumptions without calendar jumps."""
    start_year = index[0].year - 1
    end_year = index[-1].year + 1
    anchors = pd.DatetimeIndex([
        pd.Timestamp(year=y, month=m, day=15, tz=index.tz)
        for y in range(start_year, end_year + 1) for m in range(1, 13)
    ])
    targets = np.tile(np.asarray(values, dtype=float), end_year - start_year + 1)
    return np.interp(index.asi8.astype(float), anchors.asi8.astype(float), targets)


def simulate(config: dict) -> pd.DataFrame:
    expected = year_index(config)
    index = pd.date_range(expected[0] - pd.Timedelta(days=config["warmup_days"]),
                          expected[-1], freq="15min", tz=config["timezone"])
    count = len(index)
    weather_rng, cloud_rng, temperature_rng, rain_rng, power_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(config["seed"]).spawn(5)
    ]
    w, pv, station = config["weather"], config["pv"], config["station"]
    months = index.month.to_numpy() - 1
    states = np.empty(count, dtype=int)
    state = w["initial_state"]
    block_steps = int(w["regime_hours"] * 60 / config["interval_minutes"])
    for i in range(count):
        if i % block_steps == 0 and weather_rng.random() >= w["regime_persistence"]:
            state = int(weather_rng.choice(4, p=w["monthly_state_probabilities"][months[i]]))
        states[i] = state

    cloud = np.empty(count)
    c = w["cloud_means"][states[0]]
    for i, state in enumerate(states):
        c += w["cloud_relaxation"] * (w["cloud_means"][state] - c)
        c += cloud_rng.normal(0, w["cloud_innovation_std"][state])
        c = float(np.clip(c, 0, 1))
        cloud[i] = c

    east, north, up, elevation = solar_geometry(index, station["latitude_deg"], station["longitude_deg"])
    sun = np.maximum(up, 0)
    daylight = up > w["solar_sine_epsilon"]
    clear_ghi = np.zeros(count)
    clear_ghi[daylight] = (w["haurwitz_coefficient_w_m2"] * sun[daylight]
                          * np.exp(-w["haurwitz_attenuation"] / sun[daylight]))
    transmission = (w["cloud_transmission_floor"] + (1 - w["cloud_transmission_floor"])
                    * (1 - cloud) ** w["cloud_transmission_exponent"])
    ghi = clear_ghi * transmission
    diffuse_fraction = np.clip(w["clear_diffuse_fraction"] + w["cloud_diffuse_gain"]
                               * cloud ** w["cloud_diffuse_exponent"],
                               w["clear_diffuse_fraction"], w["maximum_diffuse_fraction"])
    dhi = ghi * diffuse_fraction
    dni = np.divide(ghi - dhi, sun, out=np.zeros(count), where=daylight)

    # Interpolate correlated daily anomalies rather than jumping at midnight.
    days = pd.date_range(index[0].normalize(), index[-1].normalize() + pd.Timedelta(days=1),
                         freq="D", tz=index.tz)
    anomaly_anchors = ar_noise(temperature_rng, len(days), w["daily_temperature_anomaly_ar"],
                               w["daily_temperature_anomaly_std_c"])
    daily_anomaly = np.interp(index.asi8.astype(float), days.asi8.astype(float), anomaly_anchors)
    hours = np.asarray(index.hour + index.minute / 60, dtype=float)
    amplitude = seasonal_monthly(index, w["monthly_temperature_daily_amplitude_c"])
    temperature = (seasonal_monthly(index, w["monthly_temperature_mean_c"]) + daily_anomaly
                   + amplitude * (1 - w["cloud_daily_amplitude_attenuation"] * cloud)
                   * np.cos(2 * np.pi * (hours - w["temperature_peak_local_hour"]) / 24)
                   + w["cloud_temperature_offset_c"] * cloud
                   + ar_noise(temperature_rng, count, w["temperature_noise_ar"], w["temperature_noise_std_c"]))
    temperature = np.clip(temperature, w["temperature_min_c"], w["temperature_max_c"])

    rain = np.zeros(count)
    wet = False
    for i, state in enumerate(states):
        scale = w["monthly_rain_scale_mm_per_interval"][months[i]]
        if state == 3:
            probability = w["rain_wet_continuation_probability"] if wet else w["rain_wet_start_probability"]
            wet = rain_rng.random() < probability
            if wet:
                rain[i] = rain_rng.gamma(w["rain_gamma_shape"], scale)
        else:
            wet = False
            if state == 2 and rain_rng.random() < w["overcast_drizzle_probability"]:
                rain[i] = rain_rng.gamma(w["rain_gamma_shape"], scale * w["drizzle_scale_fraction"])
        rain[i] = min(rain[i], w["rain_interval_max_mm"])

    tilt, azimuth = np.radians([pv["tilt_deg"], pv["azimuth_deg"]])
    incidence = up * np.cos(tilt) + np.sin(tilt) * (north * np.cos(azimuth) + east * np.sin(azimuth))
    poa = (dni * np.maximum(incidence, 0) + dhi * (1 + np.cos(tilt)) / 2
           + ghi * pv["albedo"] * (1 - np.cos(tilt)) / 2)
    cell_temperature = temperature + (pv["noct_c"] - 20) / 800 * poa
    temperature_factor = np.maximum(0, 1 + pv["temperature_coefficient_per_c"]
                                    * (cell_temperature - pv["reference_cell_temperature_c"]))
    power_noise = np.clip(ar_noise(power_rng, count, pv["relative_noise_ar"],
                                   pv["relative_noise_stationary_std"]),
                           -pv["relative_noise_clip"], pv["relative_noise_clip"])
    power = (station["capacity_kw"] * poa / pv["reference_irradiance_w_m2"]
             * temperature_factor * (1 - pv["system_loss_fraction"]) * (1 + power_noise))
    power = np.clip(power, 0, station["capacity_kw"])
    power[~daylight] = 0
    frame = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in index],
        "pv_power_kw": power,
        "SWGDN": ghi, "DHI": dhi, "DNI": dni,
        "temperature_c": temperature, "rain": rain, "cloud_cover": cloud,
        "weather_type": np.asarray(w["states"])[states],
        "solar_elevation_deg": elevation,
        "station_id": station["station_id"], "is_simulated": "true",
    })
    frame = frame.loc[index >= expected[0], COLUMNS].reset_index(drop=True)
    frame[NUMERIC] = frame[NUMERIC].round(config["output"]["float_decimals"])
    return frame


def csv_bytes(frame: pd.DataFrame, config: dict) -> bytes:
    text = frame.to_csv(index=False, float_format=f"%.{config['output']['float_decimals']}f", lineterminator="\n")
    return text.encode(config["output"]["csv_encoding"])


def interval_integrals(values, step_hours):
    """Trapezoidal approximation; next year's midnight boundary is zero."""
    values = np.asarray(values, dtype=float)
    return (values + np.r_[values[1:], 0.0]) * step_hours / 2


def monthly_summary(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_convert(config["timezone"])
    hours = config["interval_minutes"] / 60
    energy = interval_integrals(frame["pv_power_kw"], hours)
    irradiation = interval_integrals(frame["SWGDN"], hours) / 1000
    items = []
    for month in range(1, 13):
        mask = index.month == month
        f = frame.loc[mask]
        items.append({
            "month": f"{config['year']}-{month:02d}", "station_id": config["station"]["station_id"],
            "sample_count": int(mask.sum()), "simulated_energy_kwh": float(energy[mask].sum()),
            "peak_power_kw": float(f["pv_power_kw"].max()),
            "equivalent_full_load_hours": float(energy[mask].sum() / config["station"]["capacity_kw"]),
            "ghi_irradiation_kwh_m2": float(irradiation[mask].sum()),
            "rainfall_mm": float(f["rain"].sum()), "mean_temperature_c": float(f["temperature_c"].mean()),
            "daylight_hours_approx": float((f["solar_elevation_deg"] > 0).sum() * hours),
            "rainy_intervals": int((f["rain"] > 0).sum()), "is_simulated": "true",
        })
    return pd.DataFrame(items)


def validate_frame(frame: pd.DataFrame, config: dict) -> dict:
    require(list(frame.columns) == COLUMNS, "CSV columns or order differ from contract")
    expected = year_index(config)
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_convert(config["timezone"])
    require(len(frame) == len(expected) and index.equals(expected), "Year coverage/order/duplicates/15-minute grid failed")
    require(frame["timestamp"].str.endswith("+08:00").all(), "Every timestamp must explicitly use +08:00")
    require(not frame.isna().any().any(), "Missing values found")
    require(np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all(), "Non-finite numerical values found")
    require(frame["station_id"].eq(config["station"]["station_id"]).all(), "Station id mismatch")
    require(frame["is_simulated"].astype(str).str.lower().eq("true").all(), "Simulation flags missing")
    require(frame["pv_power_kw"].between(0, config["station"]["capacity_kw"]).all(), "Power outside simulated capacity")
    require((frame[["SWGDN", "DHI", "DNI", "rain"]] >= 0).all().all(), "Negative irradiation or rainfall")
    require(frame["cloud_cover"].between(0, 1).all(), "Cloud cover outside [0,1]")
    require(frame["temperature_c"].between(config["weather"]["temperature_min_c"],
                                           config["weather"]["temperature_max_c"]).all(), "Temperature out of bounds")
    night = frame["solar_elevation_deg"] <= 0
    require(frame.loc[night, ["pv_power_kw", "SWGDN", "DHI", "DNI"]].eq(0).all().all(), "Nonzero night power/irradiance")
    cos_z = np.maximum(np.sin(np.radians(frame["solar_elevation_deg"])), 0)
    identity_error = np.abs(frame["SWGDN"] - frame["DHI"] - frame["DNI"] * cos_z)
    require(identity_error.max() <= config["output"]["irradiance_tolerance_w_m2"], "GHI/DHI/DNI identity failed")
    computed_elevation = solar_geometry(index, config["station"]["latitude_deg"], config["station"]["longitude_deg"])[-1]
    require(np.max(np.abs(computed_elevation - frame["solar_elevation_deg"])) <= 1e-6, "Solar geometry mismatch")
    require(set(frame["weather_type"]) == set(config["weather"]["states"]), "Expected all four weather regimes")
    days = frame["pv_power_kw"].to_numpy().reshape(-1, 96)
    require(np.all(np.any(np.abs(np.diff(days, axis=0)) > 0.001, axis=1)), "Repeated adjacent daily curves")
    daylight_counts = (frame["solar_elevation_deg"].to_numpy().reshape(-1, 96) > 0).sum(axis=1)
    require(daylight_counts.max() > daylight_counts.min(), "No seasonal daylight variation")
    require((frame["rain"] > 0).any() and (frame["rain"] == 0).any(), "Wet/dry variability missing")
    return {
        "row_count": len(frame), "days": len(frame) // 96, "start": frame["timestamp"].iloc[0],
        "end": frame["timestamp"].iloc[-1], "missing_values": 0,
        "power_min_kw": float(frame["pv_power_kw"].min()), "power_max_kw": float(frame["pv_power_kw"].max()),
        "night_rows": int(night.sum()), "irradiance_identity_max_abs_error_w_m2": float(identity_error.max()),
        "weather_counts": {str(k): int(v) for k, v in frame["weather_type"].value_counts().items()},
        "temperature_min_c": float(frame["temperature_c"].min()), "temperature_max_c": float(frame["temperature_c"].max()),
        "minimum_daylight_hours_approx": float(daylight_counts.min() / 4),
        "maximum_daylight_hours_approx": float(daylight_counts.max() / 4),
    }


def check_predictor(csv_path: Path, predictor_dir: Path) -> dict:
    require((predictor_dir / "pv_forecast" / "data.py").is_file(), f"Predictor not found: {predictor_dir}")
    sys.path.insert(0, str(predictor_dir.resolve()))
    module = importlib.import_module("pv_forecast.data")
    weather = ("SWGDN", "DHI", "DNI", "temperature_c", "rain")
    regular = module.load_and_regularize(csv_path, module.InputSpec(history_weather_columns=weather))
    require(regular.quality.missing_target_rows == 0 and regular.quality.missing_weather_rows == 0,
            "Predictor found missing values")
    require(regular.quality.interpolated_target_rows == 0 and regular.quality.interpolated_weather_rows == 0,
            "Predictor interpolation was unexpectedly required")
    require(len(regular.frame) == 35040, "Predictor changed annual row count")
    examples = []
    # One full-history/full-horizon example in every month; avoid training,
    # TensorFlow and the large overlapping full-year window tensor.
    for month in range(1, 13):
        start = int(np.flatnonzero(regular.frame.index.month == month)[0]) + 3 * 96
        for cols in [(), weather]:
            windows = module.build_windows(regular, start, start + 96, history_weather_columns=cols)
            require(windows.history_power.shape == (1, 288, 1), "History window shape failed")
            require(windows.target.shape == (1, 96), "Target window shape failed")
            require(windows.future_features.shape == (1, 96, 5), "Calendar feature shape failed")
            require(windows.gru_history.shape == (1, 288, 1 + len(cols)), "Historical weather shape failed")
            require(np.isfinite(windows.future_features).all(), "Non-finite forecast features")
        examples.append(regular.frame.index[start].isoformat())
    return {
        "status": "passed", "quality": regular.quality.as_dict(),
        "sample_forecast_starts": examples, "history_shape": [1, 288, 1], "target_shape": [1, 96],
        "calendar_feature_shape": [1, 96, 5], "observed_weather_history_shape": [1, 288, 6],
        "training_performed": False,
        "scope": "仅验证数据读取及样本构造，不验证模型训练、时间分区、预测精度或发布能力",
        "data_module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "simulation_config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--predictor-dir", type=Path, default=HERE.parents[1] / "光伏")
    parser.add_argument("--validate-only", action="store_true", help="Keep data unchanged; refresh validation report")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    require(config["year"] == 2025, "This delivery and its compatibility check target year 2025")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / config["output"]["data_filename"]
    summary_path = directory / config["output"]["monthly_filename"]
    report_path = directory / config["output"]["report_filename"]
    report = {"dataset_id": config["dataset_id"], "is_simulated": True, "status": "failed"}
    try:
        generated = simulate(config)
        validate_frame(generated, config)
        generated_bytes = csv_bytes(generated, config)
        summary = monthly_summary(generated, config)
        # Independent regeneration checks the generator itself, not just a file copy.
        second_bytes = csv_bytes(simulate(config), config)
        require(generated_bytes == second_bytes, "Regeneration is not byte-identical")
        if not args.validate_only:
            data_path.write_bytes(generated_bytes)
            summary_path.write_bytes(csv_bytes(summary, config))
            if args.config.resolve() != (directory / "simulation_config.json").resolve():
                write_json(directory / "simulation_config.json", config)
        require(data_path.read_bytes() == generated_bytes, "Existing CSV differs from seeded regeneration")
        loaded = pd.read_csv(data_path, encoding=config["output"]["csv_encoding"], dtype={"is_simulated": str})
        facts = validate_frame(loaded, config)
        loaded_summary = pd.read_csv(summary_path, encoding=config["output"]["csv_encoding"], dtype={"is_simulated": str})
        require(list(loaded_summary.columns) == list(summary.columns) and len(loaded_summary) == 12,
                "Monthly summary format mismatch")
        require(summary_path.read_bytes() == csv_bytes(monthly_summary(loaded, config), config),
                "Monthly summary differs from exported data integration")
        annual_energy = float(interval_integrals(loaded["pv_power_kw"], 0.25).sum())
        require(abs(loaded_summary["simulated_energy_kwh"].sum() - annual_energy) < 12e-6,
                "Monthly and annual energy disagree beyond rounding tolerance")
        require(loaded_summary["sample_count"].sum() == len(loaded), "Monthly sample counts disagree")
        compatibility = check_predictor(data_path, args.predictor_dir)
        report.update({
            "status": "passed", "checks": facts,
            "annual_simulated_energy_kwh": annual_energy,
            "annual_equivalent_full_load_hours": annual_energy / config["station"]["capacity_kw"],
            "annual_ghi_irradiation_kwh_m2": float(interval_integrals(loaded["SWGDN"], 0.25).sum() / 1000),
            "annual_rainfall_mm": float(loaded["rain"].sum()),
            "energy_integration": config["output"]["energy_integration"],
            "reproducibility": {"seed": config["seed"], "repeat_generation_byte_identical": True,
                                "disk_matches_regeneration": True},
            "predictor_compatibility": compatibility,
            "hashes_sha256": {
                data_path.name: hashlib.sha256(data_path.read_bytes()).hexdigest(),
                summary_path.name: hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "config_canonical_json": hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "generator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "source_evidence": [
                {"path": s["path"], "used_for": s["used_for"], "not_established": s["not_established"],
                 "sha256": hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() if Path(s["path"]).is_file() else None}
                for s in config["sources"]
            ],
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "limitations": config["limitations"],
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=True))
        return 1
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "rows": facts["row_count"],
                      "peak_kw": facts["power_max_kw"], "energy_kwh": annual_energy,
                      "output_dir": str(directory), "training_performed": False}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
