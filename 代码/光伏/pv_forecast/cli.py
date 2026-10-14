"""Command line interface for the independent PV forecasting package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import InputSpec
from .pipeline import TrainingConfig, evaluate_bundle, predict_record, train_model
from .experiments import RollingFold, train_experiment


def _input_spec(args) -> InputSpec:
    weather = tuple(filter(None, (args.weather_columns or "").split(",")))
    mapping = {}
    if args.source_timestamp_column:
        mapping[args.source_timestamp_column] = "timestamp"
    if args.source_target_column:
        mapping[args.source_target_column] = "pv_power_kw"
    return InputSpec(
        timestamp_col="timestamp",
        target_col="pv_power_kw",
        weather_columns=weather,
        history_weather_columns=tuple(filter(None, args.history_weather_columns.split(","))),
        latitude=args.latitude, longitude=args.longitude,
        coordinate_basis="user_configured" if args.latitude is not None else "not_configured",
        timezone=args.timezone,
        input_kind=args.input_kind,
        interval_minutes=args.input_interval_minutes,
        column_map=mapping,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Photovoltaic power forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "evaluate", "recalibrate", "backtest", "correct-point"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input", required=True, help="CSV or Excel input")
        cmd.add_argument("--timezone", default="Asia/Shanghai")
        cmd.add_argument("--weather-columns", "--future-weather-columns", default="", help="future forecast columns; requires issued forecast archive")
        cmd.add_argument("--history-weather-columns", default="", help="observed weather used only before the forecast origin")
        cmd.add_argument("--future-weather-input", help="CSV/Excel archive: issued_at,timestamp,weather columns")
        cmd.add_argument("--latitude", type=float)
        cmd.add_argument("--longitude", type=float)
        cmd.add_argument("--source-timestamp-column", default=None)
        cmd.add_argument("--source-target-column", default=None)
        cmd.add_argument("--input-kind", choices=["power", "interval_energy"], default="power")
        cmd.add_argument("--input-interval-minutes", type=int, default=None)
    sub.choices["train"].add_argument("--output-dir", required=True)
    sub.choices["train"].add_argument("--interval-conditioning", choices=["pooled", "daylight"], default="pooled")
    sub.choices["train"].add_argument("--night-lower-bound", choices=["empirical", "zero"], default="empirical")
    sub.choices["train"].add_argument("--skip-gru", action="store_true", help="只训练并评估基线")
    sub.choices["train"].add_argument("--min-gru-days", type=int, default=60)
    sub.choices["train"].add_argument("--gru-epochs", type=int, default=50)
    sub.choices["train"].add_argument("--gru-batch-size", type=int, default=128)
    sub.choices["train"].add_argument("--gru-patience", type=int, default=8)
    sub.choices["train"].add_argument("--max-gru-samples", type=int, default=0, help="0 = all windows")
    sub.choices["train"].add_argument("--gru-units", type=int, default=32)
    sub.choices["train"].add_argument("--require-gru", action="store_true", help="GRU 无法训练时直接报错")
    sub.choices["train"].add_argument("--quiet", action="store_true", help="关闭终端训练日志，仍保存到 training.log")
    sub.choices["train"].add_argument("--learning-rate", type=float, default=0.0003)
    sub.choices["train"].add_argument("--lr-patience", type=int, default=2)
    sub.choices["train"].add_argument("--lr-factor", type=float, default=0.5)
    sub.choices["train"].add_argument("--min-lr", type=float, default=0.00001)
    sub.choices["train"].add_argument("--min-delta", type=float, default=0.00001)
    sub.choices["train"].add_argument("--no-reduce-lr", action="store_true")
    sub.choices["train"].add_argument("--rolling-validation", action="store_true")
    sub.choices["train"].add_argument("--folds", help="JSON list of name/train_end/validation_end (exclusive)")
    sub.choices["predict"].add_argument("--bundle-dir", required=True)
    sub.choices["predict"].add_argument("--output", required=True)
    sub.choices["predict"].add_argument("--forecast-start", default=None)
    sub.choices["evaluate"].add_argument("--bundle-dir", required=True)
    sub.choices["evaluate"].add_argument("--output", required=False)
    sub.choices["recalibrate"].add_argument("--bundle-dir", required=True)
    sub.choices["recalibrate"].add_argument("--output-dir", required=True)
    sub.choices["recalibrate"].add_argument("--nominal-coverage", type=float, default=None)
    sub.choices["recalibrate"].add_argument("--min-step-samples", type=int, default=30)
    sub.choices["recalibrate"].add_argument("--night-lower-bound", choices=["empirical", "zero"], default=None)
    sub.choices["backtest"].add_argument("--bundle-dir", nargs="+", required=True)
    sub.choices["backtest"].add_argument("--output", required=True)
    sub.choices["backtest"].add_argument("--batch-windows", type=int, default=512)
    sub.choices["correct-point"].add_argument("--bundle-dir", required=True)
    sub.choices["correct-point"].add_argument("--output-dir", required=True)
    sub.choices["correct-point"].add_argument("--batch-windows", type=int, default=1024)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    spec = _input_spec(args)
    if args.command == "train":
        config = TrainingConfig(
            skip_gru=args.skip_gru,
            min_gru_days=args.min_gru_days,
            gru_epochs=args.gru_epochs,
            gru_batch_size=args.gru_batch_size,
            gru_patience=args.gru_patience,
            max_gru_samples=args.max_gru_samples,
            gru_units=args.gru_units,
            require_gru=args.require_gru,
            gru_learning_rate=args.learning_rate, reduce_lr=not args.no_reduce_lr,
            lr_patience=args.lr_patience, lr_factor=args.lr_factor,
            min_lr=args.min_lr, min_delta=args.min_delta,
            interval_conditioning=args.interval_conditioning,
            night_lower_bound=args.night_lower_bound,
        )
        progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
        if args.folds and not args.rolling_validation:
            raise ValueError("--folds requires --rolling-validation")
        if args.rolling_validation:
            if args.no_reduce_lr or args.gru_units != 32:
                raise ValueError("rolling validation uses the fixed control/scheduled32/scheduled16 candidates")
            folds = [RollingFold(**fold) for fold in json.loads(Path(args.folds).read_text(encoding="utf-8"))] if args.folds else None
            result = train_experiment(args.input, args.output_dir, spec, config, progress=progress,
                                      folds=folds, future_weather_source=args.future_weather_input)
        else:
            result = train_model(args.input, args.output_dir, spec, config, progress=progress,
                                 future_weather_source=args.future_weather_input)
    elif args.command == "predict":
        result = predict_record(args.bundle_dir, args.input, args.output, spec, args.forecast_start,
                                future_weather_source=args.future_weather_input)
    elif args.command == "recalibrate":
        from .recalibration import recalibrate_bundle
        result = recalibrate_bundle(args.bundle_dir, args.input, args.output_dir, spec,
            nominal_coverage=args.nominal_coverage, min_step_samples=args.min_step_samples,
            future_weather_source=args.future_weather_input,
            night_lower_bound=args.night_lower_bound,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    elif args.command == "correct-point":
        from .point_experiments import fit_point_correction_bundle
        result = fit_point_correction_bundle(args.bundle_dir, args.input, args.output_dir, spec,
            batch_windows=args.batch_windows, future_weather_source=args.future_weather_input,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    elif args.command == "backtest":
        from .backtesting import backtest_bundles
        result = backtest_bundles(args.bundle_dir, args.input, args.output, spec,
            batch_windows=args.batch_windows, future_weather_source=args.future_weather_input,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    else:
        visualizations_dir = (
            Path(args.output).parent / "visualizations" if args.output else None
        )
        result = evaluate_bundle(args.bundle_dir, args.input, spec, visualizations_dir, args.future_weather_input)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
