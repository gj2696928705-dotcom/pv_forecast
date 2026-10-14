"""黄花站模拟数据：完整 GRU 训练、可视化、验证选模与留出测试。"""

from pathlib import Path
from datetime import datetime
import json
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# 以脚本位置定位项目，可从任意工作目录启动。
code_dir = Path(__file__).resolve().parent
root = code_dir.parent.parent
sys.path.insert(0, str(code_dir))

from pv_forecast.data import InputSpec
from pv_forecast.pipeline import (
    TrainingConfig,
    train_model,
    predict_record,
    evaluate_bundle,
)
from pv_forecast.experiments import RollingFold, train_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=(
        root / "代码" / "数据" / "光伏数据" / "黄花站_2025年_15分钟光伏模拟数据.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=0.00001)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--history-weather-columns", default="SWGDN,temperature_c,cloud_cover")
    parser.add_argument("--future-weather-columns", default="")
    parser.add_argument("--future-weather-input", type=Path, help="含 issued_at 与 timestamp 的预报归档")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--single-split", action="store_true", help="显式跳过滚动对照，仅训练一组模型")
    parser.add_argument("--folds", type=Path, help="自定义滚动验证 JSON；默认验证 5、7、9 月")
    parser.add_argument("--max-gru-samples", type=int, default=0,
                        help="默认 0 表示全部窗口；正数用于显式限制调试样本量")
    args = parser.parse_args(argv)
    output = args.output_dir or code_dir / "运行结果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    latitude, longitude = args.latitude, args.longitude
    coordinate_basis = "user_configured" if latitude is not None else "not_configured"
    default_input = root / "代码" / "数据" / "光伏数据" / "黄花站_2025年_15分钟光伏模拟数据.csv"
    if latitude is None and longitude is None and args.input.resolve() == default_input.resolve():
        station = json.loads((default_input.parent / "simulation_config.json").read_text(encoding="utf-8"))["station"]
        latitude, longitude = station["latitude_deg"], station["longitude_deg"]
        coordinate_basis = station["coordinate_basis"]
    spec = InputSpec(timezone="Asia/Shanghai", latitude=latitude, longitude=longitude,
        coordinate_basis=coordinate_basis,
        history_weather_columns=tuple(filter(None, args.history_weather_columns.split(","))),
        weather_columns=tuple(filter(None, args.future_weather_columns.split(","))))
    if args.folds and args.single_split:
        parser.error("--folds 仅用于滚动验证")
    if not args.single_split and args.units != 32:
        parser.error("滚动验证固定比较 32/32/16 单元；自定义 --units 请配合 --single-split")
    config = TrainingConfig(
        skip_gru=False, require_gru=True, max_gru_samples=args.max_gru_samples,
        gru_epochs=args.epochs, gru_batch_size=args.batch_size,
        gru_patience=args.patience, gru_units=args.units,
        gru_learning_rate=args.learning_rate, lr_patience=args.lr_patience,
        lr_factor=args.lr_factor, min_lr=args.min_lr, min_delta=args.min_delta,
    )
    def progress(message):
        print(message, flush=True)

    progress("开始离线 GRU 训练（默认黄花站文件为模拟数据）。")
    progress(f"输出目录：{output}")
    try:
        if args.single_split:
            result = train_model(args.input, output, spec, config, progress=progress,
                                 future_weather_source=args.future_weather_input)
        else:
            folds = [RollingFold(**fold) for fold in json.loads(args.folds.read_text(encoding="utf-8"))] if args.folds else None
            result = train_experiment(args.input, output, spec, config, progress=progress,
                                     folds=folds, future_weather_source=args.future_weather_input)
    except Exception as exc:
        progress(f"训练失败：{exc}")
        raise

    progress("生成未来 24 小时预测、参考图片及末日回测……")
    forecast = predict_record(output, args.input, output / "forecast.json", spec,
                              asset_scope="HH-Huanghua-CSS", future_weather_source=args.future_weather_input)
    evaluation = evaluate_bundle(output, args.input, spec, output / "visualizations", args.future_weather_input)
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # 只显示异常，不逐条打印重复路径和 JSON 内容。
    for record in (result["metrics"]["gru"]["training"], result["metrics"], forecast, evaluation):
        for error in record.get("visualization_errors", []):
            progress(f"图片生成失败 [{error['id']}]：{error['message']}")
        if record.get("status") in {"unavailable", "degraded"}:
            progress(f"输出状态 {record['status']}："
                     f"{record.get('reason') or record.get('confidence', {}).get('limitations', [])}")
    progress("完成。权重、逐轮指标、训练曲线与预测结果已保存到上述目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
