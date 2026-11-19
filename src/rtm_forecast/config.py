"""Pipeline settings read from params.yaml."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataParams:
    raw_path: Path
    clean_path: Path
    report_path: Path
    price_min: float
    price_max: float
    bid_min: float
    bid_max: float
    demand_min: float
    demand_max: float
    stale_min_blocks: int


def load_data_params(path: Path = Path("params.yaml")) -> DataParams:
    with path.open(encoding="utf-8") as handle:
        section = yaml.safe_load(handle)["data"]
    return DataParams(
        raw_path=Path(section["raw_path"]),
        clean_path=Path(section["clean_path"]),
        report_path=Path(section["report_path"]),
        price_min=float(section["price_min"]),
        price_max=float(section["price_max"]),
        bid_min=float(section["bid_min"]),
        bid_max=float(section["bid_max"]),
        demand_min=float(section["demand_min"]),
        demand_max=float(section["demand_max"]),
        stale_min_blocks=int(section["stale_min_blocks"]),
    )


@dataclass(frozen=True)
class FeatureParams:
    features_path: Path
    report_path: Path
    imputation: str
    lags_days: tuple[int, ...]
    min_history_days: int
    demand_policy: str
    demand_reliable_from: date
    asinh_scales: tuple[float, ...]
    asinh_train_days: int
    holidays_path: Path
    stale_min_blocks: int


def load_feature_params(path: Path = Path("params.yaml")) -> FeatureParams:
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    section = settings["features"]
    return FeatureParams(
        features_path=Path(section["features_path"]),
        report_path=Path(section["report_path"]),
        imputation=str(section["imputation"]),
        lags_days=tuple(int(d) for d in section["lags_days"]),
        min_history_days=int(section["min_history_days"]),
        demand_policy=str(section["demand_policy"]),
        demand_reliable_from=date.fromisoformat(str(section["demand_reliable_from"])),
        asinh_scales=tuple(float(x) for x in section["asinh_scales"]),
        asinh_train_days=int(section["asinh_train_days"]),
        holidays_path=Path(section["holidays_path"]),
        stale_min_blocks=int(settings["data"]["stale_min_blocks"]),
    )


@dataclass(frozen=True)
class ExperimentParams:
    validation_days: int
    test_days: int
    seed: int
    tuning_trials: int
    tuning_refit_every_days: int
    naive_reference: str


def load_experiment_params(path: Path = Path("params.yaml")) -> ExperimentParams:
    with path.open(encoding="utf-8") as handle:
        section = yaml.safe_load(handle)["experiment"]
    return ExperimentParams(
        validation_days=int(section["validation_days"]),
        test_days=int(section["test_days"]),
        seed=int(section["seed"]),
        tuning_trials=int(section["tuning_trials"]),
        tuning_refit_every_days=int(section["tuning_refit_every_days"]),
        naive_reference=str(section["naive_reference"]),
    )


@dataclass(frozen=True)
class WindowParams:
    start: date
    end: date
    rtm_dir: Path
    dam_dir: Path
    psp_dir: Path
    clean_path: Path
    report_path: Path


def load_window_params(path: Path = Path("params.yaml")) -> WindowParams:
    with path.open(encoding="utf-8") as handle:
        section = yaml.safe_load(handle)["window"]
    return WindowParams(
        start=date.fromisoformat(str(section["start"])),
        end=date.fromisoformat(str(section["end"])),
        rtm_dir=Path(section["rtm_dir"]),
        dam_dir=Path(section["dam_dir"]),
        psp_dir=Path(section["psp_dir"]),
        clean_path=Path(section["clean_path"]),
        report_path=Path(section["report_path"]),
    )
