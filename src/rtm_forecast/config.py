"""Pipeline settings read from params.yaml."""

from dataclasses import dataclass
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
