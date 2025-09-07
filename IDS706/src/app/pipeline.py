from pathlib import Path
import polars as pl

from ..utils.config import ensure_dirs, list_parquet_files
from ..settings import (
    RAW_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    BRONZE_PATH,
    SILVER_PATH,
    TOP_SKILLS_PATH,
    TARGET_ROLES,
)
from ..infra.io_polars import PolarsLocalRepository
from ..infra.transformers import (
    CleanJobTransformer,
    RoleFilterTransformer,
    TextJoinTransformer,
)
from ..infra.aggregators import TopSkillsAggregator


class JobsPipeline:
    """Orchestrates raw -> bronze -> silver -> gold tables."""

    def __init__(self):
        self.repo = PolarsLocalRepository()
        self.cleaner = CleanJobTransformer()
        self.role_filter = RoleFilterTransformer(TARGET_ROLES)
        self.texter = TextJoinTransformer()
        self.topskills = TopSkillsAggregator(topk=40)

    def build(self) -> None:
        """Run the end-to-end table build with whatever is in data/raw."""
        ensure_dirs(BRONZE_DIR, SILVER_DIR, GOLD_DIR)

        # 1) Load all raw parquet files
        raw_files = [str(p) for p in list_parquet_files(RAW_DIR)]
        if not raw_files:
            raise FileNotFoundError(f"No .parquet files found in {RAW_DIR}")

        lf = self.repo.load_many(raw_files)

        # 2) Clean/normalize -> bronze
        lf_bronze = self.cleaner.run(lf)
        self.repo.save_lazy(lf_bronze, str(BRONZE_PATH))

        # 3) Role filter + text join -> silver
        lf_silver = self.role_filter.run(lf_bronze)
        lf_silver = self.texter.run(lf_silver)
        self.repo.save_lazy(lf_silver, str(SILVER_PATH))

        # 4) Top skills aggregate -> gold
        lf_top = self.topskills.aggregate(lf_silver)
        self.repo.save_lazy(lf_top, str(TOP_SKILLS_PATH))

        # Optional: small console hints (no heavy collect)
        print(f"✅ Bronze written: {BRONZE_PATH}")
        print(f"✅ Silver written: {SILVER_PATH}")
        print(f"✅ Top skills written: {TOP_SKILLS_PATH}")
