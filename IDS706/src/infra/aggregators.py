import polars as pl
from ..domain.ports import Aggregator


class TopSkillsAggregator(Aggregator):
    def __init__(self, topk: int = 40):
        self.topk = topk

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.select(["skills_list"])
            .explode("skills_list")
            .filter(pl.col("skills_list").is_not_null() & (pl.col("skills_list") != ""))
            .group_by("skills_list")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(self.topk)
        )
