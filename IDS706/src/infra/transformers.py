# src/infra/transformers.py
import re
import polars as pl
from ..domain.ports import Transformer


# -------------------------
# 1) Cleaning / normalization
# -------------------------
class CleanJobTransformer(Transformer):
    """Normalize raw columns into a consistent schema and basic typing."""

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        cols_set = set(lf.collect_schema().names())

        def first_present(cands: list[str]) -> pl.Expr:
            for c in cands:
                if c in cols_set:
                    return pl.col(c)
            return pl.lit(None)

        title_candidates = ["job_title", "title", "position"]
        company_candidates = ["company", "company_name", "employer"]
        location_candidates = ["location", "job_location", "city"]
        # ← 加了 Kaggle 常見欄位
        posted_candidates = [
            "posted_time",
            "posted_at",
            "date_posted",
            "post_date",
            "first_seen",
            "last_processed_time",
            "created_at",
            "timestamp",
        ]
        desc_candidates = ["description", "job_description", "desc", "job_summary"]
        skills_candidates = ["skills", "skill_list", "tags", "job_skills"]
        work_type_candidates = [
            "work_type",
            "job_type",
            "onsite_remote",
            "onsite_remote_hybrid",
            "employment_type",
            "remote_status",
        ]
        seniority_candidates = ["seniority", "experience_level", "level", "job_level"]

        out = (
            lf
            # 1) 先把「來源欄位 → 標準欄位」映射好
            .with_columns(
                [
                    first_present(title_candidates).cast(pl.Utf8).alias("title"),
                    first_present(company_candidates).cast(pl.Utf8).alias("company"),
                    first_present(location_candidates).cast(pl.Utf8).alias("location"),
                    first_present(posted_candidates).cast(pl.Utf8).alias("posted_raw"),
                    first_present(desc_candidates).cast(pl.Utf8).alias("desc"),
                    first_present(work_type_candidates)
                    .cast(pl.Utf8)
                    .alias("work_type"),
                    first_present(seniority_candidates)
                    .cast(pl.Utf8)
                    .alias("seniority"),
                    first_present(skills_candidates).cast(pl.Utf8).alias("skills_raw"),
                ]
            )
            # 2) 立刻建立 title_lc 與 posted_at（避免後面 select 找不到）
            .with_columns(
                [
                    pl.when(pl.col("title").is_not_null())
                    .then(pl.col("title").str.to_lowercase())
                    .otherwise(pl.lit(None))
                    .alias("title_lc"),
                    pl.coalesce(
                        [
                            # 含時區的字串先用 UTC 解析，再移除時區
                            pl.col("posted_raw")
                            .str.strptime(pl.Datetime(time_zone="UTC"), strict=False)
                            .dt.replace_time_zone(None),
                            # 再嘗試一般無時區字串
                            pl.col("posted_raw").str.strptime(
                                pl.Datetime, strict=False
                            ),
                        ]
                    ).alias("posted_at"),
                ]
            )
            # 3) 基本過濾
            .filter(pl.col("title").is_not_null() & pl.col("company").is_not_null())
            # 4) skills 正規化（JSON list 或逗號分隔）
            .with_columns(
                [
                    pl.when(pl.col("skills_raw").is_not_null())
                    .then(
                        pl.when(pl.col("skills_raw").str.starts_with("["))
                        .then(pl.col("skills_raw").str.json_decode(pl.List(pl.Utf8)))
                        .otherwise(pl.col("skills_raw").str.split(","))
                    )
                    .otherwise(pl.lit([]))
                    .list.eval(
                        pl.element().cast(pl.Utf8).str.strip_chars().str.to_lowercase()
                    )
                    .alias("skills_list")
                ]
            )
            # 5) 最終投影
            .select(
                [
                    "title",
                    "title_lc",
                    "company",
                    "location",
                    "desc",
                    "work_type",
                    "seniority",
                    "posted_at",
                    "skills_list",
                ]
            )
        )
        return out


# -------------------------
# 2) Role filter
# -------------------------
class RoleFilterTransformer(Transformer):
    """Keep only rows whose title matches any of the target roles (regex)."""

    def __init__(self, roles: list[str]):
        self.roles = roles

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not self.roles:
            return lf
        # Build a single regex: \b(role1|role2|role3)\b
        pattern = r"\b(" + "|".join(re.escape(r) for r in self.roles) + r")\b"
        return lf.filter(pl.col("title_lc").str.contains(pattern))


# -------------------------
# 3) Text join for NLP
# -------------------------
class TextJoinTransformer(Transformer):
    """Concatenate title/description/skills into a single text column for NLP."""

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("title_lc").fill_null(""),
                        pl.lit(" "),
                        pl.col("desc").fill_null("").str.to_lowercase(),
                        pl.lit(" "),
                        pl.col("skills_list").list.join(" "),
                    ],
                    separator="",
                ).alias("text")
            ]
        ).select(
            [
                "title_lc",
                "company",
                "location",
                "seniority",
                "work_type",
                "posted_at",
                "skills_list",
                "text",
            ]
        )


# -------------------------
# 4) Derive work_type/seniority when missing
# -------------------------
class DeriveWorkTypeTransformer(Transformer):
    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [
                pl.when(pl.col("work_type").is_not_null() & (pl.col("work_type") != ""))
                .then(pl.col("work_type"))
                .otherwise(
                    pl.when(
                        pl.col("text").str.contains(r"\b(remote|work from home|wfh)\b")
                    )
                    .then(pl.lit("remote"))
                    .when(pl.col("text").str.contains(r"\bhybrid\b"))
                    .then(pl.lit("hybrid"))
                    .when(pl.col("text").str.contains(r"\b(on[- ]?site)\b"))
                    .then(pl.lit("onsite"))  # onsite/on-site/on site
                    .otherwise(pl.lit("NA"))
                )
                .alias("work_type")
            ]
        )


class DeriveSeniorityTransformer(Transformer):
    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [
                pl.when(pl.col("seniority").is_not_null() & (pl.col("seniority") != ""))
                .then(pl.col("seniority"))
                .otherwise(
                    pl.when(pl.col("title_lc").str.contains(r"\bintern(ship)?\b"))
                    .then(pl.lit("intern"))
                    .when(pl.col("title_lc").str.contains(r"\bjunior|jr\.?\b"))
                    .then(pl.lit("junior"))
                    .when(pl.col("title_lc").str.contains(r"\bmid(-| )?level\b"))
                    .then(pl.lit("mid"))
                    .when(pl.col("title_lc").str.contains(r"\bsenior|sr\.?\b"))
                    .then(pl.lit("senior"))
                    .when(pl.col("title_lc").str.contains(r"\blead\b"))
                    .then(pl.lit("lead"))
                    .when(pl.col("title_lc").str.contains(r"\b(principal|staff)\b"))
                    .then(pl.lit("principal"))
                    .when(pl.col("title_lc").str.contains(r"\bmanager|mgr\.?\b"))
                    .then(pl.lit("manager"))
                    .otherwise(pl.lit("NA"))
                )
                .alias("seniority")
            ]
        )
