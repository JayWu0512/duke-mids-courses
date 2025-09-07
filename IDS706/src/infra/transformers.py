# src/infra/transformers.py
import re
import polars as pl
from ..domain.ports import Transformer


class CleanJobTransformer(Transformer):
    """Normalize columns and produce a consistent schema."""

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        # Cache schema names once to avoid repeated expensive resolution
        cols_set = set(lf.collect_schema().names())

        def first_present(cands: list[str]) -> pl.Expr:
            for c in cands:
                if c in cols_set:
                    return pl.col(c)
            return pl.lit(None)

        title_candidates = ["job_title", "title", "position"]
        company_candidates = ["company", "company_name", "employer"]
        location_candidates = ["location", "job_location", "city"]
        posted_candidates = ["posted_time", "posted_at", "date_posted", "post_date"]
        desc_candidates = ["description", "job_description", "desc"]
        skills_candidates = ["skills", "skill_list", "tags"]
        work_type_candidates = ["work_type", "onsite_remote", "onsite_remote_hybrid"]
        seniority_candidates = ["seniority", "experience_level"]

        out = (
            lf.with_columns(
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
            .filter(pl.col("title").is_not_null() & pl.col("company").is_not_null())
            .with_columns(
                [
                    pl.col("title").str.to_lowercase().alias("title_lc"),
                    pl.col("posted_raw")
                    .str.strptime(pl.Datetime, strict=False)
                    .alias("posted_at"),
                ]
            )
            # Normalize skills: JSON list → List[Utf8]; else comma split
            .with_columns(
                [
                    pl.when(pl.col("skills_raw").is_not_null())
                    .then(
                        pl.when(pl.col("skills_raw").str.starts_with("["))
                        .then(
                            pl.col("skills_raw").str.json_decode(pl.List(pl.Utf8))
                        )  # dtype required by your Polars
                        .otherwise(pl.col("skills_raw").str.split(","))
                    )
                    .otherwise(pl.lit([]))
                    .list.eval(
                        pl.element().cast(pl.Utf8).str.strip_chars().str.to_lowercase()
                    )
                    .alias("skills_list")
                ]
            )
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


class RoleFilterTransformer(Transformer):
    """Keep only rows whose title matches any of the target roles (regex)."""

    def __init__(self, roles: list[str]):
        self.roles = roles

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        # If no roles specified, just pass-through
        if not self.roles:
            return lf

        # Build a single regex: \b(role1|role2|role3)\b
        pattern = r"\b(" + "|".join(re.escape(r) for r in self.roles) + r")\b"
        return lf.filter(pl.col("title_lc").str.contains(pattern))


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
