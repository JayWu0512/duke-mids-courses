# scripts/make_test_data.py
from __future__ import annotations

import pathlib
import random
import sys
from typing import Iterable, Optional, List

import polars as pl

# -------------------------
# Paths & constants
# -------------------------
PROJ = pathlib.Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
OUT = PROJ / "data" / "test"
SEED = 42
random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)

# Ensure project src on sys.path (future-proof)
if str(PROJ / "src") not in sys.path:
    sys.path.append(str(PROJ / "src"))


# -------------------------
# Helpers
# -------------------------
def stratified_sample(df: pl.DataFrame, by: str, n_per_group: int) -> pl.DataFrame:
    """Stratified sampling: take up to n_per_group rows from each group."""
    if by not in df.columns:
        raise ValueError(
            f'Stratified sample "by" column "{by}" not found. Available: {df.columns}'
        )
    return (
        df.with_columns(pl.int_range(0, pl.len()).shuffle(seed=SEED).alias("__rand__"))
        .with_columns(pl.col("__rand__").rank("ordinal").over(by).alias("__rk__"))
        .filter(pl.col("__rk__") <= n_per_group)
        .drop(["__rand__", "__rk__"])
    )


def read_if_exists(path: pathlib.Path) -> Optional[pl.LazyFrame]:
    """Return a LazyFrame if parquet exists, else None."""
    return pl.scan_parquet(path) if path.exists() else None


def normalize_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalize raw columns into a consistent schema:
      job_id, title, title_lc, company, location, desc, posted_at, skills_list
    (Lightweight version of your CleanJobTransformer.)
    """
    # collect_schema per-source (avoid concat issues)
    schema_names = set(lf.collect_schema().names())

    def pick(cands: Iterable[str]) -> pl.Expr:
        for c in cands:
            if c in schema_names:
                return pl.col(c)
        return pl.lit(None)

    title_candidates = ["job_title", "title", "position"]
    company_candidates = ["company", "company_name", "employer"]
    location_candidates = ["location", "job_location", "city"]
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
    jobid_candidates = ["job_id", "id", "jobkey", "posting_id"]

    lf = (
        lf.with_columns(
            [
                pick(title_candidates).cast(pl.Utf8).alias("title"),
                pick(company_candidates).cast(pl.Utf8).alias("company"),
                pick(location_candidates).cast(pl.Utf8).alias("location"),
                pick(posted_candidates).cast(pl.Utf8).alias("posted_raw"),
                pick(desc_candidates).cast(pl.Utf8).alias("desc"),
                pick(skills_candidates).cast(pl.Utf8).alias("skills_raw"),
                pick(jobid_candidates).alias("job_id"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("title").is_not_null())
                .then(pl.col("title").str.to_lowercase())
                .otherwise(pl.lit(None))
                .alias("title_lc"),
                pl.coalesce(
                    [
                        pl.col("posted_raw")
                        .str.strptime(pl.Datetime(time_zone="UTC"), strict=False)
                        .dt.replace_time_zone(None),
                        pl.col("posted_raw").str.strptime(pl.Datetime, strict=False),
                    ]
                ).alias("posted_at"),
            ]
        )
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
        .select(
            [
                "job_id",
                "title",
                "title_lc",
                "company",
                "location",
                "desc",
                "posted_at",
                "skills_list",
            ]
        )
    )

    # Filter obvious junk
    lf = lf.filter(pl.col("title").is_not_null() & pl.col("company").is_not_null())
    return lf


# -------------------------
# Main
# -------------------------
def main() -> None:
    # Load raw sources (each may or may not exist)
    job_summary = read_if_exists(RAW / "job_summary.parquet")
    linkedin_posts = read_if_exists(RAW / "linkedin_job_postings.parquet")
    job_skills_lf = read_if_exists(RAW / "job_skills.parquet")

    # presence check
    if job_summary is None and linkedin_posts is None:
        raise FileNotFoundError(
            "No job postings found under data/raw/. Expected one of: "
            "job_summary.parquet or linkedin_job_postings.parquet"
        )

    # Normalize each source first
    normalized_sources: List[pl.LazyFrame] = []
    for src in (job_summary, linkedin_posts):
        if src is not None:
            normalized_sources.append(normalize_lazy(src))

    # Concat after normalization (same schema)
    base_norm = (
        normalized_sources[0]
        if len(normalized_sources) == 1
        else pl.concat(normalized_sources, how="vertical")
    )

    # Collect to DataFrame for sampling
    jobs_df = base_norm.collect()

    if (
        "title_lc" not in jobs_df.columns
        or jobs_df["title_lc"].null_count() == jobs_df.height
    ):
        raise ValueError(
            "No usable title/title_lc found after normalization. "
            "Check your raw files' column names."
        )

    # Prefer title_lc for grouping
    by_col = "title_lc" if "title_lc" in jobs_df.columns else "title"

    # Stratified sample (tune n_per_group if needed)
    tiny_jobs = stratified_sample(jobs_df, by=by_col, n_per_group=10)

    # Inject a couple of edge cases if possible
    if "desc" in tiny_jobs.columns and tiny_jobs.height >= 2:
        long_text = "lorem ipsum " * 500
        # Use a single expression to avoid duplicate 'desc' in the same with_columns call.
        tiny_jobs = tiny_jobs.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit(long_text))
            .when(pl.arange(0, pl.len()) == 1)
            .then(pl.lit(None))  # set NULL on row 1
            .otherwise(pl.col("desc"))
            .alias("desc")
        )

    # Silver-ish: text column
    jobs_text = tiny_jobs.select(
        [
            "job_id",
            "title",
            "company",
            "location",
            pl.col("desc").fill_null("").str.to_lowercase().alias("text"),
        ]
    )

    # Prepare top_skills
    if job_skills_lf is not None and "job_id" in tiny_jobs.columns:
        tiny_ids = set(tiny_jobs["job_id"].drop_nulls().to_list())
        skills_df = job_skills_lf.collect()
        if {"job_id", "skill"}.issubset(skills_df.columns):
            tiny_sk = skills_df.filter(pl.col("job_id").is_in(list(tiny_ids)))
            top_skills = (
                tiny_sk.group_by("skill")
                .agg(pl.len().alias("count"))
                .sort(pl.col("count"), descending=True)
            )
        else:
            # Fallback to exploding skills_list
            top_skills = (
                tiny_jobs.with_columns(pl.col("skills_list").alias("skill"))
                .explode("skill")
                .drop_nulls("skill")
                .group_by("skill")
                .agg(pl.len().alias("count"))
                .sort(pl.col("count"), descending=True)
            )
    else:
        top_skills = (
            tiny_jobs.with_columns(pl.col("skills_list").alias("skill"))
            .explode("skill")
            .drop_nulls("skill")
            .group_by("skill")
            .agg(pl.len().alias("count"))
            .sort(pl.col("count"), descending=True)
        )

    # Write tiny test files
    tiny_jobs.write_parquet(OUT / "tiny_jobs.parquet", compression="zstd")
    jobs_text.write_parquet(OUT / "tiny_jobs_text.parquet", compression="zstd")
    top_skills.write_parquet(OUT / "tiny_top_skills.parquet", compression="zstd")

    print(
        "✅ Wrote:\n"
        f"  - {OUT/'tiny_jobs.parquet'}\n"
        f"  - {OUT/'tiny_jobs_text.parquet'}\n"
        f"  - {OUT/'tiny_top_skills.parquet'}"
    )


if __name__ == "__main__":
    main()
