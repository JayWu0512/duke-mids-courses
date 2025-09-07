from pathlib import Path

# Base folders (relative to project root). Adjust if you run from elsewhere.
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# File names for standardized outputs
BRONZE_PATH = BRONZE_DIR / "jobs.parquet"
SILVER_PATH = SILVER_DIR / "jobs_text.parquet"
TOP_SKILLS_PATH = GOLD_DIR / "top_skills.parquet"

# Role filters for this project
TARGET_ROLES = [
    "data scientist",
    "data analyst",
    "data engineer",
    "software engineer",
]
