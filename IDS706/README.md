[![Python Template for IDS706](https://github.com/JayWu0512/duke-mids-courses/actions/workflows/ids706-ci.yml/badge.svg)](https://github.com/JayWu0512/duke-mids-courses/actions/workflows/ids706-ci.yml)

# LinkedIn Jobs & Skills Analysis

This project analyzes LinkedIn job postings and skills data using a structured data pipeline (Polars) and machine learning exploration (KMeans clustering).

## Project Structure

```
data/
├── raw/                 # Original Kaggle parquet files
│   ├── job_skills.parquet
│   ├── job_summary.parquet
│   └── linkedin_job_postings.parquet
├── bronze/              # Cleaned + normalized
│   └── jobs.parquet
├── silver/              # Role-filtered + text-joined
│   └── jobs_text.parquet
└── gold/                # Aggregated skills & final outputs
    └── top_skills.parquet

notebooks/
├── 01_eda.ipynb         # Data inspection & visualization
└── 02_kmeans.ipynb      # ML exploration (TF-IDF + clustering)

scripts/
└── download_kaggle.py   # Kaggle download & parquet conversion

src/
├── app/                 # Pipeline orchestration (Typer CLI)
│   ├── cli.py
│   └── pipeline.py
├── infra/               # IO adapters, transformers, aggregators
│   ├── aggregators.py
│   ├── io_polars.py
│   └── transformers.py
├── domain/              # Ports/abstractions
│   └── ports.py
└── utils/               # Helpers & config
    └── settings.py

Makefile                 # Common commands (build/install)
requirements.txt         # Dependencies
README.md                # Project documentation
```


## Data Source
- Job postings dataset (CSV/Parquet) with fields including **title, company, location, work_type, seniority, and listed skills**.
- Dataset is filtered to a manageable slice for computation in the notebooks.


## Pipeline

The pipeline builds multiple layers of data:

1. **Raw** → Original Kaggle parquet files.
2. **Bronze** → Cleaned and normalized schema.
3. **Silver** → Role-filtered, text-joined job postings.
4. **Gold** → Aggregated top skills.

Run with:

```bash
make build
```

Outputs will be written into `data/bronze/`, `data/silver/`, and `data/gold/`.

## Notebooks

- **01_eda.ipynb**:  
  Data inspection, null analysis, column distributions, posting trends.  
  Example: work type breakdown (onsite, hybrid, remote).

- **02_kmeans.ipynb**:  
  Machine learning exploration with TF-IDF + KMeans clustering.  
  Includes elbow method to choose K and visualization of top TF-IDF terms per cluster.

## Analysis Steps

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- **Health checks** on missing values and duplicates.  
- **Distributions** of roles, work types, and seniority levels.  
- **Top locations & companies** by posting frequency.  
- **Time series analysis**: posting trend by month.  
- **Skills analysis**:  
  - Top skills overall  
  - Top skills by role  
  - Co-occurrence network of skills  

### 2. KMeans Clustering (`02_kmeans.ipynb`)
- Feature engineering: transforming skills & job features into vectors.  
- Running **KMeans clustering** to group similar jobs.  
- Evaluating clusters using **silhouette score**.  
- Visualizing clusters to interpret relationships among job postings.

## Visualization
- Bar charts for top skills, roles, and companies.  
- Line charts for posting trends over time.  
- Scatter plots of clusters in reduced dimensions.  
- Skill co-occurrence diagrams.

All plots include **axis labels and titles** for clarity.

## Insights

- Many job postings are missing work type and seniority information → imputation and text-based derivation improves coverage.
- Distribution: majority of postings are "onsite", but remote/hybrid roles exist in smaller proportions.
- KMeans clustering (TF-IDF on job text) reveals meaningful groupings:  
  - **Cluster 0**: software / embedded software roles.  
  - Other clusters capture data-focused roles (data scientist, analyst, engineer).  
- The elbow method suggested **K ≈ 6** as a reasonable trade-off for cluster coherence.

## Requirements

Install dependencies:

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

Key packages: `polars`, `typer`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow`.

---
