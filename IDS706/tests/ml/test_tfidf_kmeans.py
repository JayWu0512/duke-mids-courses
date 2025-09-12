from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
from pathlib import Path
import polars as pl

sklearn = pytest.importorskip("sklearn")

TEST_DIR = Path("data/test")


def test_tfidf_kmeans_runs():
    df = pl.read_parquet(TEST_DIR / "tiny_jobs_text.parquet")
    texts = df["text"].to_list()

    X = TfidfVectorizer(min_df=1).fit_transform(texts)
    k = 3 if X.shape[0] >= 3 else max(1, X.shape[0])
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(X)

    # Correct length and reasonable number of unique clusters (can be < k)
    assert len(labels) == X.shape[0]
    unique_labels = len(set(labels.tolist()))
    assert 1 <= unique_labels <= k
