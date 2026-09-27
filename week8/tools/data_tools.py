from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd

CRITICAL_COLUMNS = ["title", "type", "release_year"]


def load_csv(file_path: str) -> Tuple[bool, str, List[str], str]:
    """Read a CSV file and return basic schema information."""
    try:
        df = pd.read_csv(file_path)
        shape = f"{df.shape[0]} rows x {df.shape[1]} columns"
        return True, "", list(df.columns), shape
    except Exception as exc:
        return False, str(exc), [], ""


def validate_data(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate dataset quality.

    The dataset is INVALID when:
      - it cannot be read,
      - it is empty,
      - critical columns are missing,
      - critical columns contain null values.
    Missing values in non-critical columns are tolerated (warning only).
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return False, str(exc), {}

    if len(df) == 0:
        return False, "Dataset is empty", {}

    missing_cols = [c for c in CRITICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"Missing critical columns: {missing_cols}", {}

    nulls = df[CRITICAL_COLUMNS].isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad) > 0:
        return False, f"Null values in critical columns: {bad.to_dict()}", {}

    details = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "total_nulls": int(df.isnull().sum().sum()),
    }
    return True, "", details


def clean_data(file_path: str, output_dir: str = "output") -> Tuple[bool, str, str, int]:
    """
    Self-healing repair step used by the retry node.

    Actions:
      1. Drop rows with nulls in critical columns.
      2. Fill remaining categorical nulls with 'Unknown'.
      3. Save the repaired dataset to output/cleaned_data.csv.

    Returns: (success, error, cleaned_path, rows_removed)
    """
    try:
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        cleaned_path = str(out / "cleaned_data.csv")

        df = pd.read_csv(file_path)
        original_rows = len(df)

        df = df.dropna(subset=[c for c in CRITICAL_COLUMNS if c in df.columns])
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("Unknown")

        df.to_csv(cleaned_path, index=False)
        return True, "", cleaned_path, original_rows - len(df)
    except Exception as exc:
        return False, str(exc), file_path, 0


def compute_statistics(file_path: str) -> Dict[str, Any]:
    """Compute Netflix-oriented statistics: distributions, tops, and trends."""
    df = pd.read_csv(file_path)

    stats: Dict[str, Any] = {"total_titles": int(len(df))}

    if "type" in df.columns:
        stats["type_distribution"] = {
            k: int(v) for k, v in df["type"].value_counts().items()
        }

    if "country" in df.columns:
        countries = (
            df["country"].dropna().str.split(",").explode().str.strip()
        )
        stats["top_countries"] = {
            k: int(v) for k, v in countries.value_counts().head(10).items()
        }

    if "listed_in" in df.columns:
        genres = (
            df["listed_in"].dropna().str.split(",").explode().str.strip()
        )
        stats["top_genres"] = {
            k: int(v) for k, v in genres.value_counts().head(10).items()
        }

    if "rating" in df.columns:
        stats["rating_distribution"] = {
            k: int(v) for k, v in df["rating"].value_counts().head(8).items()
        }

    if "date_added" in df.columns:
        years = pd.to_datetime(df["date_added"], errors="coerce").dt.year
        yearly = years.value_counts().sort_index()
        stats["yearly_growth"] = {
            int(k): int(v) for k, v in yearly.items() if pd.notna(k)
        }

    if "director" in df.columns:
        directors = (
            df["director"].dropna().str.split(",").explode().str.strip()
        )
        stats["top_directors"] = {
            k: int(v) for k, v in directors.value_counts().head(10).items()
        }

    return stats


def create_charts(file_path: str, output_dir: str = "output") -> List[str]:
    """Generate and save PNG charts. Returns the list of saved file paths."""
    df = pd.read_csv(file_path)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    chart_paths: List[str] = []

    if "type" in df.columns:
        plt.figure(figsize=(7, 6))
        df["type"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", colors=["#3B82F6", "#EC4899"]
        )
        plt.title("Movies vs TV Shows")
        plt.ylabel("")
        plt.tight_layout()
        path = out / "type_distribution.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        chart_paths.append(str(path))

    if "country" in df.columns:
        countries = (
            df["country"].dropna().str.split(",").explode().str.strip()
        )
        plt.figure(figsize=(10, 5))
        countries.value_counts().head(10).plot(kind="bar", color="#10B981")
        plt.title("Top 10 Countries by Title Count")
        plt.ylabel("Titles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = out / "top_countries.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        chart_paths.append(str(path))

    if "date_added" in df.columns:
        years = pd.to_datetime(df["date_added"], errors="coerce").dt.year
        yearly = years.value_counts().sort_index()
        yearly = yearly[yearly.index.notna()]
        plt.figure(figsize=(10, 5))
        yearly.plot(kind="line", marker="o", color="#F59E0B", linewidth=2)
        plt.title("Titles Added to Netflix per Year")
        plt.ylabel("Titles added")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = out / "yearly_growth.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        chart_paths.append(str(path))

    if "listed_in" in df.columns:
        genres = (
            df["listed_in"].dropna().str.split(",").explode().str.strip()
        )
        plt.figure(figsize=(9, 5))
        genres.value_counts().head(10).plot(kind="barh", color="#8B5CF6")
        plt.title("Top 10 Most Common Genres")
        plt.xlabel("Titles")
        plt.tight_layout()
        path = out / "top_genres.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        chart_paths.append(str(path))

    return chart_paths