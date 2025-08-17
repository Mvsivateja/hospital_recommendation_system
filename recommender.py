
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default weights (tweak these to taste)
WEIGHTS = {
    "text": 0.45,        # specialization/services match
    "rating": 0.20,      # 0..1
    "distance": 0.20,    # 0..1 after decay
    "cost": 0.10,        # 0..1 (lower is better → higher score)
    "bonuses": 0.05,     # 24x7 / insurance
}

def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def load_hospitals(csv_path: str = "data/hospitals.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = [
        "name","city","specializations","services","rating","avg_fee",
        "is_24x7","accepts_insurance","distance_km","address"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # Coerce types
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["avg_fee"] = pd.to_numeric(df["avg_fee"], errors="coerce").fillna(df["avg_fee"].median() if df["avg_fee"].notna().any() else 300)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(5)
    df["is_24x7"] = df["is_24x7"].astype(str).str.lower().isin(["1","true","yes","y"])
    df["accepts_insurance"] = df["accepts_insurance"].astype(str).str.lower().isin(["1","true","yes","y"])
    # Build corpus for TF–IDF
    def row_text(r):
        parts = [
            str(r.get("name","")),
            str(r.get("city","")),
            str(r.get("specializations","")).replace("|", " "),
            str(r.get("services","")).replace("|", " "),
        ]
        return " ".join(parts)
    df["corpus"] = df.apply(row_text, axis=1)
    return df

def build_index(df: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vec.fit_transform(df["corpus"].fillna(""))
    return vec, X

def recommend(
    df: pd.DataFrame,
    vec: TfidfVectorizer,
    X: np.ndarray,
    selected_specs: Optional[List[str]] = None,
    query_text: str = "",
    cities: Optional[List[str]] = None,
    min_rating: float = 0.0,
    max_fee: Optional[float] = None,
    need_24x7: bool = False,
    need_insurance: bool = False,
    max_distance_km: Optional[float] = None,
    top_k: int = 10,
):
    # Filters
    mask = pd.Series([True]*len(df))
    if cities and len(cities) > 0 and "Any" not in cities:
        mask &= df["city"].isin(cities)
    if min_rating:
        mask &= df["rating"] >= float(min_rating)
    if max_fee is not None:
        mask &= df["avg_fee"] <= float(max_fee)
    if need_24x7:
        mask &= df["is_24x7"]
    if need_insurance:
        mask &= df["accepts_insurance"]
    if max_distance_km is not None:
        mask &= df["distance_km"] <= float(max_distance_km)

    df_f = df[mask].copy()
    if df_f.empty:
        return df_f.assign(score=[])

    # Build query string
    specs_text = " ".join(selected_specs or [])
    query = (specs_text + " " + (query_text or "")).strip()
    if query == "":
        # Avoid zero vector; use zeros
        text_score = np.zeros(len(df_f))
    else:
        qv = vec.transform([query])
        sim = cosine_similarity(qv, X[mask.values]).ravel()
        text_score = sim

    # Other components
    rating_score = _normalize_series(df_f["rating"])  # 0..1
    # Distance decay: closer gets higher score
    dist = df_f["distance_km"].astype(float).clip(lower=0.0)
    dist_score = np.exp(-0.15 * dist)  # 1 at 0km, ~0.22 at 10km
    # Cost: lower is better → invert normalized fee
    fee_norm = _normalize_series(df_f["avg_fee"].astype(float))
    cost_score = 1.0 - fee_norm

    # Bonuses (when available)
    bonuses = ((df_f["is_24x7"].astype(int) + df_f["accepts_insurance"].astype(int)) / 2.0)

    # Final score
    score = (
        WEIGHTS["text"] * text_score +
        WEIGHTS["rating"] * rating_score +
        WEIGHTS["distance"] * dist_score +
        WEIGHTS["cost"] * cost_score +
        WEIGHTS["bonuses"] * bonuses
    )

    df_f = df_f.copy()
    df_f["match_score"] = text_score
    df_f["score"] = score
    df_f = df_f.sort_values("score", ascending=False).head(top_k)

    # Simple reason text
    def reason(r):
        bits = []
        if r["match_score"] > 0.05:
            bits.append("matches your specialty/services")
        if r["rating"] >= 4.0:
            bits.append("high rating")
        if r["distance_km"] <= 5:
            bits.append("nearby")
        if r["is_24x7"]:
            bits.append("24x7")
        if r["accepts_insurance"]:
            bits.append("insurance accepted")
        return ", ".join(bits) or "good overall fit"
    df_f["why"] = df_f.apply(reason, axis=1)

    # Clean columns for display
    show_cols = [
        "name","city","address","specializations","services","rating",
        "avg_fee","is_24x7","accepts_insurance","distance_km","why","score"
    ]
    return df_f[show_cols]
