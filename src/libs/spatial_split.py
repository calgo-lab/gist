import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _filter_static_columns(columns: Iterable[str], exclude_terms: Iterable[str]) -> list[str]:
    """Drop columns whose names match any forbidden pattern."""
    if not exclude_terms:
        return list(columns)
    lowered = [(c, c.lower()) for c in columns]
    keep = []
    for original, low in lowered:
        if any(term in low for term in exclude_terms):
            continue
        keep.append(original)
    return keep


def _prepare_static_matrix(gws_bb: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    """Return a per-well static feature frame suitable for clustering."""
    subset = gws_bb[['id'] + static_cols].copy()
    subset = (
        subset.groupby('id', as_index=False)
        .first()
        .set_index('id')
        .sort_index()
    )
    subset = subset.apply(pd.to_numeric, errors='coerce')
    subset = subset.replace([np.inf, -np.inf], np.nan)
    med = subset.median(numeric_only=True)
    subset = subset.fillna(med).fillna(0.0)
    return subset


def spatial_train_subset(
    gws_bb: pd.DataFrame,
    static_regex: str,
    train_fraction: float,
    n_clusters: int,
    rng_seed: int,
    *,
    exclude_terms: Iterable[str] | None = None,
    save_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster wells by static attributes and sample a spatial train subset.

    Returns a filtered groundwater frame containing only the selected wells and a
    metadata frame with cluster assignments and the spatial split label.
    """
    frac = max(0.0, min(1.0, float(train_fraction)))
    if frac == 0.0:
        raise ValueError("train_fraction must be > 0 to build a spatial train set.")

    static_cols = list(gws_bb.filter(regex=static_regex).columns)
    static_cols = _filter_static_columns(static_cols, exclude_terms or [])
    if not static_cols:
        raise ValueError("No static columns available for spatial clustering.")

    static_matrix = _prepare_static_matrix(gws_bb, static_cols)

    cluster_count = max(1, min(int(n_clusters), len(static_matrix)))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(static_matrix.values)
    kmeans = KMeans(n_clusters=cluster_count, random_state=int(rng_seed), n_init=10)
    clusters = kmeans.fit_predict(scaled)

    info = pd.DataFrame(
        {'id': static_matrix.index.to_list(), 'cluster': clusters}
    ).sort_values('id').reset_index(drop=True)

    rng = np.random.default_rng(int(rng_seed))
    train_ids = []
    for cluster_label in sorted(info['cluster'].unique()):
        ids = info.loc[info['cluster'] == cluster_label, 'id'].to_numpy()
        if len(ids) == 0:
            continue
        take = int(math.ceil(len(ids) * frac))
        take = min(len(ids), max(1, take))
        if take == len(ids):
            chosen = ids
        else:
            chosen = rng.choice(ids, size=take, replace=False)
        train_ids.extend(chosen.tolist())

    train_ids_set = set(train_ids)
    info['spatial_split'] = np.where(
        info['id'].isin(train_ids_set), 'spatial_train', 'spatial_holdout'
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        info.to_csv(save_path, index=False)

    filtered = gws_bb[gws_bb['id'].isin(train_ids_set)].copy()
    return filtered, info
