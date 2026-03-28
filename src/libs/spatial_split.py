import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def save_spatial_split(split_df, splits_root, dataset):
    fname = f"spatial_split_{dataset}.csv"
    splits_root.mkdir(parents=True, exist_ok=True)
    out_path = splits_root / fname
    split_df.to_csv(out_path, index=False)
    return out_path


def resolve_split_path(splits_root, dataset, spatial_cfg=None):
    cfg = spatial_cfg if isinstance(spatial_cfg, dict) else {}
    file_cfg = str(cfg.get("file", "")).strip()
    if not file_cfg:
        return Path(splits_root) / f"spatial_split_{dataset}.csv"
    p = Path(file_cfg)
    if p.is_absolute():
        return p
    return (Path(splits_root).parent / p).resolve()


def filter_static_columns(columns, exclude_terms):
    if not exclude_terms:
        return list(columns)
    lowered = [(c, c.lower()) for c in columns]
    keep = []
    for original, low in lowered:
        if any(term in low for term in exclude_terms):
            continue
        keep.append(original)
    return keep


def prepare_static_matrix(gws_bb, static_cols):
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


def spatial_train_subset(gws_bb, static_regex,train_fraction, n_clusters, rng_seed, exclude_terms, save_path):
    
    frac = float(train_fraction)
    
    static_cols = list(gws_bb.filter(regex=static_regex).columns)
    static_cols = filter_static_columns(static_cols, exclude_terms or [])

    static_matrix = prepare_static_matrix(gws_bb, static_cols)

    cluster_count = int(n_clusters)
    scaler = StandardScaler()
    static_values = static_matrix.values
    scaled = scaler.fit_transform(static_values)
    kmeans = KMeans(n_clusters=cluster_count, random_state=int(rng_seed), n_init=10)
    clusters = kmeans.fit_predict(scaled)

    info = pd.DataFrame(
        {'id': static_matrix.index.to_list(), 'cluster': clusters}
    ).sort_values('id').reset_index(drop=True)

    rng = np.random.default_rng(int(rng_seed))
    train_ids = []
    ids_count = 0
    for cluster_label in sorted(info['cluster'].unique()):
        ids = info.loc[info['cluster'] == cluster_label, 'id'].to_numpy()
        ids_count += len(ids)
        if len(ids) == 0:
            continue
        take = int(math.ceil(len(ids) * frac))
        take = min(len(ids), max(1, take))
        chosen = rng.choice(ids, size=take, replace=False)
        train_ids.extend(chosen.tolist())

    train_ids_set = set(list(train_ids))
    info['spatial_split'] = np.where(
        info['id'].isin(train_ids_set), 'spatial_train', 'spatial_holdout'
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        info.to_csv(save_path, index=False)

    filtered = gws_bb[gws_bb['id'].isin(train_ids_set)].copy()
    return filtered, info


def spatial_split_random_max_dist(coords_df, train_fraction=0.9, k=3, d_percentile=25, rng_seed=42, save_path=None):
    coords = coords_df[["id", "x_25833", "y_25833"]].drop_duplicates("id").dropna(
        subset=["x_25833", "y_25833"]
    ).reset_index(drop=True)

    XY = coords[["x_25833", "y_25833"]].to_numpy()
    tree = cKDTree(XY)
    dists, _ = tree.query(XY, k=k + 1)
    mean_dist_k = dists[:, 1:].mean(axis=1)

    d_threshold = np.percentile(mean_dist_k, d_percentile)
    eligible_ids = coords.loc[mean_dist_k <= d_threshold, "id"].to_numpy()

    n_total = len(coords)
    n_holdout = int(round(n_total * (1.0 - train_fraction)))

    if len(eligible_ids) < n_holdout:
        raise ValueError(
            f"Not enough eligible wells ({len(eligible_ids)}) for holdout target "
            f"({n_holdout}). Increase d_percentile."
        )

    rng = np.random.default_rng(rng_seed)
    holdout_ids = set(rng.choice(eligible_ids, size=n_holdout, replace=False).tolist())

    result = coords[["id"]].copy()
    result["cluster"] = 0
    result["spatial_split"] = np.where(
        result["id"].isin(holdout_ids), "spatial_holdout", "spatial_train"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)

    return result


def load_or_create_split(gws_bb, static_regex, train_fraction, n_clusters, rng_seed, exclude_terms, save_path):

    if save_path and Path(save_path).exists():
        info = pd.read_csv(save_path)
        train_ids = set(info.loc[info["spatial_split"] == "spatial_train", "id"])
        filtered = gws_bb[gws_bb["id"].isin(train_ids)].copy()
        return filtered, info

    return spatial_train_subset(
        gws_bb,
        static_regex=static_regex,
        train_fraction=train_fraction,
        n_clusters=n_clusters,
        rng_seed=rng_seed,
        exclude_terms=exclude_terms,
        save_path=save_path,
    )
