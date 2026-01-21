import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
        if take == len(ids):
            chosen = ids
        else:
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
