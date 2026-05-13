from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = ROOT / "src" / "scripts" / "joint" / "temporal"
for p in [str(SRC_ROOT), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gru_model import GRUSeq2Seq  # noqa: E402

STATIC_FEATURE_REGEX = (
    "eumohp_(.+)_(.+)_(.*[1])"
    "|shannongeom10kmsha"
    "|entgeom10kment"
    "|unigeom10kmuni"
    "|gwn"
    "|huek250.+_(kf).+"
    "|corine"
    "|^TWI_dgm50_r1000m$"
    "|^gok$"
    "|^parde_seasonality$"
    "|^GW_recharge_r1000m$"
    "|^siwa_verweilzeit_j$"
    "|^gw_gespannt_bin$"
    "|^hydroraum_Entlastungsgebiete$"
    "|^hydroraum_Transitgebiete$"
    "|^hydroraum_Speisungsgebiete$"
)
COV_COLS = ["tas_5km", "hurs_5km", "pr_5km", "tag_sin", "tag_cos"]
TRAIN_CUTOFF = pd.Timestamp("20160101")
VAL_CUTOFF = pd.Timestamp("20200101")


def _load_dataset() -> pd.DataFrame:
    import yaml
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    path = Path(data_cfg["full_merged_path"])
    if not path.is_absolute():
        path = ROOT / path
    df = pq.read_table(path).to_pandas()
    df["datum"] = pd.to_datetime(df["datum"]).astype("datetime64[ns]")
    if "gw_gespannt" in df.columns:
        df["gw_gespannt_bin"] = (df["gw_gespannt"] == "gespannt").astype("float32")
    return df


def _build_windows(df, in_len, out_len, cov_cols, well_stats, well_static,
                   start_after: pd.Timestamp | None = None):
    """Build sliding windows. If start_after is set, only emit windows whose
    forecast end date is strictly after that timestamp (test period only)."""
    x_past_rows, x_future_rows, y_rows, x_static_rows, meta = [], [], [], [], []
    static_size = len(next(iter(well_static.values()))) if well_static else 0
    cutoff = np.datetime64(start_after) if start_after is not None else None

    for gid, g in df.groupby("id"):
        if gid not in well_stats or gid not in well_static:
            continue
        g = g.sort_values("datum")
        vals_y = g["gws"].to_numpy(dtype=np.float32)
        vals_cov = g[cov_cols].to_numpy(dtype=np.float32)
        times = g["datum"].to_numpy()
        mean_y, std_y = well_stats[gid]
        vals_y_norm = (vals_y - mean_y) / std_y
        static_vec = well_static[gid]

        for i in range(in_len, len(g) - out_len + 1):
            end_t = times[i + out_len - 1]
            if cutoff is not None and end_t <= cutoff:
                continue
            x_past_rows.append(
                np.concatenate([vals_y_norm[i - in_len:i].reshape(in_len, 1),
                                vals_cov[i - in_len:i]], axis=1)
            )
            x_future_rows.append(vals_cov[i:i + out_len])
            y_rows.append(vals_y_norm[i:i + out_len])
            x_static_rows.append(static_vec)
            meta.append((gid, end_t))

    x_past = np.stack(x_past_rows).astype(np.float32)
    x_future = np.stack(x_future_rows).astype(np.float32)
    y = np.stack(y_rows).astype(np.float32)
    x_static = np.stack(x_static_rows).astype(np.float32)
    return x_past, x_future, y, x_static, meta


def _run_inference(model, x_past, x_future, x_static, device, batch_size=4096):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_past), batch_size):
            xp = torch.from_numpy(x_past[start:start + batch_size]).to(device)
            xf = torch.from_numpy(x_future[start:start + batch_size]).to(device)
            xs = torch.from_numpy(x_static[start:start + batch_size]).to(device)
            chunks.append(model(xp, xf, xs).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _nrmse_pw(pred, y_norm, meta, well_stats, df_ranges):
    """Median per-well nRMSE (range-normalised)."""
    ids = [m[0] for m in meta]
    pred_flat = pd.DataFrame({"id": ids, "pred": list(pred), "y": list(y_norm)})

    per_well = {}
    for gid, g in pred_flat.groupby("id"):
        mean_y, std_y = well_stats[gid]
        preds_m = np.stack(g["pred"].values) * std_y + mean_y
        ys_m = np.stack(g["y"].values) * std_y + mean_y
        rmse = float(np.sqrt(np.mean((preds_m - ys_m) ** 2)))
        r = df_ranges.get(gid, None)
        if r is None or r < 1e-6:
            continue
        per_well[gid] = rmse / r

    values = list(per_well.values())
    return float(np.median(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="0346")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of permutation draws per feature (results are averaged)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{args.run_id}"
    print(f"Loading model from {run_dir}")

    with (run_dir / "scalers.pkl").open("rb") as f:
        scalers = pickle.load(f)
    cov_scaler = scalers["cov_scaler"]
    well_stats = scalers["well_stats"]
    static_scaler = scalers.get("static_scaler")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — failing so the pod is rescheduled on a GPU node")
    device = torch.device("cuda")
    print(f"Device: {device}")
    checkpoint = torch.load(run_dir / "model.pt", map_location=device)
    model = GRUSeq2Seq(
        past_input_size=checkpoint["past_input_size"],
        future_input_size=checkpoint["future_input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        out_len=checkpoint["out_len"],
        static_input_size=checkpoint.get("static_input_size", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Loading dataset...")
    df = _load_dataset()

    # Create hydroraum one-hots for models trained with add_hydroraum_onehot=True
    if "hydroraum" in df.columns:
        for cat in ["Entlastungsgebiete", "Transitgebiete", "Speisungsgebiete"]:
            df[f"hydroraum_{cat}"] = (df["hydroraum"] == cat).astype("float32")

    static_cols = [c for c in df.columns if re.search(STATIC_FEATURE_REGEX, c)]

    # Align static_cols with the model's actual static_input_size.
    # Older models (e.g. 0346) lack siwa and/or hydroraum; newer ones may include them.
    expected_static = checkpoint.get("static_input_size", 0)
    if expected_static > 0 and len(static_cols) != expected_static:
        possibly_absent = ["siwa_verweilzeit_j"] + [c for c in static_cols if c.startswith("hydroraum_")]
        for feat in possibly_absent:
            if feat in static_cols and len(static_cols) > expected_static:
                static_cols = [c for c in static_cols if c != feat]
        if len(static_cols) != expected_static:
            raise RuntimeError(
                f"static_cols size {len(static_cols)} != model static_input_size {expected_static}"
            )
        print(f"Note: trimmed static_cols to {len(static_cols)} to match model's static_input_size={expected_static}")

    print(f"Static features ({len(static_cols)}): {static_cols}")

    # build per-well raw static vectors (unscaled), then scale
    well_static_raw = {}
    for gid, g in df.groupby("id"):
        if gid not in well_stats:
            continue
        well_static_raw[gid] = g[static_cols].iloc[0].fillna(0.0).to_numpy(dtype=np.float32)

    well_static_scaled = {}
    for gid, vec in well_static_raw.items():
        if static_scaler is not None:
            well_static_scaled[gid] = static_scaler.transform(vec.reshape(1, -1)).squeeze(0)
        else:
            well_static_scaled[gid] = vec

    # per-well GWL range over full observation period (for nRMSE normalisation)
    df_ranges = {}
    for gid, g in df.groupby("id"):
        vals = g["gws"].dropna().to_numpy()
        if len(vals) >= 2:
            df_ranges[gid] = float(vals.max() - vals.min())

    print("Building windows (temporal test period: post-2020)...")
    x_past, x_future, y, x_static, meta = _build_windows(
        df, in_len=52, out_len=16, cov_cols=COV_COLS,
        well_stats=well_stats, well_static=well_static_scaled,
        start_after=VAL_CUTOFF,
    )
    print(f"Test windows: {len(x_past)}, wells: {len(set(m[0] for m in meta))}")

    # scale covariates
    x_past_cov = cov_scaler.transform(
        x_past[:, :, 1:].reshape(-1, len(COV_COLS))
    ).reshape(x_past[:, :, 1:].shape)
    x_future_s = cov_scaler.transform(
        x_future.reshape(-1, len(COV_COLS))
    ).reshape(x_future.shape)
    x_past_s = np.concatenate([x_past[:, :, :1], x_past_cov], axis=2)

    print("Running baseline inference...")
    pred_base = _run_inference(model, x_past_s, x_future_s, x_static, device)
    baseline = _nrmse_pw(pred_base, y, meta, well_stats, df_ranges)
    print(f"Baseline nRMSE_pw: {baseline:.4f}")

    # all-features-zeroed
    x_static_zero = np.zeros_like(x_static)
    pred_zero = _run_inference(model, x_past_s, x_future_s, x_static_zero, device)
    zeroed = _nrmse_pw(pred_zero, y, meta, well_stats, df_ranges)
    print(f"All-zeroed nRMSE_pw: {zeroed:.4f}  (delta={zeroed - baseline:+.4f})")

    results = []
    for fi, feat in enumerate(static_cols):
        deltas = []
        for _ in range(args.n_repeats):
            xs_perm = x_static.copy()
            perm_idx = rng.permutation(len(xs_perm))
            xs_perm[:, fi] = xs_perm[perm_idx, fi]
            pred_perm = _run_inference(model, x_past_s, x_future_s, xs_perm, device)
            nrmse_perm = _nrmse_pw(pred_perm, y, meta, well_stats, df_ranges)
            deltas.append(nrmse_perm - baseline)
        delta = float(np.mean(deltas))
        pct = delta / baseline * 100
        results.append({"feature": feat, "delta_nrmse_pw": delta, "pct_change": pct})
        print(f"  [{fi+1:2d}/{len(static_cols)}] {feat}: delta={delta:+.4f} ({pct:+.1f}%)")

    results.sort(key=lambda r: r["delta_nrmse_pw"], reverse=True)

    out_path = ROOT / "reports" / "feature_importance" / f"gru_perm_importance_{args.run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# GRU Static Feature Permutation Importance (Run {args.run_id})",
        "",
        f"**Model**: GRU_FCOV_{args.run_id}",
        f"**Split**: all 1040 wells, temporal test period (post-2020-01-01)",
        f"**Metric**: median nRMSE_pw (range-normalised per-well RMSE)",
        f"**Baseline nRMSE_pw**: {baseline:.4f}",
        f"**All-zeroed nRMSE_pw**: {zeroed:.4f} (delta={zeroed - baseline:+.4f})",
        f"**n_repeats**: {args.n_repeats} (results averaged over permutation draws)",
        "",
        "| Rank | Feature | Delta nRMSE_pw | % Change |",
        "|------|---------|----------------|----------|",
    ]
    for rank, r in enumerate(results, 1):
        lines.append(
            f"| {rank} | {r['feature']} | {r['delta_nrmse_pw']:+.4f} | {r['pct_change']:+.1f}% |"
        )
    lines += [
        "",
        "## Summary",
        f"- Delta > 0: feature helps (shuffling it hurts performance)",
        f"- Delta < 0: feature hurts (model uses it wrong or adds noise)",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
