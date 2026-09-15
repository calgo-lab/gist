import numpy as np
import pandas as pd


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def nrmse_iqr(pred, obs):
    q25, q75 = np.quantile(obs, [0.25, 0.75])
    return rmse(pred, obs) / (q75 - q25)


def nse(pred, obs, y_bar=None):
    if y_bar is None:
        y_bar = obs.mean()
    denom = np.sum((obs - y_bar) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((pred - obs) ** 2) / denom)


def rmbe(pred, obs):
    return float(np.mean(pred - obs) / np.std(obs))


def mae(pred, obs):
    return float(np.mean(np.abs(pred - obs)))


def kge(pred, obs):
    corr = np.corrcoef(obs, pred)[0, 1]
    variability = np.std(pred) / np.std(obs)
    bias = pred.mean() / obs.mean()
    return float(1.0 - np.sqrt((corr - 1) ** 2 + (variability - 1) ** 2 + (bias - 1) ** 2))


METRICS = {
    "nRMSE": nrmse_iqr,
    "RMSE": rmse,
    "NSE": nse,
    "rMBE": rmbe,
    "MAE": mae,
    "KGE": kge,
}


def get_metrics(prediction_df, real_col, forecast_col, id_col, metrics_subset=None, train_means=None):
    """Per-well, per-horizon metrics in long format (metric, value, id, horizon).

    train_means: optional {well_id: training-period mean}, used as the NSE baseline.
    """
    names = [m for m in METRICS if metrics_subset is None or m in metrics_subset]
    df = prediction_df if id_col in prediction_df.columns else prediction_df.reset_index()

    rows = []
    for (well, horizon), group in df.groupby([id_col, "horizon"]):
        obs = group[real_col].to_numpy(dtype=float)
        pred = group[forecast_col].to_numpy(dtype=float)
        for name in names:
            if name == "NSE" and train_means is not None:
                value = nse(pred, obs, y_bar=train_means.get(well))
            else:
                value = METRICS[name](pred, obs)
            rows.append({"metric": name, "value": round(value, 3), "id": well, "horizon": horizon})
    return pd.DataFrame(rows)
