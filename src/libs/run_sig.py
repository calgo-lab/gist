def _sanitize_model_prefix(run_sig, model):
    if not model:
        return run_sig
    prefix = f"{model}_"
    if run_sig.startswith(prefix):
        return run_sig[len(prefix):]
    return run_sig


def resolve_tft_run_sig(tft_cfg, model="TFT"):

    raw = str(tft_cfg.get("run_sig", "")).strip()
    if raw and raw.lower() != "auto":
        return _sanitize_model_prefix(raw, model)

    data_cfg = tft_cfg.get("data", {}) if isinstance(tft_cfg.get("data", {}), dict) else {}
    training_cfg = tft_cfg.get("training", {}) if isinstance(tft_cfg.get("training", {}), dict) else {}
    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}

    dataset = tft_cfg.get("dataset", "sample")
    in_len = int(data_cfg.get("in_len", 52))
    out_len = int(data_cfg.get("out_len", 16))
    statics = int(bool(data_cfg.get("statics", True)))
    epochs = int(training_cfg.get("epochs", 50))
    batch_size = int(training_cfg.get("batch_size", 4096))
    seed = int(training_cfg.get("seed", 40))
    spf = str(float(spatial_cfg.get("train_fraction", 0.8))).replace(".", "p")
    sc = int(spatial_cfg.get("cluster_count", 20))
    ss = int(spatial_cfg.get("split_seed", 42))

    return (
        f"in{in_len}_out{out_len}_ep{epochs}_bs{batch_size}_stat{statics}_seed{seed}_{dataset}"
        f"_spf{spf}_sc{sc}_ss{ss}"
    )
