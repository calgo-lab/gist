Plan until late March/early April:

1. Clean Repo
2. Try out until Feb 27: Create a strong (or as strong as possible) GRU separate baseline, then implement a differentiable spatial head and combine both in one joint-training pipeline
3. Lock one evaluation setup (same splits, horizons, metrics, seed policy), use it for everything (so far more trial and error); especially coming up with the right metrics and splits is non-trivial
4. Only if 2 worked: Compare separate vs joint on the same GRU-based architecture; TFT+kriging stay reference benchmark
5. If enough time, swap GRU with a custom TFT-like temporal module in the same pipeline, then optionally test transfer to other datasets


Planned contributions:

1. A modular end-to-end spatiotemporal pipeline that supports joint backprop training
2. Fair separate-vs-joint comparison on the same controllable architecture (starting with GRU-based temporal module)
3. Robustness analysis across different splits/split hardness levels
4. A defined set of robust metrics appropriate for this use case
5. Practical guide on when joint training is worth it versus separate baselines
6. Potentially: Uncertainty evaluation (calibration/coverage) in addition to NSE/RMSE/MAE
7. Potentially: Proof of transferability to other datasets in the same or different domains
