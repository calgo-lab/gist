Plan until early April:

1. Create 4 different spatial splits: <br>
    a. ***Static features cluster-based distributed*** <br>
    b. Random split <br>
    c. Randomly assign blocks from grid to train/holdout (harder) <br>
    d. Spatially stratified split with proximity constraint (easier) <br>
2. ***Create a strong GRU separate baseline***
3. Implement a custom GP head
4. Combine both in one joint-training pipeline
5. Finalize evaluation metrics (find good baselines)
6. Lock one evaluation setup (same splits, horizons, metrics, seeds)
7. Compare separate vs joint on the same GRU-based architecture for all 4 splits
8. Potentially improve kriging part