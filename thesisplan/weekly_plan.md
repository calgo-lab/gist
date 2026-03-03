weekly plan thesis  WEEK 1 (Feb 23 – Mar 1)
- [x] Clean Repo
- [x] Implement GRU baseline and run GRU-only on S1
- [x] Improve GRU until it matches TFT performance
- [x] Start writing: Thesis draft

WEEK 2 (Mar 2 – Mar 8)

- [x] Continue outlier analysis: get numeric metrics for number of observations of k nearest training set neighbors of each holdout well; get average euclidean distance to k nearest training set neighbors of each holdout well
- [ ] Finalize set of metrics and add overall NSE and per-well-NSE for comparison
- [ ] Freeze and document S1 (current split)
- [ ] Create metric tables for TFT, TFT + Kriging and GRU, all on S1
- [ ] Implement custom GP and run GRU predictions on custom GP
- [ ] Implement joint GRU + GP joint training pipeline

WEEK 3 (Mar 9 – Mar 15)
- [ ] Implement and save S2 (random)
- [ ] Implement and save S3 (blocked, 20 km)
- [ ] Implement and save S4 (buffered blocked), compute median nearest-neighbor distance; set D for S4
- [ ] Run separate and joint pipeline on S2, S3, S4
- [ ] Compare outliers and metrics for all split
- [ ] Create split-sensitivity table across all splits

WEEK 4 (Mar 16 – Mar 22)
- [ ]  Consolidate experiment status across S1–S4 (what is final vs pending)
- [ ]  Analyze failure modes and robustness patterns (tail errors, spatial distance effects, split sensitivity)
- [ ]  Prioritize and execute only the highest-impact fixes
- [ ]  Draft/update Results subsections based on confirmed findings

WEEK 5 (Mar 23 – Mar 29)
- [ ] Clean re-run of final experiments for reproducibility (fixed seeds, final configs)
- [ ] Final benchmark tables across splits (S1–S4)
- [ ] Finalize all plots + captions
- [ ] Draft complete Results section (benchmarks + sensitivity + tail risk)

WEEK 6 (Mar 30 – Apr 5)
- [ ]  Integrate all finalized runs into one benchmark table set
- [ ]  Freeze final metric definitions and outlier definitions
- [ ]  Draft Evaluation chapter end-to-end (setup, hypotheses, results, limitations)
- [ ]  Prepare supervisor-ready intermediate draft + issue tracker

WEEK 7 (Apr 6 – Apr 12)
- [ ] Optional: try one mitigation for catastrophic wells (choose one):
    - [ ] residual kriging refinement OR
    - [ ] DeepKriging pilot (only if minimal)
- [ ] Improve Discussion draft: interpret split sensitivity + tail risk + limitations
- [ ] Refine figures for clarity

WEEK 8 (Apr 13 – Apr 19)
- [ ] Freeze experiments (no new runs after this week)
- [ ] Finalize Methods section text
- [ ] Finalize all tables/figures and ensure rerun scripts exist

WEEK 9 (Apr 20 – Apr 26)
- [ ] Finish Introduction + Related Work
- [ ] Polish Methods + Evaluation Protocol
- [ ] Insert final figures in the document

WEEK 10 (Apr 27 – May 3)
- [ ] Finish Discussion + Limitations + Conclusion
- [ ] Send full draft to supervisors

WEEK 11 (May 4 – May 10)
- [ ] Integrate supervisor feedback
- [ ] Tighten narrative and figure explanations

WEEK 12 (May 11 – May 17)
- [ ] Final formatting, references, proofreading
- [ ] Final read-through and polish
- [ ] Submit
