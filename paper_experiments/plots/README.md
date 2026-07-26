# Paper figure generation

Every figure in the paper regenerates from one script in this folder. All
scripts are CPU-only (they read result CSVs/JSON produced by the experiment
pipelines, never rerun compute) and can run on a login node.

## Setup

Any Python with `matplotlib`, `numpy`, and `pandas` (on the cluster, the
`omnilens` conda env works). Everything is CPU-only.

## Regenerate everything

```bash
./run_all.sh          # or: python plot_<name>.py for a single figure
```

All data the scripts read lives in `../experiments/*/data` (repo-relative
paths; the folder is self-contained and can be copied anywhere). Outputs land
in `../figures/` as pdf+png, and each pdf is also copied into the paper tree
(`paper/figures/`) when it exists and is writable — otherwise
the copy is skipped with a warning. `lens_param_explosion` additionally syncs
its png, since the paper embeds it.

## Scripts

| Script | Figures | Data |
| --- | --- | --- |
| `plot_method_diagrams.py` | fig_method_estimator, fig_lens_template, fig_memory_optimizer, fig_memory_readout | schematics + readout_scaling_8b benchmark |
| `plot_lens_param_explosion.py` | lens_param_explosion (Fig. 1) | self-contained formulas |
| `plot_rank_tradeoff.py` | fig_rank_tradeoff (+2panel) | canonical Table-1 numbers |
| `plot_gpt2_rank_ablation.py` | rank_ablation_exact_kl_1col, rank_ablation_layerwise_heatmap | gpt2_preemptable_sweep_1000 evals |
| `plot_figure3_main.py` | figure3_main | GPT-2 estimator-comparison evals |
| `plot_rs_baseline.py` | fig_rs_layerwise | gpt2_seed_study_1000 evals |
| `plot_top_token_agreement.py` | top_token_agreement_heatmap | layerwise_top_token_agreement.csv |
| `plot_pearson_kendall_heatmap.py` | layerwise_pearson_kendall_heatmap | layerwise_pearson_kendall.csv |
| `plot_causal_lift.py` | fig_causal_lift_bars | tweak_factor_analysis |
| `plot_site_selection.py` | fig_site_selection_bars | alllens_injection_summary_parity250.csv |
| `plot_injection_damage.py` | fig_injection_damage_eff, fig_injection_damage_top1 | alllens_injection_summary_parity250.csv |
| `plot_lens_predicts_injection.py` | fig_lens_predicts_injection | tweak_factor_analysis |
| `plot_llama_case_studies.py` | fig_injection_heatmaps_8b/70b, fig_injection_curves, fig_dart_heatmap_8b/70b, fig_dart_layerprofile | tweak_factor_analysis + toxicity_ablation |
| `plot_dart_concentration.py` | fig_dart_concentration_bars | dart_crosslens_summary_parity250.csv |
| `plot_allsites_ablation.py` | fig_allsites_ablation_bars | 05 data (allhooks + GPT-2 evaluation_900step) |
| `plot_toxin.py` | fig_toxin_zero, fig_toxin_soft | toxicity_ablation ablation summaries |
| `plot_injection_detection.py` | fig_injection_detection_bars | trajectory_apps detection_auroc.csv |
| `plot_trajectory_grid.py` | fig_trajectory_grid | trajectory_apps grids |
| `plot_rank_structure.py` | fig_rank_structure | 06 rank structure |
| `plot_case_study_summary.py` | fig_case_study_summary | 03/04/05 combined (reuses case_common loaders) |
| `plot_finegrained.py` | fig_finegrained_bars (not in the paper; not synced) | trajectory_apps |

`case_common.py` holds the shared paths/loaders for the case-study bar
charts; `../style/lens_style.py` is the shared style (fonts, palette,
footprints, `finalize()` save path). All figures carry no top titles and no
bold text; `finalize()` enforces the title rule on save.
