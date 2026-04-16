# Scripts Overview

This directory contains two kinds of scripts.

## Public entry points

These scripts are intended to be launched directly by a researcher:

- `run_foundation.py` - prepares the shared experiment base (`A0-A5`).
- `run_h1_interpretable_features.py` - checks hypothesis H1.
- `run_h2_mask_effect.py` - checks hypothesis H2.
- `run_h3_noise_regimes.py` - checks hypothesis H3.
- `run_h4_aggregation.py` - checks hypothesis H4.
- `run_h5_deep_vs_handcrafted.py` - checks hypothesis H5.
- `run_all_hypotheses.py` - runs the full reproducible experiment end to end.

## Internal orchestration and stage helpers

These files are used by the public entry points and are kept in the repository
because they contain reusable implementation details:

- `reproducibility_blocks.py` - high-level orchestration shared by public runners.
- `run_a1_to_a4.py`, `run_a3_to_a5.py`, `run_a6_to_a11.py`, `run_a9_deep_baselines.py` - lower-level stage runners for the original experiment plan.
- `build_modeling_workset.py` - prepares the modeling subset used by H5.
- `render_markdown_report_html.py` - utility for rendering Markdown reports into HTML.

If you only want to reproduce the experiment, start with the public entry points
or the `Makefile` targets documented in `README.md` and `docs/reproducibility_guide.md`.
