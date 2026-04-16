# Data Access

Heavyweight datasets are not stored in this repository.

To run the experiment:

1. Prepare the original dataset directory `data`.
2. Prepare the masks dataset directory `data_masks`.
3. Copy `configs/datasets/local_paths.template.json` to `configs/datasets/local_paths.json`.
4. Put absolute paths to these two directories into `local_paths.json`.

After that, run:

```bash
make all
```

or:

```bash
./.venv/bin/python scripts/run_all_hypotheses.py
```
