# IEEE-CIS local data (not committed)

Place the IEEE-CIS Fraud Detection training CSVs here. This prototype **does not download** them.

Expected files (gitignored):

- `train_transaction.csv`
- `train_identity.csv`

Source: IEEE-CIS Fraud Detection (Kaggle / Vesta). Obtain the files from the official competition/dataset listing yourself. Do not invent or rely on unofficial mirrors.

Set `IEEE_DATA_DIR` to this directory (default) or another local path.

Raw CSVs are intentionally excluded from Git. The IEEE-CIS experiment is an offline public-dataset evaluation and is **not** live scoring or production payment-fraud performance.

See `docs/ieee-cis-evaluation.md`.
