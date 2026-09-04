# IEEE-CIS data audit

The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.

- Source: `IEEE_CIS_CSV`
- Dataset available: `True`
- Label: OFFLINE PUBLIC DATASET EVALUATION

## Transaction table
- Rows: 590540
- Columns: 394
- Duplicate rows: 0
- Duplicate TransactionIDs: 0
- Memory bytes: 1279380966
- Target: {'positive': 20663, 'negative': 569877, 'other_or_null': 0, 'prevalence': 0.03499000914417313}

## Identity table
- Rows: 144233
- Columns: 41
- Duplicate TransactionIDs: 0

## Join
- Key: `TransactionID` (left join identity onto transaction)
- Rows before (txn): 590540
- Rows after: 590540
- Unmatched identity rows: 0
- Identity coverage: 0.2442391709283029
- One-to-many identity keys: 0

## Joined frame
- Rows: 590540
- Columns: 435
- Numerical columns: 390
- Categorical columns: 43
- Identity columns: 40
- Time-related columns: ['TransactionDT', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']

## Highest missingness (joined)
- `id_24`: 99.1962%
- `id_25`: 99.131%
- `id_07`: 99.1271%
- `id_08`: 99.1271%
- `id_21`: 99.1264%
- `id_26`: 99.1257%
- `id_22`: 99.1247%
- `id_23`: 99.1247%
- `id_27`: 99.1247%
- `dist2`: 93.6284%
- `D7`: 93.4099%
- `id_18`: 92.3607%
- `D13`: 89.5093%
- `D14`: 89.4695%
- `D12`: 89.041%
- `id_03`: 88.7689%
- `id_04`: 88.7689%
- `D6`: 87.6068%
- `id_33`: 87.5895%
- `D8`: 87.3123%

Suspicious or unused columns are listed in the leakage report rather than dropped silently.
