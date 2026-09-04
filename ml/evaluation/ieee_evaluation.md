# IEEE-CIS evaluation

The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.

- Source: `IEEE_CIS_CSV`
- Live active model (unchanged): `xgb-iforest-v1-calibrated`
- IEEE status: OFFLINE CANDIDATE
- Runtime seconds: 132.31985795900255

## Split
{
  "strategy": "chronological",
  "official_evaluation_split": "chronological",
  "random_stratified_used_as_official": false,
  "train": {
    "name": "train",
    "n": 413378,
    "fraud": 14538,
    "legitimate": 398840,
    "prevalence": 0.03516878014795175,
    "time_min": 86400,
    "time_max": 10437996
  },
  "validation": {
    "name": "validation",
    "n": 88581,
    "fraud": 3042,
    "legitimate": 85539,
    "prevalence": 0.03434145019812375,
    "time_min": 10438003,
    "time_max": 13151840
  },
  "test": {
    "name": "test",
    "n": 88581,
    "fraud": 3083,
    "legitimate": 85498,
    "prevalence": 0.03480430340592226,
    "time_min": 13151880,
    "time_max": 15811131
  },
  "constraints": {
    "max_train_lt_min_validation": true,
    "max_validation_lt_min_test": true
  },
  "preprocessing_fit_on": "train_only"
}

## Experiment comparison (frozen chronological test, threshold 0.5)

| Experiment | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_transaction_only | transaction | 0.3333169051250197 | 0.8155603175494288 | 0.11310208778563209 | 0.669477781381771 | 0.19351209450590662 | 0.1893026737467543 |
| B_transaction_card | transaction+card | 0.34992949575387666 | 0.8322174742568519 | 0.12313088425327672 | 0.6490431397988972 | 0.2069928623150926 | 0.16667056539334255 |
| C_transaction_identity | transaction+identity | 0.3655249294630912 | 0.8160713283211668 | 0.12504724707068163 | 0.6438533895556277 | 0.20942132193912538 | 0.16244824440337785 |
| D_transaction_behavioral | transaction+behavioral | 0.31271449125269707 | 0.8214024058257008 | 0.11098055970946379 | 0.6740188128446318 | 0.19058100609895906 | 0.19469461273947927 |
| E_transaction_graph | transaction+graph | 0.3748364293730476 | 0.8651765394401753 | 0.12557179515787126 | 0.7301329873499838 | 0.21428911418915703 | 0.18333762193267678 |
| F_combined | transaction+card+address+identity+email+behavioral+graph | 0.3455053854135874 | 0.869226073365801 | 0.11329628046173579 | 0.7735971456373663 | 0.19764647385431341 | 0.2183208963952373 |
| ablation_no_graph | transaction+card+address+identity+email+behavioral | 0.33380969019759066 | 0.8335708812448335 | 0.11338289962825279 | 0.6727213752838145 | 0.19405847953216374 | 0.18968864768766522 |
