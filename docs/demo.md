# Demo script (~10 minutes)

1. `uvicorn` + `next dev` (or `docker compose up`)
2. Login `admin@razorguard.local` / `prototype-pass`
3. Command floor — totals come from `/api/v1/analytics`
4. Live wire — seeded mixed traffic
5. Range → scenario `normal` (mostly APPROVE) then `stolen_account` (elevated scores)
6. Open a flagged transaction — four component scores, rules, SHAP, graph JSON
7. Run agent — show `evidence[].status` including any `unavailable`
8. Entity graph — `dev_farm_01` / `203.0.113.200` after a `fraud_ring` simulation
9. Telemetry — PR-AUC / F1 from `ml/evaluation`
10. Record BLOCK with a written reason — appears in audit + case file

Say explicitly: configurable (not industry) thresholds, synthetic labels vs a separate public-data track, potential rings, calibrated P(fraud) vs risk score, no Razorpay affiliation. `find_fraud_cluster` returns “no suspicious cluster identified” when evidence is insufficient.
