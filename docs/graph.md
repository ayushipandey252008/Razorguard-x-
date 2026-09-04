# Graph

See **[graph-backend.md](./graph-backend.md)** for NetworkX vs Neo4j, configuration, Docker, and reset.

This is an independent student prototype and is not an official Razorpay product.

## Currently implemented

Nodes (NetworkX): USER, DEVICE, IP, MERCHANT, LOCATION, PAYMENT  
Edges: used_device, used_ip, paid_merchant, located_at, used_payment  

Neo4j (optional) also stores Transaction nodes and MADE / USED_* / AT_* / USES_* relationships. Query methods stay on `GraphBackend`.

`connected_users` walks only USER/DEVICE/IP. Shared merchant or location is **not** an account connection.

Ring detection (`detect_potential_rings`) uses a user–user projection of shared device/IP. Minimum size 3. Language is **potential fraud ring**, not a confirmed label. Cluster ids are stable hashes of the user set.

Transaction graph risk uses:

- extra users on the same device
- extra users on the same IP
- merchant overlap **only** with those shared-infra peers
- a capped component if the user sits in a ≥3 cluster

Evidence (`score_basis`, device/IP user lists, cluster id) is returned with the score. Merchant-only hops do not inflate the score.

In-memory NetworkX is cleared and rebuilt from persisted transactions at process start. Neo4j, when connected, persists across restarts. SQL `graph_entities` / `graph_relationships` remain an upsert write path.

## Baseline

`GraphBackend` protocol in `backend.py` / `interface.py` is the swap point. `factory.py` selects NetworkX or Neo4j.

## Future work (not implemented)

GNNs, distributed production graph ops. Do not claim those from this repo.
