# Session notes — 2026-11-27

## 1. Graph features into LightGBM (`--lgbm-graph-features`)
Leakage-safe entity-aggregate features (past-only counts, amount stats, recency, and
fraud-rate target encodings built from training-period labels only) for pseudo-user,
card1, address, payer/recipient email and device. Test AUC / PR-AUC, IEEE-CIS,
chronological 70/15/15 split:

| LightGBM variant | size | seeds | test AUC | test PR-AUC |
|---|---|---|---|---|
| plain, no V | 300k | 3 | 0.9032±0.0010 | 0.5565±0.0050 |
| + graph features, no V | 300k | 3 | 0.9206±0.0001 | 0.6288±0.0042 |
| plain, no V | full | 1 | 0.8947 | 0.5107 |
| + graph features, no V | full | 1 | 0.9203 | 0.5843 |

Ablation: nearly all of the gain comes from the label-based fraud-rate columns;
deployable lift depends on how fast fraud labels are confirmed.

## 2. Black Friday 2026 tech deals (checked Nov 27, USD, US retailers)
- Lenovo LOQ 15.3" (Ryzen 7 250, RTX 5060 115W, 16GB, 512GB): $1,149.99 at Best Buy.
- Sony WH-1000XM5: $169.99 at Best Buy (was $398).
- GitHub Copilot: 30% off Copilot Fusion model costs for paid plans through Dec 31, 2026;
  plan prices unchanged (Pro $10, Pro+ $39, Max $100 per month).

## 3. AI tools pricing check
No Black Friday plan-price cuts found on ChatGPT Plus, Claude Pro/Max, or Cursor.

## 4. Tariko search
No Tariko repository exists under Kingpin139 (public or private). The only unrelated
public match is Tarikrac/Tariko. No new repo was created.
