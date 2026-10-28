# Surge & Fleet Playbook

## Surge thresholds
- Demand 20-40% above SARIMA expected + headroom to 95% bound: no surge. Log observation.
- 40-80% above expected: apply multiplier 1.3x in the affected cluster only.
- > 80% above expected (CI breach): multiplier 2.0x capped after 15 minutes if demand normalizes.

## Vehicle tier allocation
- High surge windows: push Premium (Black/XL) share to 2:1 premium-to-standard.
- Excess demand beyond premium capacity: rebalance standard tier from adjacent low-demand zones.
- Airport cluster (Cluster 0): hold 20% reserve fleet at all times; long-haul trips dominate revenue (avg fare INR 1,507, avg distance 24.9 km).

## Fleet deployment rules
- Deploy standby drivers at ~40% of the observed trip shortfall.
- Never deploy more than 110% of shortfall in one shot (oversupply crashes per-driver earnings).
- After 30 minutes, re-evaluate: if observed demand stays > 85% of pre-brief level, release 25% of deployed standby back.

## Price-quote fairness
Every quoted fare must stay within the RandomFareModel band (R2 = 0.972 fare predictor).
Quotes outside 1.15x of the expected fare require manager approval.
