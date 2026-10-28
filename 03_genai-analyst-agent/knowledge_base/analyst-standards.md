# Analyst Documentation Standards

## Evidence discipline
1. Every claim in an analyst brief must cite either (a) a queried dataset result, (b) a
   retrieved policy/knowledge-base passage, or (c) a computed number.
2. If evidence is missing, state "insufficient evidence" and list what query would close the gap.
   Never invent numbers.

## Numbers format
- Percentages: one decimal (e.g., 26.6%).
- Money: currency symbol + no decimals for yearly, 2 decimals for per-unit (e.g., $4,557/month).
- Use thousands separators everywhere.

## Brief structure (Markdown)
## Situation
## Key numbers
## Recommendation
## Risk & counter-evidence
## Next check

## Escalation triggers (escalate to human, don't auto-act)
- Complaints involving billing disputes or legal threats
- Any recommendation costing more than $25/month per customer
- Data anomalies: duplicate customer IDs, negative charges, impossible tenure (>120 months)
