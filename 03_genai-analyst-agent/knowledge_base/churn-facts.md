# Churn Facts (verified from Telco EDA)

- Overall churn: 26.6% of 7,032 active customers.
- Month-to-month contracts churn at 42.7%; two-year contracts near zero. Chi-square = 1179.5.
- Month-to-month + fiber-optic customers = 72% of total revenue at risk.
- Churn by charge quartile is NON-monotonic (11.3% -> 24.6% -> 37.6% -> 32.9%): the most
  expensive quartile churns LESS, so price alone is not the driver.
- Paperless billing: independent churn driver (+8.3 pp average lift), survives a
  contract x tenure x payment-method triple control (worse in 31 of 38 cells).
- Senior citizens churn at 41.7% vs 23.7% baseline — material, not negligible.
- Fiber-optic churn effect is largely confounded by contract mix; always control for
  contract before blaming the service itself.
- Fiber + electronic check = the highest-risk payment/service combo (45.3% churn).
