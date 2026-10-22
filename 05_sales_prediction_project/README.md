# Sales Prediction

> Regression model that forecasts product sales from outlet and item attributes.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

**Result: RÂ² = 0.91 Â· MSE = 2.86**

*Completed as Task 1 of the Oasis Infobyte Data Science Internship (OIBSIP).*

---

## Problem

Sales forecasting drives inventory planning, and both directions of error are expensive:

- **Under-forecast** â†’ stockouts, lost revenue, and customers who go elsewhere
- **Over-forecast** â†’ capital tied up in unsold inventory, plus storage and spoilage costs

The task is to predict sales from item and outlet characteristics: MRP, outlet size, item type, and outlet location.

---

## Approach

```
Raw sales data
  â†“ Load and inspect
  â†“ Clean â€” handle missing values and duplicates
  â†“ Exploratory data analysis â€” distributions, correlations
  â†“ Encode categorical variables (outlet type, item category, location)
  â†“ Train Linear Regression
  â†“ Evaluate â€” MSE, MAE, RMSE, RÂ²
```

### Why Linear Regression suffices here

The obvious question is whether a more powerful model would do better. The results argue no: **an RÂ² of 0.91 from a linear model tells you the underlying relationship is already largely linear.**

MRP relates to sales in a roughly proportional way, and the outlet attributes shift that relationship up or down without introducing complex interactions. A Random Forest could capture non-linearities â€” but where few exist, it adds computational cost and loses interpretability for marginal gain.

This is worth stating explicitly because it contrasts with the [car price prediction](../06_car_price_prediction_project/) project in this repository, where depreciation *is* strongly non-linear and a Random Forest is genuinely the better choice. **Matching model complexity to the structure of the problem is the actual skill** â€” not reaching for the most powerful algorithm by default.

---

## Results

| Metric | Value |
|--------|-------|
| **RÂ² Score** | **0.91** |
| Mean Squared Error | 2.86 |

### Interpretation

**RÂ² of 0.91 means the model explains 91% of the variance in sales.** The remaining 9% comes from factors not present in the dataset â€” seasonality, local promotions, competitor activity, weather.

For retail forecasting this is a strong result. It suggests the features captured here (price, outlet characteristics, product category) account for most of what drives sales volume, and that the linear relationship assumption holds well.

---

## Running It

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook sales_prediction.ipynb
```

> **Dataset not included.** The notebook expects `sales dataset.csv` in the same folder. Supply your own copy or update the `read_csv()` path in the first cell. Saved notebook outputs preserve the results above.

---

## Next Steps

- **Residual analysis** â€” plot residuals against predictions to confirm the linearity assumption holds across the full range rather than just on average
- **Cross-validation** â€” a single train/test split can be optimistic; k-fold would give a more reliable RÂ² estimate
- **Feature importance** â€” inspect coefficients to identify which outlet attributes matter most for planning decisions
- **Add temporal features** â€” seasonality and promotional calendars likely explain a meaningful share of the residual 9%

---

[â¬… Back to all ML projects](../README.md)
