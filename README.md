# Customer Credit Risk Analysis

**Dataset:** UCI Statlog German Credit Data (Hofmann, 1994)  
**Records:** 1,000 customers · 14 features  
**Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

## What this project covers

A German bank wants to understand which customers are likely to default on credit. This notebook works through the full analytical pipeline from raw data to a production-ready risk model.

- **Exploratory analysis** — default rates by purpose, employment, age, and account status
- **8-factor credit scoring model** — 0–100 scale, risk-tiered output (Low / Medium / High / Very High)
- **ML classification** — Logistic Regression, Random Forest, and Gradient Boosting compared on AUC
- **Executive dashboard** — portfolio KPIs, risk exposure by purpose, credit at risk by tier

## Results

| Model | AUC |
|-------|-----|
| Logistic Regression | ~0.72 |
| Random Forest | ~0.76 |
| Gradient Boosting | ~0.78 |

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook credit_risk_analysis.ipynb
```

## Files

```
credit_risk/
├── credit_risk_analysis.ipynb   # Main notebook
├── german_credit_data.csv       # Dataset (UCI-faithful structure)
└── README.md
```

## Key findings

- Customers with negative checking accounts have a 2× higher default rate
- Education and repair loans carry disproportionate default risk
- The scoring model achieves ~0.78 AUC — meaningfully better than random selection
- Employment tenure under 1 year is a strong standalone red flag
