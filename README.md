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

## Visualizations

The analysis includes the following visualizations (stored in the `images/` folder):

- **01_exploratory_overview.png** — Exploratory data analysis with distributions and summary statistics
- **02_credit_score_analysis.png** — Credit score model performance and risk tier breakdown
- **03_model_evaluation.png** — ML model comparison and performance metrics
- **04_executive_dashboard.png** — Executive summary dashboard with portfolio KPIs and risk exposure

![Exploratory Overview](images/01_exploratory_overview.png)
![Credit Score Analysis](images/02_credit_score_analysis.png)
![Model Evaluation](images/03_model_evaluation.png)
![Executive Dashboard](images/04_executive_dashboard.png)

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook credit_risk_analysis.ipynb
```

## Files

```
customer-credit-risk-analysis/
├── credit_risk_analysis.ipynb      # Main notebook
├── german_credit_data.csv          # Dataset (UCI-faithful structure)
├── images/                         # Visualizations
│   ├── 01_exploratory_overview.png
│   ├── 02_credit_score_analysis.png
│   ├── 03_model_evaluation.png
│   └── 04_executive_dashboard.png
└── README.md
```

## Key findings

- Customers with negative checking accounts have a 2× higher default rate
- Education and repair loans carry disproportionate default risk
- The scoring model achieves ~0.78 AUC — meaningfully better than random selection
- Employment tenure under 1 year is a strong standalone red flag
