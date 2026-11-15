# Evaluating Machine Learning for Next-Day Direction Prediction in the Pakistan Stock Exchange

## Overview

Can machine learning predict next-trading-day stock direction in the Pakistan Stock Exchange (PSX), and can it outperform simple prediction methods on unseen historical data?

This project investigates that question using historical daily stock data and machine-learning classification.

The task is binary classification:
- `1`: The next trading session's closing price is higher than today's.
- `0`: The next session's closing price is lower than or equal to today's.

The prediction is made after the current trading session, using only information available at that point.

The objective is to conduct a fair experiment, not to build a trading system or claim that financial markets can be reliably predicted. A model that fails to outperform simple baselines is still a valid experimental finding.

## Technologies

- Python
- pandas and NumPy
- matplotlib
- scikit-learn
- yfinance
- Jupyter Notebook
- Git and GitHub

## Dataset and Preparation

**Stock:** Lucky Cement (LUCK.KA)  
**Source:** Yahoo Finance through `yfinance`  
**Period:** 1 January 2015 – 13 October 2026  
**Raw observations:** 3,053  
**Observations after cleaning:** 2,885  
**Labelled observations:** 2,884

### Data cleaning

A significant data issue was identified involving Lucky Cement's 5-for-1 stock split in April 2025.

Yahoo Finance contained inconsistent price adjustments, including an apparent price drop of approximately 80% on 10 April 2025 and three February observations on the post-split price scale.

The data was corrected by:
- Dividing pre-10 April 2025 prices by five.
- Multiplying corresponding historical volume by five.
- Retaining the three February observations already on the adjusted scale.
- Removing 168 zero-volume observations.

Additional checks covered missing values, duplicate dates, price consistency and unusually large daily movements.

Data files are not committed to the repository. Run Notebook 01 to download and prepare the dataset.

## Methodology

### Features

Six technical features were initially investigated, calculated using historical information only:

- Previous-day return (`ret_1`)
- Five-day return (`ret_5`)
- Ten-day return volatility (`vol_10`)
- Volume change (`vol_chg`)
- High-low price range (`hl_range`)
- Distance from the 20-day moving average (`ma_dist`)

### Baselines

Two simple prediction methods were used:

1. **Majority-class baseline:** Always predicts the most common class in the development dataset.
2. **Previous-direction baseline:** Predicts the same direction as the previous trading session.

### Machine-learning model

Logistic Regression was selected because it is suitable for binary classification, produces probabilities and provides a relatively interpretable starting point.

Features were standardised using `StandardScaler`.

The experiment compared two feature sets (all six features and three core features) and regularisation values of `C = 0.001, 0.01, 0.1, 1, 10`.

Hyperparameters and feature selection were evaluated using five-fold expanding-window time-series cross-validation on development data, with log-loss as the selection metric.

### Evaluation protocol

- Data was sorted chronologically.
- The first 75% was used for development and the final 25% for testing.
- One boundary observation was removed between development and test sets.
- The test set was not used for model selection.
- Accuracy, balanced accuracy, ROC-AUC, log-loss and precision/recall were examined.
- Paired bootstrap resampling with 5,000 samples was used to estimate confidence intervals for accuracy differences.

The primary hypothesis required Logistic Regression to outperform both baselines, with the 95% confidence interval of each accuracy difference entirely above zero.

## Results and Findings

### LUCK.KA baseline experiment

The selected configuration was Logistic Regression using the three-core-feature set (`core3`) with `C = 0.001`.

**Test observations: 721**

| Model | Test accuracy |
|---|---:|
| Majority-class baseline | 49.38% |
| Previous-direction baseline | 53.54% |
| Logistic Regression | 49.51% |

Logistic Regression achieved approximately 4.02 percentage points lower accuracy than the previous-direction baseline.

### Statistical comparison

The paired bootstrap comparison produced the following 95% confidence intervals:

| Comparison | Accuracy difference | 95% CI |
|---|---:|---:|
| Logistic Regression − Majority | +0.14 pp | [-0.97, +1.25] pp |
| Logistic Regression − Previous direction | -4.02 pp | [-8.88, +1.25] pp |
| Previous direction − Majority | +4.16 pp | [-1.25, +9.29] pp |

The intervals include zero, so the experiment did not establish a statistically clear improvement over either baseline under the predefined decision rule.

### Conclusion

The experiment did not provide convincing evidence that the selected Logistic Regression model could predict next-day LUCK.KA direction more accurately than simple baseline methods.

This is an important finding: using machine learning does not automatically produce better predictions than simpler alternatives.

The results apply to this stock, dataset and experimental setup. They should not be generalised to the entire PSX.

Detailed results are available in [the LUCK baseline experiment report](reports/luck_baseline_results.md).

## Limitations

- **Single stock:** The current completed experiment focuses on LUCK.KA and cannot establish performance across the entire PSX.
- **Limited features:** The model uses a relatively small set of technical indicators.
- **Model complexity:** Logistic Regression captures linear relationships and may miss nonlinear patterns.
- **Data source:** Yahoo Finance data was not independently cross-checked against official PSX records.
- **Market changes:** Historical relationships may not remain predictive under changing market conditions.
- **Statistical assumptions:** The paired bootstrap treats test days as independent, although financial observations can exhibit temporal dependence.
- **Trading profitability:** Accuracy does not establish profitability. Transaction costs, liquidity, slippage and price limits were not modelled.

## Future Improvements

The project can be extended through several carefully controlled experiments:

1. **Feature engineering:** Investigate additional past-only indicators such as RSI, MACD, momentum and alternative volatility measures.
2. **Alternative models:** Compare Logistic Regression with Random Forest and XGBoost to investigate whether nonlinear models capture useful patterns.
3. **Stacking ensemble:** Combine predictions from multiple base models using a Logistic Regression meta-model. Chronological out-of-fold predictions will be required to prevent data leakage.
4. **Stability analysis:** Evaluate model performance across different chronological portions of the test period.
5. **Broader evaluation:** Investigate additional PSX stocks to determine whether findings generalise across companies.
6. **Trading simulation:** Explore whether any predictive improvement translates into potential returns after realistic transaction costs and trading constraints.

These are proposed extensions, not completed results. Any new experiment will have its own documented methodology and evaluation procedure.

## Repository Structure

```text
psx-ml-direction/
├── data/                 # Generated locally; not tracked
├── notebooks/            # Data analysis and experiments
├── reports/              # Detailed experiment reports
├── src/                  # Reusable Python modules
├── requirements.txt
└── README.md
```

## Reproducibility

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hamzahafeezshaikh0-ui/psx-ml-direction.git
cd psx-ml-direction

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Open the project in Jupyter or VS Code and run the notebooks in numerical order, starting with Notebook 01.

Notebook 01 downloads and prepares the data locally. Subsequent notebooks use the generated datasets.

## Disclaimer

This is an educational machine-learning research project, not financial advice or a trading recommendation.