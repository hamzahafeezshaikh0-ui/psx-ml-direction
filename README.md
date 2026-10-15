# Evaluating Machine Learning for Next-Day Direction Prediction in the Pakistan Stock Exchange

## Research question

Can machine learning predict next-trading-day market direction for selected PSX stocks, and does it outperform simple baseline prediction methods on genuinely unseen historical data?

The task is binary classification: the label is 1 if the next session's close is higher than today's close, and 0 otherwise. The prediction is made after today's close, using only information available by then.

The purpose is not to build a trading system or claim that the market can be predicted reliably. The aim is a fair experiment to see whether a simple model adds predictive value over simple alternatives. A result close to the baselines is a valid finding.

## Approach

- **Tools:** Python, pandas, NumPy, matplotlib, scikit-learn.
- **Data:** daily PSX stock data, starting with one stock (LUCK) and expanding to about three liquid stocks.
- **Features:** a small set built only from past prices and volume (previous-day return, 5-day return, 10-day volatility, volume change, high-low range, distance from the 20-day moving average).
- **Baselines:** always predict the most common class from the training data, and repeat the previous day's direction.
- **Model:** logistic regression.
- **Evaluation:** chronological split (never shuffled), with the last part of the data kept untouched as the final test. Accuracy is reported with confidence intervals, alongside balanced accuracy, confusion matrix and log-loss.

## Data

- **Source:** Yahoo Finance via the yfinance library, ticker LUCK.KA (Lucky Cement), daily Open/High/Low/Close/Volume.
- **Coverage:** 2015-01-01 to 2026-10-13. 3,053 raw rows, 2,885 trading days after cleaning.
- **Note:** prices come from Yahoo, so they may differ from prices on a broker screen.

### Data problem found and fixed

Lucky Cement did a 5-for-1 stock split in April 2025. Yahoo's data did not handle it correctly:

- The price fell about 80% in one day (about 1,529 to about 314 on 2025-04-10), which looked like a crash but was just the split.
- Three rows in Feb 2025 (26-28 Feb) were already on the post-split scale, so the price flipped back and forth.

**Fix:** divided prices before 2025-04-10 by 5, multiplied their volume by 5, left the three Feb rows as they were, and removed 168 zero-volume rows (holidays, the trading suspension, filler).

**Check:** after the fix, the price chart is smooth and the volume chart has no lasting step. The largest daily move is about 10%, which looks consistent with the PSX daily price limit.

### Data quality checks (all passed)

- Sorted oldest to newest, 0 duplicate dates, 0 missing values, 0 rows with High < Low.
- Zero-change days: 0.07% of days.
- Largest daily moves (about 8-10%) look like real limit moves, not errors.

## Limitations

- One data source, not cross-checked against the PSX portal. The checks catch large errors, not small ones.
- Data files are not stored in this repo. Run notebook 01 to download and repair them.
- Predictive accuracy does not establish that a trading strategy would be profitable, because transaction costs, liquidity and price limits are not modelled.

## Disclaimer

This is an educational project and not financial advice.