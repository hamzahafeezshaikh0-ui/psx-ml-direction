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

## Hypotheses (written before any model was evaluated on the test period)

- **H0:** a logistic regression model's accuracy on the final test period is no better than the accuracy of each of the two baselines.
- **H1:** the model beats both baselines on test accuracy, and the 95% bootstrap interval of the accuracy difference (model minus baseline) excludes zero for both comparisons.
- **Secondary hypothesis:** repeating the previous day's direction beats always predicting the majority class.
- **Primary metric:** accuracy on the final test period. Secondary: balanced accuracy, log-loss, precision/recall, confusion matrices.
- **Decision rule:** the model "beats" a baseline only if the lower end of the 95% interval of the accuracy difference is above zero. Otherwise the result is reported as no clear evidence of improvement.


## Protocol (fixed before evaluation)

- Chronological split of each stock's daily data: first 75% = development, last 25% = test. One boundary row between them is dropped.
- Majority class is chosen from development data only.
- Features (all from past data only): previous-day return, 5-day return, 10-day volatility of returns, volume vs its 20-day average, (high - low) / close, distance from the 20-day moving average.
- Model: logistic regression with standardised features. Feature set (all 6, or 3 core ones) and regularisation C (0.001 to 10) are chosen by 5-fold expanding-window cross-validation on development data, using log-loss.
- Prediction rule: predict "up" if the model's probability is above 0.5.
- The test set is used once for the main result. Nothing is tuned after seeing it.
- Comparison: paired bootstrap (5,000 resamples of test days) of the accuracy difference. It treats days as independent, which is a limitation.
- Stocks: LUCK first (chosen because its data was already prepared). Then, from the list PSO, HUBC, INDU in that order, the first two that pass the data checks: at least 2,500 trading days since 2015 after removing zero-volume days, and no unresolved daily price move above 11%. All stocks are reported, including failures. The same procedure is run on each stock.
- Any change to this protocol after this commit is written down in the decision log.


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
- The exploratory statistics in the "Target and exploratory findings" section (up share, returns, lag-1 correlation) were computed on the full 2015-2026 sample, which includes the later test period. No modelling decision was based on them. All later exploration and tuning use development data only.



## Target and exploratory findings

**Target:** for each trading day, the label is 1 if the next session's close is higher than today's close, and 0 if it is lower or the same. The last row has no next day and is removed, leaving 2,884 labelled days. Alignment was checked by hand on the first and last rows.

**Findings (LUCK, 2015-2026):**
- 48.2% of days are up and 51.8% are down or flat. The up share varies by year (44.4% to 54.1%), so the majority-class baseline will be chosen from training data only.
- Daily returns are centred near zero (mean +0.07%, median -0.05%) with fat tails. Extreme days reach about ±10%, which matches the PSX daily price limits.
- The correlation between today's and yesterday's return is 0.108: small, and not evidence of predictability on its own.

## Decision log

Choices made while building the project, with the reason for each. New entries are added as decisions are made.

- **Classification, not regression:** the question is up or not-up, and returns are dominated by rare large moves.
- **Chronological split with a dropped boundary row:** shuffling would let the model train on days after the ones it is tested on.
- **Majority class chosen from development data only:** choosing it from the test labels would be peeking.
- **Logistic regression:** it matches a 0/1 target, gives probabilities, and is easy to verify by hand.
- **Cross-validation gap:** cross-validation uses a gap of one row between training and validation parts (TimeSeriesSplit gap=1), matching the dropped boundary row between development and test. For a one-day label this is a conservative precaution, not a necessity. Added before any model was evaluated.
- **LUCK first:** its data was already cleaned. The other stocks are chosen by the rule committed in advance.

## Experiment reports

Detailed experimental results, model comparisons and ongoing development:

- [LUCK baseline experiment — results and findings](reports/luck_baseline_results.md)


## Disclaimer

This is an educational project and not financial advice.
