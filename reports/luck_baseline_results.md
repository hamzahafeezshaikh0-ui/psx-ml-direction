# LUCK Baseline Experiment — Results and Analysis

**Project:** Evaluating Machine Learning for Next-Day Direction Prediction in the Pakistan Stock Exchange  
**Stock:** Lucky Cement (LUCK)  
**Experiment status:** Baseline completed; further model development ongoing.

## 1. Objective

This experiment investigates whether Logistic Regression can predict the next-trading-day direction of Lucky Cement shares using historical price and volume data.

The prediction target is binary:

- **UP (1):** The next trading day's closing price is higher than the current day's closing price.
- **DOWN (0):** The next trading day's closing price is equal to or lower than the current day's closing price.

The model is compared against two simple baselines to determine whether it provides predictive value beyond straightforward approaches.

## 2. Dataset and experimental setup

The experiment uses daily historical OHLCV data for Lucky Cement (`LUCK.KA`).

| Item | Details |
|---|---|
| Development period | 28 January 2015 – 13 October 2023 |
| Test period | 17 October 2023 – 12 October 2026 |
| Development observations | 2,143 |
| Test observations | 721 |
| Prediction horizon | Next trading day |
| Model | Logistic Regression |

The data was divided chronologically rather than randomly. This preserves the time order of observations and prevents future observations from being used to train a model that predicts the past.

A boundary observation was removed to prevent a development-set target from depending on a closing price belonging to the test period.

## 3. Features

Six features were initially investigated:

| Feature | Description |
|---|---|
| `ret_1` | One-session percentage return |
| `ret_5` | Five-session percentage return |
| `vol_10` | Ten-session rolling standard deviation of returns |
| `vol_chg` | Current volume relative to its 20-session average |
| `hl_range` | Daily high-low range relative to closing price |
| `ma_dist` | Closing price relative to its 20-session moving average |

All features use information available on or before the prediction date.

Two feature configurations were compared:

- **all6:** All six features.
- **core3:** `ret_1`, `ret_5`, and `vol_10`.

## 4. Model selection

Logistic Regression was selected as the initial machine-learning model.

The model was implemented using a pipeline containing:

1. `StandardScaler` to standardise the input features.
2. `LogisticRegression` to estimate the probability of an UP movement.

Hyperparameter and feature-set selection were performed using five-fold expanding-window time-series cross-validation on the development dataset.

A one-observation gap was used between training and validation portions to reduce target overlap at fold boundaries.

The selection metric was log loss, where lower values indicate better probability estimates.

The original search considered:

`C = [0.001, 0.01, 0.1, 1, 10]`

The selected configuration was:

- **Features:** `core3`
- **Regularisation:** L2
- **C:** 0.001
- **Cross-validation log loss:** 0.6913

The selected model was then fitted on the full development dataset and evaluated on the chronological test period.

## 5. Baselines

The model was compared against two simple approaches.

**Baseline A — Majority class**

Always predicts the most frequent target class in the development dataset.

**Baseline B — Previous direction**

Predicts UP if the current day's return is positive and DOWN otherwise.

These baselines provide reference points for assessing whether Logistic Regression offers an improvement over simple prediction strategies.

## 6. Test results

The following results were obtained on the 721 test observations.

| Method | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| Majority class | 49.38% | 50.00% | 0.6952 |
| Previous direction | 53.54% | 53.53% | N/A |
| Logistic Regression | 49.51% | 50.11% | 0.6949 |

### ROC-AUC

| Method | ROC-AUC |
|---|---:|
| Previous direction | 0.5343 |
| Logistic Regression | 0.5320 |

For the previous-direction baseline, the current day's return was used as a continuous prediction score when calculating ROC-AUC.

ROC-AUC measures how well a model ranks actual UP observations above actual DOWN observations. A value of 0.5 represents random ranking.

Both approaches achieved values only slightly above 0.5.

## 7. Statistical comparison

Paired bootstrap confidence intervals were calculated for differences in test accuracy.

| Comparison | Difference | 95% confidence interval |
|---|---:|---:|
| Logistic Regression − Majority | +0.14 percentage points | −0.97 to +1.25 pp |
| Logistic Regression − Previous direction | −4.02 percentage points | −8.88 to +1.25 pp |
| Previous direction − Majority | +4.16 percentage points | −1.25 to +9.29 pp |

All three confidence intervals include zero.

Consequently, the experiment does not provide clear statistical evidence that any of the methods outperforms the others in accuracy.

The original hypothesis that Logistic Regression would outperform both baselines is therefore not supported by these results.

## 8. Model diagnostics

Additional diagnostics were conducted to understand the Logistic Regression model's predictions.

The model predicted UP on only 17 of the 721 test observations.

| Diagnostic | Result |
|---|---:|
| UP predictions | 17 |
| UP precision | 52.94% |
| UP recall | 2.47% |
| Percentage of observations predicted UP | 2.36% |
| Logistic Regression ROC-AUC | 0.5320 |

The predicted probabilities were concentrated around 0.475, indicating that the model generally assigned probabilities close to the classification threshold of 0.5.

Exploratory probability and threshold analysis showed that changing the threshold substantially changed the number of UP predictions. However, these observations do not establish that an alternative threshold would generalise better.

## 9. Interpretation

The initial Logistic Regression model did not demonstrate a meaningful predictive advantage over the simple baselines.

The previous-direction baseline achieved higher accuracy and ROC-AUC than Logistic Regression. The Logistic Regression model achieved marginally lower log loss than the majority baseline, but the difference was very small.

The model's low UP recall also indicates that it rarely identified actual UP movements at the default 0.5 threshold.

These findings suggest that the current feature set and linear model may not capture enough information to predict next-day price direction reliably.

The results should not be interpreted as evidence that machine learning cannot predict stock direction. They indicate that this particular model, feature set and experimental configuration did not establish a meaningful advantage.

## 10. Evaluation notes

The model was trained exclusively on the development dataset. Feature selection and hyperparameter selection for the primary baseline were performed using chronological cross-validation on development data.

After the initial evaluation, additional diagnostic analyses were conducted to better understand the model's predictions, including probability distributions, classification thresholds and ROC-AUC.

These analyses did not involve fitting the model on test observations or changing its learned weights using test data. The original model configuration and test results remain the primary baseline results.

For future experiments, model selection will continue to rely on development-set cross-validation, with a separate unseen period reserved for final evaluation.

## 11. Next steps

The next stage of the project will investigate whether predictive performance can be improved through:

- Additional momentum and volatility features.
- Alternative feature combinations.
- Other classification models.
- Consistent evaluation across multiple PSX stocks.
- Comparisons against the same simple baselines.

Any improvements will be selected using chronological cross-validation rather than test-set performance.

A genuinely unseen future period should be reserved for final confirmation of the completed approach.

## 12. Conclusion

The LUCK baseline experiment established a reproducible starting point for evaluating next-day stock direction prediction.

The initial Logistic Regression model did not outperform the previous-direction baseline and showed very limited ability to identify UP movements.

Rather than treating this as a successful prediction model, the results provide a baseline against which future experiments can be compared.
