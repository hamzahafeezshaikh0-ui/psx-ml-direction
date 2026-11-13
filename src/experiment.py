
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from features import make_features, FEATURES
from evaluation import time_split, evaluate, diff_ci

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_SETS = {
    "all6": FEATURES,
    "core3": ["ret_1", "ret_5", "vol_10"]
}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def make_model(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000)
    )


def _diff_row(name, y, p1, p2):
    d, lo, hi = diff_ci(y, p1, p2)
    return {
        "comparison": name,
        "diff": d,
        "ci_low": lo,
        "ci_high": hi
    }


def run_experiment(df, label="stock"):
    """df needs Open, High, Low, Close, Volume and target. Same procedure for every stock."""

    data = make_features(df[PRICE_COLS]).join(df["target"])
    dev, test = time_split(data)

    n_test_before = len(test)
    dev, test = dev.dropna(), test.dropna()

    assert len(test) == n_test_before, "rows lost to NaN in the test period"

    y_dev = dev["target"].to_numpy()
    y_test = test["target"].to_numpy()

    # 1) Choose feature set and C on development data only
    #    (expanding-window CV, log-loss)

    cv = TimeSeriesSplit(n_splits=5, gap=1)
    rows = []

    for fs_name, cols in FEATURE_SETS.items():
        for C in C_GRID:
            scores = cross_val_score(
                make_model(C),
                dev[cols],
                y_dev,
                cv=cv,
                scoring="neg_log_loss"
            )

            rows.append({
                "features": fs_name,
                "C": C,
                "cv_log_loss": -scores.mean()
            })

    cv_table = (
        pd.DataFrame(rows)
        .sort_values("cv_log_loss")
        .reset_index(drop=True)
    )

    best = cv_table.iloc[0]

    chosen = {
        "features": best["features"],
        "C": float(best["C"])
    }

    cols = FEATURE_SETS[chosen["features"]]

    # 2) Fit the final model on all development data

    model = make_model(chosen["C"]).fit(dev[cols], y_dev)

    # 3) Generate predictions on the untouched test data

    prob_M = model.predict_proba(test[cols])[:, 1]
    pred_M = (prob_M > 0.5).astype(int)

    # 4) Generate baseline predictions

    majority = int(y_dev.mean() > 0.5)

    pred_A = np.full(len(test), majority)
    prob_A = np.full(len(test), y_dev.mean())

    pred_B = (test["ret_1"] > 0).astype(int).to_numpy()

    # 5) Evaluate all three methods

    name_M = f"LogReg ({chosen['features']}, C={chosen['C']})"

    results = pd.DataFrame([
        evaluate(y_test, pred_A, "A: majority class", prob_A),
        evaluate(y_test, pred_B, "B: previous direction"),
        evaluate(y_test, pred_M, name_M, prob_M),
    ])

    # 6) Compare accuracy differences

    diffs = pd.DataFrame([
        _diff_row("LogReg - A", y_test, pred_M, pred_A),
        _diff_row("LogReg - B", y_test, pred_M, pred_B),
        _diff_row("B - A", y_test, pred_B, pred_A),
    ])

    # 7) Extract learned model coefficients

    coefs = pd.Series(model[-1].coef_[0], index=cols)

    # 8) Return all results

    return {
        "label": label,
        "results": results,
        "diffs": diffs,
        "cv_table": cv_table,
        "chosen": chosen,
        "coefs": coefs,
        "y_test": y_test,
        "pred_A": pred_A,
        "pred_B": pred_B,
        "pred_M": pred_M,
        "dev_range": (
            dev.index.min().date(),
            dev.index.max().date(),
            len(dev)
        ),
        "test_range": (
            test.index.min().date(),
            test.index.max().date(),
            len(test)
        ),
    }
