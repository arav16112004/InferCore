"""Compare probability estimates using only the original 80 training flowers.

Run from the repository root with the pinned environment:
    .venv/bin/python -m scripts.evaluate_model > models/selection.json
"""
import json
from itertools import product

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CANDIDATES = list(product((False, True), (0.1, 1.0, 10.0)))
BASELINE = (False, 1.0)


def estimator(standardize, c):
    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=2026)
    return make_pipeline(StandardScaler(), clf) if standardize else clf


def metrics(y, probabilities):
    return {
        "log_loss": float(log_loss(y, probabilities)),
        "correct": int(np.sum(probabilities.argmax(axis=1) == y)),
        "samples": len(y),
    }


def compare_candidates(features, labels, folds):
    results = []
    for scale, c in CANDIDATES:
        probabilities = cross_val_predict(
            estimator(scale, c), features, labels, cv=folds, method="predict_proba"
        )
        results.append({"standardize": scale, "C": c, **metrics(labels, probabilities)})
    return results


def evaluate():
    from sklearn import __version__ as sklearn_version

    features, labels = load_iris(return_X_y=True)
    features, labels = features[labels != 0].astype(np.float32), labels[labels != 0] - 1
    x, _, y, _ = train_test_split(features, labels, test_size=0.2, random_state=2026,
                                   stratify=labels)
    # Full-training CV chooses a configuration for the final fit. Its winner's
    # score is optimistic; separate outer predictions assess the choice procedure.
    full_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    full_results = compare_candidates(x, y, full_cv)
    outer_results = []
    for seed in (4101, 4102, 4103):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit, evaluate_on) in enumerate(outer.split(x, y)):
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed * 10 + fold)
            candidates = compare_candidates(x[fit], y[fit], inner)
            selected = min(candidates, key=lambda row: row["log_loss"])
            summary = {"outer_seed": seed, "fold": fold, "standardize": selected["standardize"],
                       "C": selected["C"]}
            for label, params in (("baseline", BASELINE),
                                  ("selected", (selected["standardize"], selected["C"]))):
                clf = estimator(*params).fit(x[fit], y[fit])
                summary[label] = metrics(y[evaluate_on], clf.predict_proba(x[evaluate_on]))
            outer_results.append(summary)

    aggregate = {}
    for label in ("baseline", "selected"):
        rows = [split[label] for split in outer_results]
        count = sum(row["samples"] for row in rows)
        aggregate[label] = {
            "log_loss": sum(row["log_loss"] * row["samples"] for row in rows) / count,
            "correct": sum(row["correct"] for row in rows),
            "evaluations": count,
        }
    winner = min(full_results, key=lambda row: row["log_loss"])
    return {"sklearn": sklearn_version, "training_rows": len(y), "original_split_seed": 2026,
            "selection_metric": "inner validation log loss (lower is better)",
            "full_training_cv": full_results,
            "final_configuration": {"standardize": winner["standardize"], "C": winner["C"]},
            "nested_outer_folds": outer_results, "nested_aggregate": aggregate}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(evaluate(), indent=2))
