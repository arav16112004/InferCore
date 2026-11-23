"""Reproducible trained binary Iris classifier exported with standard ONNX ops.

Input: sepal length/width and petal length/width, centimetres, float32 [1,4].
Output: [P(versicolor), P(virginica)], float32 [1,2]. Setosa is excluded.
This tiny educational model is for serving mechanics, not model research.
"""
import hashlib
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import MODEL


def train():
    x, y = load_iris(return_X_y=True)
    x, y = x[y != 0].astype(np.float32), y[y != 0] - 1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2026, stratify=y)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000, random_state=2026),
    )
    clf.fit(x_train, y_train)
    return clf, x_test, y_test


def generate(path=MODEL):
    clf, x_test, y_test = train()
    # Fold standardization into the affine transform. Workers still accept cm.
    scaler = clf.named_steps["standardscaler"]
    logistic = clf.named_steps["logisticregression"]
    weights = logistic.coef_ / scaler.scale_
    bias = logistic.intercept_ - weights @ scaler.mean_
    constants = [("weights", weights.T.astype(np.float32)),
                 ("bias", bias.astype(np.float32)),
                 ("one", np.array([1.0], dtype=np.float32))]
    nodes = [helper.make_node("MatMul", ["features", "weights"], ["linear"]),
             helper.make_node("Add", ["linear", "bias"], ["logits"]),
             helper.make_node("Sigmoid", ["logits"], ["p1"]),
             helper.make_node("Sub", ["one", "p1"], ["p0"]),
             helper.make_node("Concat", ["p0", "p1"], ["scores"], axis=1)]
    graph = helper.make_graph(nodes, "infercore_iris_logistic",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(a, n) for n, a in constants])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              ir_version=9, producer_name="InferCore")
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model": "binary Iris logistic regression", "train_samples": 80,
                "test_samples": 20, "test_accuracy": float(clf.score(x_test, y_test)),
                "split_seed": 2026, "opset": 17, "ir_version": 9,
                "classes": ["versicolor", "virginica"],
                "C": 10.0, "standardization": "folded into weights and intercept",
                "features_cm": ["sepal_length", "sepal_width", "petal_length", "petal_width"]}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
