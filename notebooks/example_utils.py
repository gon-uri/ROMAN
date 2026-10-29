"""Small, reproducible MiniRocket examples for the ROMAN notebooks.

The raw S=1 baseline and normalized ROMAN inputs follow the paper's recorded
MiniRocket protocol. Timing covers complete pipelines on a single thread.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter

# Configure numba before importing sktime's ROCKET implementation.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "roman_examples_numba"))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from threadpoolctl import threadpool_limits

from roman import RomanOperator


def load_dataset(name, data_root=None):
    """Load a fixed-length archive dataset in (cases, channels, time) form."""
    data_root = data_root or os.environ.get("ROMAN_DATA_ROOT")
    kwargs = {"extract_path": str(data_root)} if data_root else {}
    X_train, y_train = load_UCR_UEA_dataset(name, split="train", return_type="numpy3D", **kwargs)
    X_test, y_test = load_UCR_UEA_dataset(name, split="test", return_type="numpy3D", **kwargs)
    # Preserve the loader's precision when fitting the normalization statistics.
    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)


def make_model(scale=4, seed=0, num_kernels=10_000):
    """Build a complete raw-input-to-label pipeline; S=1 bypasses ROMAN."""
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    steps = []
    if scale > 1:
        steps.append(RomanOperator(S=scale, alpha=.5, min_timesteps_per_channel=32,
                                   normalization=True))
    steps.extend([MiniRocketMultivariate(num_kernels=num_kernels, random_state=seed, n_jobs=1),
                  RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))])
    return make_pipeline(*steps)


def warm_up():
    """Compile MiniRocket and initialize the numerical libraries outside timers."""
    X = np.random.default_rng(0).normal(size=(12, 2, 256))
    y = np.array(["a", "b"] * 6)
    with threadpool_limits(limits=1):
        for scale in [1, 4]:
            model = make_model(scale, seed=0, num_kernels=84)
            model.fit(X, y)
            model.predict(X)


def timed_predictions(predict, expected, repeats=5):
    """Median warm full-test-batch prediction time, checking every prediction."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    np.testing.assert_array_equal(predict(), expected)  # Untimed warm-up.
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        actual = predict()
        durations.append(perf_counter() - start)
        np.testing.assert_array_equal(actual, expected)
    return float(np.median(durations)), durations


def benchmark_single(X_train, y_train, X_test, y_test, seeds=(0, 1, 2),
                     scales=(1, 4), repeats=5):
    """Accuracy over seeds; training once per seed; warm test-only predictions.

    Training includes ROMAN fit/transform, training-feature extraction and ridge
    fitting. Each prediction recomputes all test features from the raw inputs.
    Downloading, loading and first-time compilation are outside these timers.
    """
    warm_up()
    rows = []
    with threadpool_limits(limits=1):
        for index, seed in enumerate(seeds):
            # Alternate the ordering to reduce a systematic ordering advantage.
            order = list(scales) if index % 2 == 0 else list(reversed(scales))
            for scale in order:
                model = make_model(scale, seed)
                start = perf_counter()
                model.fit(X_train, y_train)
                train_s = perf_counter() - start
                prediction = model.predict(X_test)
                inference_s, samples = timed_predictions(lambda: model.predict(X_test), prediction, repeats)
                rows.append(dict(scale=scale, seed=seed,
                                 test_accuracy=accuracy_score(y_test, prediction),
                                 training_s=train_s, inference_s=inference_s,
                                 inference_repetitions_s=samples))
                print(f"S={scale}, seed={seed}: accuracy={rows[-1]['test_accuracy']:.2%}", flush=True)
    details = pd.DataFrame(rows).sort_values(["scale", "seed"]).reset_index(drop=True)
    summary = details.groupby("scale", as_index=False).agg(
        test_accuracy=("test_accuracy", "mean"), accuracy_sd=("test_accuracy", "std"),
        training_s=("training_s", "median"), inference_s=("inference_s", "median"))
    return details, summary


def hard_vote(predictions, classes):
    """One vote per member; ties choose the first training-label class."""
    predictions = np.asarray(predictions)
    classes = np.asarray(classes)
    if predictions.ndim != 2 or predictions.shape[0] == 0:
        raise ValueError("predictions must have shape (members, cases)")
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes):
        raise ValueError("classes must contain each training-label class once")
    if not np.isin(predictions, classes).all():
        raise ValueError("Member predictions contain an unknown class")
    counts = np.stack([(predictions == label).sum(axis=0) for label in classes])
    return classes[counts.argmax(axis=0)]


def benchmark_ensembles(X_train, y_train, X_test, y_test, repeats=5):
    """Paper's five-member hard-vote comparison, fitting eight unique members."""
    baseline = [(1, seed) for seed in range(5)]
    mixed = [(1, 0), (1, 1), (2, 0), (3, 0), (4, 0)]
    classes = np.unique(y_train)
    warm_up()
    models, fit_times = {}, {}
    with threadpool_limits(limits=1):
        for scale, seed in dict.fromkeys(baseline + mixed):
            model = make_model(scale, seed)
            start = perf_counter()
            model.fit(X_train, y_train)
            fit_times[(scale, seed)] = perf_counter() - start
            models[(scale, seed)] = model

        def predict(members):
            return hard_vote([models[key].predict(X_test) for key in members], classes)

        rows, predictions = [], {}
        for name, members in [("Five S=1 models", baseline), ("Mixed S=1,1,2,3,4", mixed)]:
            prediction = predict(members)
            inference_s, repetitions = timed_predictions(lambda: predict(members), prediction, repeats)
            predictions[name] = prediction
            rows.append(dict(ensemble=name, members=len(members),
                             test_accuracy=accuracy_score(y_test, prediction),
                             correct=int(np.sum(prediction == y_test)), n_test=len(y_test),
                             training_s=sum(fit_times[key] for key in members),
                             inference_s=inference_s, inference_repetitions_s=repetitions))
    return pd.DataFrame(rows), predictions


def environment():
    """Versions and hardware context for the published illustrative measurements."""
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), threads=1,
                packages={name: importlib.metadata.version(name) for name in
                          ["numpy", "pandas", "scipy", "scikit-learn", "sktime", "numba", "threadpoolctl"]})


def display_single(summary):
    table = summary.copy()
    table["Model"] = table.scale.map(lambda s: "MiniRocket (S=1)" if s == 1 else f"ROMAN S={s} + MiniRocket")
    table["Test accuracy"] = table.test_accuracy.map(lambda v: f"{100*v:.2f}%")
    table["Training time"] = table.training_s.map(lambda v: f"{v:.3f} s")
    table["Inference time"] = table.inference_s.map(lambda v: f"{v:.3f} s")
    return table[["Model", "Test accuracy", "Training time", "Inference time"]]
