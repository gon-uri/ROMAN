"""Tests for the ROMAN operator.

Covers the documented behavior of `RomanOperator` and `choose_S_roman`:
the S=1 identity, the output-shape contract, the window-overlap and
coverage properties, the three scale-selection modes with their fallback
warnings, input handling, and scikit-learn conformance.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from roman import RomanOperator, choose_S_roman

RNG = np.random.default_rng(0)


def make_X(n=8, C=2, L=256):
    return RNG.standard_normal((n, C, L)).astype(np.float32)


# ---------------------------------------------------------------- identity


def test_s1_is_exact_identity_without_normalization():
    X = make_X()
    Z = RomanOperator(S=1, normalization=False).fit_transform(X)
    assert np.array_equal(Z, X)


def test_s1_is_identity_up_to_normalization():
    X = make_X()
    op = RomanOperator(S=1, normalization=True).fit(X)
    Z = op.transform(X)
    mean = X.mean(axis=(0, 2), dtype=np.float64)
    std = X.std(axis=(0, 2), dtype=np.float64)
    expected = (X - mean[None, :, None].astype(np.float32)) / np.maximum(
        std[None, :, None].astype(np.float32), op.eps
    )
    assert np.allclose(Z, expected, atol=1e-5)


# ------------------------------------------------------------ output shape


@pytest.mark.parametrize("L", [512, 500, 333])
@pytest.mark.parametrize("C", [1, 3])
def test_output_shape_contract(L, C):
    X = make_X(C=C, L=L)
    op = RomanOperator(S=3, min_timesteps_per_channel=16).fit(X)
    Z = op.transform(X)
    assert Z.shape == (X.shape[0], C * sum(op.windows_), op.L_base_)
    assert op.n_pseudochannels_ == C * sum(op.windows_)
    assert Z.dtype == np.float32


def test_windows_cover_each_scale_end_to_end():
    X = make_X(L=512)
    op = RomanOperator(S=3).fit(X)
    for s in range(op.S_):
        assert op.starts_[s][0] == 0
        assert op.ends_[s][-1] == op.lengths_[s]


def test_consecutive_windows_respect_overlap():
    alpha = 0.5
    X = make_X(L=512)
    op = RomanOperator(S=3, alpha=alpha).fit(X)
    max_advance = (1.0 - alpha) * op.L_base_ + 1  # +1 for integer rounding
    for s in range(op.S_):
        starts = op.starts_[s]
        if len(starts) > 1:
            assert np.all(np.diff(starts) <= max_advance)


# --------------------------------------------------------- selection modes


def test_exactly_one_mode_is_required():
    X = make_X()
    with pytest.raises(ValueError):
        RomanOperator().fit(X)
    with pytest.raises(ValueError):
        RomanOperator(S=2, max_pseudochannels=16).fit(X)


def test_budget_mode_respects_budget():
    X = make_X(C=2, L=512)
    op = RomanOperator(max_pseudochannels=12).fit(X)
    assert op.n_pseudochannels_ <= 12


def test_coverage_mode_runs():
    X = make_X(C=2, L=512)
    op = RomanOperator(N=10000, H=20).fit(X)
    assert op.S_ >= 1


def test_exact_mode_reduces_with_warning():
    X = make_X(L=64)
    with pytest.warns(UserWarning, match="mobility constraint"):
        op = RomanOperator(S=4, min_timesteps_per_channel=32).fit(X)
    assert op.S_ == 2  # 64 -> 32 is the deepest allowed scale


def test_short_series_falls_back_to_s1_with_warning():
    X = make_X(L=16)
    with pytest.warns(UserWarning, match="Falling back to S=1"):
        op = RomanOperator(S=3).fit(X)  # default L_min=32 > 16
    assert op.S_ == 1
    assert op.L_base_ == 16


def test_choose_s_roman_matches_fitted_operator():
    X = make_X(C=3, L=500)
    op = RomanOperator(S=4, min_timesteps_per_channel=16).fit(X)
    S, lengths, windows, L_base = choose_S_roman(
        C=3, alpha=op.alpha, L=500, min_timesteps_per_channel=16, S_exact=4
    )
    assert (S, lengths, windows, L_base) == (
        op.S_,
        op.lengths_,
        op.windows_,
        op.L_base_,
    )


# ---------------------------------------------------------- input handling


def test_2d_input_warns_and_is_treated_as_univariate():
    X2 = RNG.standard_normal((8, 256)).astype(np.float32)
    with pytest.warns(UserWarning, match="2D"):
        op = RomanOperator(S=2).fit(X2)
    with pytest.warns(UserWarning, match="2D"):
        Z = op.transform(X2)
    assert Z.shape[0] == 8 and op.C_ == 1


def test_transform_does_not_mutate_input():
    X = make_X(C=1, L=64)
    X[0, 0, 5] = np.nan
    before = X.copy()
    op = RomanOperator(S=2, min_timesteps_per_channel=8, normalization=False)
    op.fit(X)
    op.transform(X)
    assert np.array_equal(X, before, equal_nan=True)


def test_transform_rejects_wrong_channels_and_length():
    op = RomanOperator(S=2).fit(make_X(C=2, L=256))
    with pytest.raises(ValueError, match="channels"):
        op.transform(make_X(C=3, L=256))
    with pytest.raises(ValueError, match="length"):
        op.transform(make_X(C=2, L=128))


def test_batching_gives_identical_output():
    X = make_X(n=10, L=256)
    op = RomanOperator(S=2).fit(X)
    assert np.array_equal(op.transform(X), op.transform(X, batch_size=3))


# ------------------------------------------------------ sklearn conformance


def test_clone_of_fitted_operator_is_unfitted():
    op = RomanOperator(S=2).fit(make_X())
    fresh = clone(op)
    assert not hasattr(fresh, "S_")
    assert not any(k.endswith("_") for k in op.get_params())


def test_transform_before_fit_raises_notfittederror():
    with pytest.raises(NotFittedError):
        RomanOperator(S=2).transform(make_X())


# ----------------------------------------------------------- relevance map


def test_map_relevance_shapes_and_validation():
    X = make_X(C=2, L=256)
    op = RomanOperator(S=3).fit(X)
    rel = RNG.random(op.n_pseudochannels_).astype(np.float32)
    rel_c, rel_s = op.map_relevance(rel)
    assert rel_c.shape == (op.C_, op.L_)
    assert rel_s.shape == (op.S_, op.L_)
    with pytest.raises(ValueError):
        op.map_relevance(rel[:-1])
