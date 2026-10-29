"""Voting contracts for the optional mixed-scale ensemble example."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("sktime")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from example_utils import hard_vote


def test_five_member_vote_uses_training_class_order_for_multiclass_ties():
    predictions = np.array([
        ["b", "a", "c"],
        ["a", "b", "c"],
        ["b", "b", "a"],
        ["a", "c", "c"],
        ["c", "a", "b"],
    ])
    # The first two cases tie at two votes each; the final case has a majority.
    result = hard_vote(predictions, np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(result, ["a", "a", "c"])


def test_vote_rejects_labels_outside_the_training_classes():
    with pytest.raises(ValueError, match="unknown class"):
        hard_vote([["a", "unseen"], ["a", "b"]], ["a", "b"])
