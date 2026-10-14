import math

import pytest

from game_rsi.training import target_entropy

torch = pytest.importorskip("torch")


def test_target_entropy_matches_soft_label_floor_and_handles_zeros():
    gold = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    expected = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
    assert float(target_entropy(gold)) == pytest.approx(expected, abs=1e-7)
    assert float(target_entropy(torch.eye(3))) == 0.0


def test_target_entropy_averages_examples_with_different_uncertainty():
    gold = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    assert float(target_entropy(gold)) == pytest.approx(math.log(2) / 2, abs=1e-7)
