import numpy as np


def select_random_valid_action(valid_mask: np.ndarray, rng: np.random.Generator) -> int:
    """Baseline 1: Random Valid Dispatch (establishes valid performance floor)."""
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return 0
    return int(rng.choice(valid_indices))


def select_rule_based_action(valid_mask: np.ndarray) -> int:
    """
    Baseline 2: Rule-Based Green Heuristic.
    Action catalog order is:
      0..4: EV (10, 15, 20, 25, 30 kg)
      5..9: Petrol (10, 15, 20, 25, 30 kg)
      10..14: Diesel (10, 15, 20, 25, 30 kg)
    Picks the smallest-capacity valid EV first; falls back to smallest Petrol, then Diesel.
    """
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return 0
    return int(valid_indices[0])
