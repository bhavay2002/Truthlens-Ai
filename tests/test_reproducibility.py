import numpy as np
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def test_random_seed_reproducibility():

    set_seed(42)
    a = np.random.rand(5)

    set_seed(42)
    b = np.random.rand(5)

    assert (a == b).all()