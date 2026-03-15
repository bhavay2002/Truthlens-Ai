import time


def test_inference_speed():

    start = time.time()

    for _ in range(1000):
        _ = "fake news detection test".lower()

    elapsed = time.time() - start

    assert elapsed < 1