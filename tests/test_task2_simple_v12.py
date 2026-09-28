from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ie_alpaca.task2.simple_score_v12 import _columnwise_percentile, learned_event_weights


class SimpleScoreV12Test(unittest.TestCase):
    def test_zero_event_has_zero_risk(self):
        values = np.array([0.0, 0.0, 1.0, 2.0])
        result = _columnwise_percentile(values, np.ones(4, dtype=bool), True)
        self.assertTrue(np.array_equal(result[:2], np.zeros(2)))
        self.assertGreater(result[3], result[2])

    def test_event_weights_are_nonnegative_and_normalized(self):
        codes = [11803, 11804, 30000, 30005, 60292, 60294,
                 41001, 41002, 41003, 41004, 41005, 41009, 41023, 41029,
                 11401, 11402, 11403, 11405, 11406, 30002, 30003, 30017]
        table = pd.DataFrame({"event_code": codes,
                              "mean_abs_probability_delta": np.arange(1, len(codes) + 1)})
        weights, output = learned_event_weights(table, .5)
        self.assertEqual(len(output), len(codes))
        for value in weights.values():
            self.assertTrue((value >= 0).all())
            self.assertAlmostEqual(float(value.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
