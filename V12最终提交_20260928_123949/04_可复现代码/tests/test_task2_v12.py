from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ie_alpaca.features.daily_v12 import STATE_EVENT_CODES
from ie_alpaca.task2.safety_score_v12 import (
    DEFAULT_MAX_POINTS, SCORE_STATES, empirical_percentile, grade,
    normalize_event_weights, risk_percentiles, score_from_risks,
)


class Task2V12Contracts(unittest.TestCase):
    def test_score_is_bounded_and_deductions_reconcile(self):
        risks = pd.DataFrame({name: [0.0, .5, 1.0] for name in SCORE_STATES})
        values = score_from_risks(risks, DEFAULT_MAX_POINTS)
        self.assertTrue(values.safety_score.between(0, 100).all())
        np.testing.assert_allclose(values.safety_score + values.total_deduction, 100.0)
        self.assertGreater(values.safety_score.iloc[0], values.safety_score.iloc[1])
        self.assertGreater(values.safety_score.iloc[1], values.safety_score.iloc[2])

    def test_event_weights_sum_to_one_within_each_state(self):
        rows = []
        for state, codes in STATE_EVENT_CODES.items():
            for index, code in enumerate(codes):
                rows.append({"event_code": code, "state": state,
                             "mean_abs_probability_delta": float(index + 1)})
        weights = normalize_event_weights(pd.DataFrame(rows), uniform_share=.5)
        for state, values in weights.items():
            self.assertAlmostEqual(float(values.sum()), 1.0)
            self.assertTrue((values > 0).all(), state)

    def test_missing_event_feed_is_neutral_and_no_evidence_neutralizes_all(self):
        signals = pd.DataFrame({
            "upstream_event": [0.0, 5.0], "control_event": [0.0, 5.0],
            "control_imu": [0.0, 5.0], "proximal_event": [0.0, 5.0],
            "history_event": [0.0, 5.0], "quality_event": [0.0, 5.0],
        }, index=["a", "b"])
        reference = {name: [0.0, 1.0, 2.0] for name in (
            "model_probability", "upstream_event", "control_event", "control_imu",
            "proximal_event", "history_event", "quality_event")}
        risks = risk_percentiles(
            signals, np.asarray([.1, .9]), reference,
            np.asarray([False, False]),
            np.asarray(["context_only_no_matched_event", "prior_no_observed_behavior"]),
        )
        self.assertAlmostEqual(float(risks.loc["a", "upstream"]), .5)
        np.testing.assert_allclose(risks.loc["b", list(SCORE_STATES)].to_numpy(float), .5)

    def test_grades_have_operational_actions_and_low_evidence_override(self):
        self.assertEqual(grade(90, "高")[0], "A")
        self.assertEqual(grade(40, "高")[0], "E")
        low = grade(90, "低")
        self.assertEqual(low[0], "U")
        self.assertIn("暂缓奖惩", low[2])

    def test_empirical_percentile_is_monotone(self):
        values = empirical_percentile(np.asarray([-1.0, 1.0, 3.0]), [0.0, 1.0, 2.0])
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(((0 <= values) & (values <= 1)).all())


if __name__ == "__main__":
    unittest.main()

