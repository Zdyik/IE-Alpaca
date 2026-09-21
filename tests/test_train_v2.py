"""Contracts for time truncation, fold-local scaling and observable labels."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ie_alpaca.features.sequence_v2 import SequenceData, fit_scaler, labels_for, transform
from ie_alpaca.models.temporal_v2 import TemporalRiskNet


class TemporalV2Contracts(unittest.TestCase):
    def setUp(self):
        self.features = ["event_11804_count", "distance_km", "imu_valid_windows", "has_event_feed"]
        self.groups = {"event": [0], "trajectory": [1], "imu": [2], "quality": [3]}
        raw = np.ones((2, 60, 4), dtype=np.float32)
        raw[1, :, :] = 10
        events = np.zeros((2, 60), dtype=bool)
        events[0, 20] = True  # June 21: first day after June 20 anchor
        events[0, 59] = True  # July 30: last observed label day
        self.data = SequenceData(["train", "validation"], self.features, self.groups,
                                 raw, events, np.array([True, True]))

    def test_labels_exclude_anchor_and_mask_unobserved_horizons(self):
        rows, y, available = labels_for(self.data, np.array([0]))
        self.assertEqual(rows.tolist(), [[0, 20], [0, 30], [0, 39], [0, 46], [0, 53]])
        self.assertEqual(y[0, -1], 1)
        self.assertEqual(y[-1, 0], 1)
        self.assertEqual(available[-1].tolist(), [1, 0, 0, 0, 0])
        self.data.events[0, 20] = False
        _, y_changed, _ = labels_for(self.data, np.array([0]))
        self.assertEqual(y_changed[0, 0], 0)

    def test_validation_data_cannot_change_scaler_or_truncated_prediction(self):
        scaler = fit_scaler(self.data, {"train"})
        original, mask = transform(self.data, scaler)
        self.data.raw[1, :, :] = 1_000_000
        self.assertEqual(scaler, fit_scaler(self.data, {"train"}))
        self.data.raw[1, :, :] = 10
        torch.manual_seed(4)
        model = TemporalRiskNet(self.groups, 4, hidden=4, dropout=0).eval()
        length = torch.tensor([20])
        with torch.inference_mode():
            p1, _ = model(torch.tensor(original[:1]), torch.tensor(mask[:1]), length)
            modified = original[:1].copy()
            modified[:, 20:, :] = 1000  # future behaviour must not enter June 20 prediction
            p2, _ = model(torch.tensor(modified), torch.tensor(mask[:1]), length)
        np.testing.assert_allclose(p1.numpy(), p2.numpy(), atol=1e-7)
        self.assertTrue(bool(torch.all(p1[:, 1:] >= p1[:, :-1])))


if __name__ == "__main__":
    unittest.main()
