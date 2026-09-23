import unittest
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.daily_v7 import (
    DayScaler, NUMERIC_COLUMNS, QUALITY_COLUMNS, TOKEN_COLUMNS, build_day_table,
)
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.temporal_transformer_v7 import TemporalHazardV7
from ie_alpaca.training.temporal_v7 import _anchor_logits


def synthetic_daily():
    rows = []
    for gpsno in ("car_a", "car_b"):
        for day in range(1, 61):
            row = {"gpsno": gpsno, "day": FIRST_DAY + timedelta(days=day - 1)}
            row.update({name: 1.0 for name in NUMERIC_COLUMNS})
            row.update({name: True for name in QUALITY_COLUMNS})
            for code in EVENT_CODES:
                row[f"event_{code}_count"] = int(day == 5 and gpsno == "car_a")
                row[f"event_{code}_episodes"] = int(day == 5 and gpsno == "car_a")
            rows.append(row)
    return pd.DataFrame(rows)


class V7TemporalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_token_width_and_scaler_isolated_by_vehicle(self):
        table = build_day_table(synthetic_daily(), ["car_a", "car_b"])
        self.assertEqual(len(TOKEN_COLUMNS), 103)
        scaler = DayScaler().fit(table, {"car_a"})
        before = scaler.describe()
        changed = table.copy()
        changed.loc[changed.gpsno.eq("car_b"), "distance_km"] = 1e9
        after = DayScaler().fit(changed, {"car_a"}).describe()
        self.assertEqual(before, after)
        self.assertEqual(scaler.transform(table).shape, (120, 103))

    def test_future_days_and_padding_cannot_change_anchor_prediction(self):
        table = build_day_table(synthetic_daily(), ["car_a", "car_b"])
        scaler = DayScaler().fit(table, {"car_a"})
        x = torch.from_numpy(scaler.transform(table[table.gpsno.eq("car_a")]).reshape(1, 60, 103))
        future_changed = x.clone()
        future_changed[:, 20:] = 1e5
        for mode in ("transformer", "no_attention"):
            torch.manual_seed(7)
            net = TemporalHazardV7(103, mode=mode).eval()
            with torch.inference_mode():
                first = _anchor_logits(net, x)[0, 1]
                second = _anchor_logits(net, future_changed)[0, 1]
            self.assertTrue(torch.allclose(first, second, atol=1e-5), mode)

    def test_right_padding_and_invalid_mask(self):
        torch.manual_seed(7)
        net = TemporalHazardV7(103, mode="transformer").eval()
        x = torch.randn(2, 60, 103)
        mask = torch.zeros((2, 60), dtype=torch.bool)
        mask[:, :20] = True
        altered = x.clone()
        altered[:, 20:] = 1e6
        with torch.inference_mode():
            self.assertTrue(torch.allclose(net(x, mask), net(altered, mask), atol=1e-5))
        mask[:, 21] = True
        with self.assertRaises(ValueError):
            net(x, mask)


if __name__ == "__main__":
    unittest.main()
