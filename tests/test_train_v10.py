import unittest
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, EVENT_FEATURE_NAMES, NodeDayScaler, build_day_table
from ie_alpaca.features.daily_v7 import NUMERIC_COLUMNS, QUALITY_COLUMNS
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.risk_chain_v10 import RiskChainNetV10


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


class V10Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scaler_shape_and_vehicle_isolation(self):
        table = build_day_table(synthetic_daily(), ["car_a", "car_b"])
        scaler = NodeDayScaler().fit(table, {"car_a"})
        events, context = scaler.transform(table)
        self.assertEqual(events.shape, (120, len(EVENT_CODES), len(EVENT_FEATURE_NAMES)))
        self.assertEqual(context.shape, (120, len(CONTEXT_COLUMNS)))
        before = scaler.describe()
        restored = NodeDayScaler.from_description(before)
        restored_events, restored_context = restored.transform(table)
        self.assertTrue(np.array_equal(events, restored_events))
        self.assertTrue(np.array_equal(context, restored_context))
        changed = table.copy(); changed.loc[changed.gpsno.eq("car_b"), "distance_km"] = 1e9
        after = NodeDayScaler().fit(changed, {"car_a"}).describe()
        self.assertEqual(before, after)

    def test_anchor_prediction_cannot_read_future_days(self):
        torch.manual_seed(7)
        events = torch.rand(2, 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES))
        context = torch.rand(2, 60, len(CONTEXT_COLUMNS))
        altered_events, altered_context = events.clone(), context.clone()
        altered_events[:, 20:] = 1e4; altered_context[:, 20:] = 1e4
        for mode in ("flat", "role", "chain", "chain_shuffle", "chain_aux", "quality_chain"):
            net = RiskChainNetV10(context_width=len(CONTEXT_COLUMNS), mode=mode, dropout=0).eval()
            with torch.inference_mode():
                first = net(events, context, (20,))[0]
                second = net(altered_events, altered_context, (20,))[0]
            self.assertTrue(torch.allclose(first, second, atol=1e-5), mode)

    def test_chain_and_shuffle_are_parameter_matched_but_have_different_edges(self):
        chain = RiskChainNetV10(context_width=len(CONTEXT_COLUMNS), mode="chain")
        shuffled = RiskChainNetV10(context_width=len(CONTEXT_COLUMNS), mode="chain_shuffle")
        self.assertEqual(sum(p.numel() for p in chain.parameters()),
                         sum(p.numel() for p in shuffled.parameters()))
        self.assertNotEqual(chain.relations, shuffled.relations)

    def test_auxiliary_head_is_daily_and_causal(self):
        torch.manual_seed(9)
        net = RiskChainNetV10(context_width=len(CONTEXT_COLUMNS), mode="chain_aux", dropout=0).eval()
        events = torch.rand(1, 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES))
        context = torch.rand(1, 60, len(CONTEXT_COLUMNS))
        changed = events.clone(); changed[:, 11:] = 999
        with torch.inference_mode():
            first = net(events, context, (20,))[1][:, 10]
            second = net(changed, context, (20,))[1][:, 10]
        self.assertTrue(torch.allclose(first, second, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
