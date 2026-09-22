from datetime import timedelta
import unittest

import numpy as np
import pandas as pd
import torch

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.landmark_v4 import EVENT_CODES, build_features, context_columns, event_columns
from ie_alpaca.models.event_hazard_v4 import EventHazardNet
from ie_alpaca.training.event_v4 import FoldScaler, next_event_loss


def _daily() -> pd.DataFrame:
    frame = pd.DataFrame({"gpsno": "a", "day": [FIRST_DAY + timedelta(days=i) for i in range(60)]})
    basics = {"distance_km": 10., "drive_hours": 1., "trajectory_recorded_today": True,
              "imu_recorded_today": True, "night_distance_km": 1., "distance_invalid_rows": 0,
              "distance_valid_rows": 1, "long_gap_rows": 0, "imu_accel_peak_p99": 1.,
              "imu_gyro_peak": 1., "imu_valid_windows": 1}
    basics.update({f"event_{code}_episodes": 0 for code in EVENT_CODES})
    basics.update({f"event_{code}_count": 0 for code in EVENT_CODES})
    basics.update({"event_11803_count": 0, "event_11804_count": 0,
                   "event_11401_count": 0, "event_11402_count": 0,
                   "event_11403_count": 0, "event_30000_count": 0,
                   "event_30002_count": 0, "event_30003_count": 0,
                   "event_total_episodes": 0})
    frame = frame.assign(**basics)
    frame.loc[19, "event_11405_episodes"] = 1
    frame.loc[19, "event_11405_count"] = 12
    frame.loc[20, "event_41001_episodes"] = 100
    return frame


def test_all_codes_and_day_end_truncation():
    daily = _daily()
    original = build_features(daily, ["a"], anchors=(20,))
    assert len(event_columns()) == 24 * 9
    assert len(context_columns()) == 17
    assert original.event_11405_episodes_per_day.iloc[0] == 1 / 20
    assert original.event_11405_counts_per_day.iloc[0] == 12 / 20
    assert original.event_41001_episodes_per_day.iloc[0] == 0
    changed = daily.copy()
    changed.loc[20:, "event_41001_episodes"] = 999
    changed.loc[20:, "event_41001_count"] = 999
    changed.loc[20:, "distance_km"] = 999999
    pd.testing.assert_frame_equal(original, build_features(changed, ["a"], anchors=(20,)))


def test_vehicle_normalized_right_censored_loss():
    logits = torch.logit(torch.tensor([[.03, .06]], dtype=torch.float64))
    day = torch.tensor([[1., 0.]], dtype=torch.float64)
    observed = torch.tensor([[40., 7.]], dtype=torch.float64)
    actual = next_event_loss(logits, day, observed)
    expected = (-np.log(.03) - 7 * np.log1p(-.06)) / 2
    assert np.isclose(float(actual), expected)


def test_weights_are_relative_and_absent_events_contribute_zero():
    net = EventHazardNet(34, learned_weights=True, dropout=0)
    assert torch.isclose(net.risk_weights().mean(), torch.tensor(1.))
    event = torch.rand(2, 24, 9)
    absent = torch.zeros(2, 24)
    context = torch.zeros(2, 34)
    torch.testing.assert_close(net(event, absent, context), net(torch.zeros_like(event), absent, context))
    present = torch.ones(2, 24)
    net(event, present, context).sum().backward()
    assert net.type_delta.grad is not None
    assert torch.isfinite(net.type_delta.grad).all()


def test_scaler_fits_and_keeps_absence_zero():
    daily = _daily()
    features = build_features(daily, ["a"], anchors=(14, 20))
    scaler = FoldScaler().fit(features)
    events, present, context = scaler.transform(features)
    assert events.shape == (2, 24, 9)
    assert context.shape == (2, 34)
    assert present[0].sum() == 0
    assert present[1, EVENT_CODES.index(11405)] == 1
    assert np.isfinite(events).all() and np.isfinite(context).all()


class EventV4Contracts(unittest.TestCase):
    def test_event_inputs_obey_day_end_boundary(self):
        test_all_codes_and_day_end_truncation()

    def test_loss_respects_censoring(self):
        test_vehicle_normalized_right_censored_loss()

    def test_type_weights_and_absence(self):
        test_weights_are_relative_and_absent_events_contribute_zero()

    def test_scaling_is_finite(self):
        test_scaler_fits_and_keeps_absence_zero()
