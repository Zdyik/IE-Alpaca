from datetime import timedelta

import numpy as np
import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.landmark_v3 import build_landmarks
from ie_alpaca.training.landmark_v3 import compress_outcomes, horizon_probability, make_outcomes


def test_compressed_loss_matches_daily_likelihood():
    cases = pd.DataFrame([
        {"gpsno": "a", "anchor_day": 20, "horizon_days": 40, "first_event_day": 1, "label": 1},
        {"gpsno": "a", "anchor_day": 30, "horizon_days": 30, "first_event_day": 12, "label": 1},
        {"gpsno": "b", "anchor_day": 20, "horizon_days": 40, "first_event_day": np.nan, "label": 0},
    ])
    weighted = compress_outcomes(cases)
    q = {("a", 20): .03, ("a", 30): .07, ("b", 20): .01}
    compact = sum(-r.weight * (np.log(q[r.gpsno, r.anchor_day]) if r.y else np.log1p(-q[r.gpsno, r.anchor_day]))
                  for r in weighted.itertuples())
    direct = (-np.log(.03) - 11 * np.log1p(-.07) - np.log(.07)) / 2 - 40 * np.log1p(-.01)
    assert np.isclose(compact, direct)
    assert len(weighted) == 4
    assert np.isclose(horizon_probability(np.array([.01]), 40)[0], 1 - .99 ** 40)


def test_landmark_excludes_future_and_starts_label_next_day():
    days = pd.DataFrame({"gpsno": "a", "day": [FIRST_DAY + timedelta(days=i) for i in range(60)]})
    defaults = {
        "distance_km": 1., "drive_hours": 1., "trajectory_recorded_today": True,
        "imu_recorded_today": True, "night_distance_km": 0., "distance_invalid_rows": 0,
        "distance_valid_rows": 1, "long_gap_rows": 0, "imu_accel_peak_p99": 1.,
        "imu_gyro_peak": 1., "imu_valid_windows": 1,
    }
    from ie_alpaca.features.landmark_v3 import EVENT_FIELDS
    defaults.update({name: 0 for name in EVENT_FIELDS})
    days = days.assign(**defaults)
    days.loc[19, "event_11804_count"] = 1  # day 20 is known at day-end
    days.loc[20, "event_11803_count"] = 1  # day 21 is the next-day target
    anchor = build_landmarks(days, ["a"], anchors=(20,))
    assert anchor.event_11804_count_per_day.iloc[0] > 0
    assert anchor.event_11803_count_per_day.iloc[0] == 0
    outcome = make_outcomes(days, anchor)
    assert outcome.first_event_day.iloc[0] == 1
    changed = days.copy()
    changed.loc[20:, "distance_km"] = 1e6
    changed.loc[20:, "event_11804_count"] = 1000
    pd.testing.assert_frame_equal(anchor, build_landmarks(changed, ["a"], anchors=(20,)))
