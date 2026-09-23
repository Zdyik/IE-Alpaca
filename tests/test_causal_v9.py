from datetime import timedelta
import unittest

import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.causal_v9 import EVENT_ROLES, build_causal_landmarks
from ie_alpaca.features.landmark_v3 import EVENT_FIELDS


def _daily() -> pd.DataFrame:
    frame = pd.DataFrame({"gpsno": "a", "day": [FIRST_DAY + timedelta(days=i) for i in range(60)]})
    defaults = {
        "distance_km": 10.0, "drive_hours": 1.0, "trajectory_recorded_today": True,
        "imu_recorded_today": True, "night_distance_km": 0.0, "distance_invalid_rows": 0,
        "distance_valid_rows": 1, "long_gap_rows": 0, "imu_accel_peak_p99": 1.0,
        "imu_gyro_peak": 1.0, "imu_valid_windows": 1,
    }
    defaults.update({name: 0 for name in EVENT_FIELDS})
    for codes in EVENT_ROLES.values():
        for code in codes:
            defaults[f"event_{code}_count"] = 0
            defaults[f"event_{code}_episodes"] = 0
    return frame.assign(**defaults)


class CausalFeatureTest(unittest.TestCase):
    def test_directed_chain_has_direction_and_excludes_future(self):
        days = _daily()
        days.loc[2, ["event_41001_count", "event_41001_episodes"]] = 1
        days.loc[4, ["event_30002_count", "event_30002_episodes"]] = 1
        before, sets = build_causal_landmarks(days, ["a"], anchors=(14,))
        self.assertEqual(before.chain_fatigue_to_control_within3_given_source.iloc[0], 1)
        self.assertEqual(before.chain_control_to_severe_within3_given_source.iloc[0], 0)
        self.assertLess(len(sets["e0a_v3_base"]), len(sets["e0b_roles"]))
        self.assertLess(len(sets["e0b_roles"]), len(sets["e0c_cooccurrence"]))
        self.assertLess(len(sets["e0c_cooccurrence"]), len(sets["e0d_directed_chains"]))

        changed = days.copy()
        changed.loc[14:, "event_11804_count"] = 999
        changed.loc[14:, "event_11804_episodes"] = 999
        after, _ = build_causal_landmarks(changed, ["a"], anchors=(14,))
        pd.testing.assert_frame_equal(before, after)
