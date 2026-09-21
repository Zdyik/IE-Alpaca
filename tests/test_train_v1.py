"""Behavioral checks for leakage, vehicle split, and co-located artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import labeled_reference  # noqa: E402
from ie_alpaca.features.tabular_v1 import build_features  # noqa: E402


class V1Contracts(unittest.TestCase):
    def test_future_days_cannot_change_reference_features(self):
        days = [date(2026, 6, 1) + timedelta(days=i) for i in range(60)]
        frame = pd.DataFrame({
            "gpsno": "car_1", "day": days, "has_event_feed": True,
            "has_trajectory": True, "trajectory_recorded_today": True,
            "event_11804_count": 0, "event_11803_count": 0,
            "event_total_count": 0, "event_total_episodes": 0,
            "distance_km": 10.0, "drive_hours": 1.0,
        })
        bag = pd.DataFrame({"gpsno": ["car_1"], "anchor_date": [date(2026, 6, 20)]})
        before = build_features(frame, bag)
        frame.loc[frame.day > date(2026, 6, 20), [
            "event_11804_count", "event_11803_count", "event_total_count",
            "event_total_episodes", "distance_km", "drive_hours",
        ]] = 99999
        frame.loc[frame.day > date(2026, 6, 20), [
            "has_event_feed", "has_trajectory", "trajectory_recorded_today",
        ]] = False
        after = build_features(frame, bag)
        pd.testing.assert_frame_equal(before, after)

    def test_frozen_split_rejects_label_change(self):
        reference = pd.DataFrame({
            "gpsno": [f"car_{i:03d}" for i in range(50)],
            "label": [i % 2 for i in range(50)],
        })
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            split = load_or_create_split(reference, path, seed=2026, folds=5, holdout_fraction=0.2)
            self.assertEqual(set(split["development"]) & set(split["holdout"]), set())
            self.assertEqual(sum(map(len, split["validation_folds"].values())), len(split["development"]))
            self.assertEqual(audit_split(reference, split)["status"], "passed")
            tampered = {**split, "holdout": split["holdout"] + [split["development"][0]]}
            with self.assertRaises(ValueError):
                audit_split(reference, tampered)
            altered = reference.copy()
            altered.loc[0, "label"] = 1 - altered.loc[0, "label"]
            with self.assertRaises(ValueError):
                load_or_create_split(altered, path, seed=2026, folds=5, holdout_fraction=0.2)

    def test_wrong_label_window_is_rejected(self):
        reference = pd.DataFrame({
            "gpsno": ["a", "b"], "anchor_date": [date(2026, 6, 20)] * 2,
            "label": [0, 1],
            "label_status": ["provisional_negative", "observed_positive"],
            "input_start": [date(2026, 6, 1)] * 2,
            "input_end": [date(2026, 6, 20)] * 2,
            "future_end": [date(2026, 7, 30)] * 2,
        })
        self.assertEqual(len(labeled_reference(reference)), 2)
        reference.loc[0, "input_end"] = date(2026, 6, 21)
        with self.assertRaises(ValueError):
            labeled_reference(reference)

    def test_training_artifacts_share_one_version_folder(self):
        from train_v1 import run

        with tempfile.TemporaryDirectory() as temp:
            db_root = Path(temp)
            input_dir = db_root / "processed"
            results_root = db_root / "results"
            input_dir.mkdir()
            records = []
            bags = []
            profile = []
            first = date(2026, 6, 1)
            for i in range(500):
                gpsno = f"car_{i:03d}"
                label = int(i % 3 == 0)
                profile.append({"gpsno": gpsno, "energy_type": "test"})
                bags.append({
                    "gpsno": gpsno, "anchor_date": date(2026, 6, 20), "label": label,
                    "label_status": "observed_positive" if label else "provisional_negative",
                    "input_start": date(2026, 6, 1), "input_end": date(2026, 6, 20),
                    "future_end": date(2026, 7, 30),
                })
                bags.append({
                    "gpsno": gpsno, "anchor_date": date(2026, 7, 30), "label": None,
                    "label_status": "target_unknown",
                    "input_start": date(2026, 7, 11), "input_end": date(2026, 7, 30),
                    "future_end": date(2026, 9, 8),
                })
                for offset in range(60):
                    feed = i % 23 != 0
                    records.append({
                        "gpsno": gpsno, "day": first + timedelta(days=offset),
                        "has_event_feed": feed, "has_trajectory": True,
                        "trajectory_recorded_today": True,
                        "event_11804_count": int(feed and label and offset in (9, 17)),
                        "event_11803_count": 0,
                        "event_total_count": int(feed and label and offset in (9, 17)),
                        "event_total_episodes": int(feed and label and offset in (9, 17)),
                        "distance_km": float(5 + i % 5), "drive_hours": 1.0,
                    })
            con = duckdb.connect()
            try:
                for filename, frame in (
                    ("daily_features.parquet", pd.DataFrame(records)),
                    ("bag_index.parquet", pd.DataFrame(bags)),
                    ("profile.parquet", pd.DataFrame(profile)),
                ):
                    con.register("frame_to_save", frame)
                    con.execute("COPY frame_to_save TO ? (FORMAT PARQUET)", [str(input_dir / filename)])
                    con.unregister("frame_to_save")
            finally:
                con.close()
            run_dir = run(ROOT / "tests" / "fixtures" / "v1_smoke.json", input_dir, results_root, db_root)
            self.assertTrue((run_dir / "source" / "train_v1.py").is_file())
            self.assertTrue((run_dir / "predictions" / "oof_predictions.parquet").is_file())
            self.assertTrue((run_dir / "predictions" / "submission_candidate.csv").is_file())
            self.assertTrue((run_dir / "models" / "development_full.cbm").is_file())
            audit = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "passed")
            self.assertTrue(audit["validation_appears_once"])
            self.assertEqual(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"], "completed")
            self.assertTrue((results_root / "leaderboard.csv").is_file())
            candidate = pd.read_csv(run_dir / "predictions" / "submission_candidate.csv")
            self.assertEqual(len(candidate), 500)
            self.assertTrue(candidate.probability.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
