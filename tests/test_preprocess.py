import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import duckdb
import numpy as np
from openpyxl import Workbook

from make_model_inputs import build as build_model_inputs
from preprocess import parse_args, run


class PreprocessSmokeTest(unittest.TestCase):
    def test_label_boundaries_missing_feed_and_imu_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = source / "任务一预处理结果"
            source.mkdir()
            imu_dir = source / "3.IMU"
            trajectory_dir = source / "4.trajectory"
            imu_dir.mkdir()
            trajectory_dir.mkdir()

            book = Workbook()
            sheet = book.active
            sheet.append((
                "设备号", "能源类型", "月均行驶里程", "月均行驶时长", "月平均停留次数",
                "高速里程占比", "早晨行驶时长占比", "黄昏行驶时长占比", "夜间里程占比", "夜间行驶时长占比",
            ))
            for gpsno in (101, 102, 103, 104, 105, 106):
                sheet.append((gpsno, "电力", 1000, 50, 20, .1, .2, .2, .1, .1))
            book.save(source / "1.profile.xlsx")

            event_rows = [
                "101\t30\t114\t30002\t左车道偏移\t60\t2026-06-20 10:00:00",
                "101\t30\t114\t11804\t未遂事故\t50\t2026-06-21 10:00:00",
                "101\t30\t114\t11803\t事故\t50\t2026-07-31 10:00:00",
                "102\t30\t114\t30003\t右车道偏移\t40\t2026-06-19 10:00:00",
                "104\t30\t114\t30002\t左车道偏移\t40\t2026-07-31 11:00:00",
                "105\t30\t114\t11804\t未遂事故\t40\t2026-06-22 11:00:00",
                "106\t30\t114\t30002\t左车道偏移\t40\t2026-06-19 11:00:00",
            ]
            with zipfile.ZipFile(source / "2.events.zip", "w") as archive:
                archive.writestr("events.txt", "\n".join(event_rows) + "\n")

            trajectory_rows = [[], []]
            imu_rows = [[], []]
            for gpsno in (101, 102):
                for second in range(4):
                    ts = f"2026-06-20 10:00:0{second}"
                    part = second // 2
                    trajectory_rows[part].append(f"{gpsno}\t100\t1\t{ts}\t4\t90\t30\t114")
                    ems = "\\N" if second == 0 else "4"
                    imu_rows[part].append(
                        f"{gpsno}\tdevice-{gpsno}\t{ts}\t20260620\t{ems}\t4\t0.1\t0.0\t0.99\t1\t2\t3"
                    )
            for part in range(2):
                (trajectory_dir / f"part-{part:05d}").write_text("\n".join(trajectory_rows[part]) + "\n", encoding="utf-8")
                with gzip.open(imu_dir / f"part-{part:05d}.gz", "wt", encoding="utf-8") as stream:
                    stream.write("\n".join(imu_rows[part]) + "\n")

            run(parse_args(["--input", str(source), "--output", str(output), "--stage", "all", "--threads", "1"]))
            trajectory_mtime = (output / "trajectory_daily.parquet").stat().st_mtime_ns
            imu_mtime = (output / "imu_daily.parquet").stat().st_mtime_ns
            run(parse_args(["--input", str(source), "--output", str(output), "--stage", "trajectory", "--threads", "1", "--resume"]))
            run(parse_args(["--input", str(source), "--output", str(output), "--stage", "imu", "--threads", "1", "--resume"]))
            self.assertEqual((output / "trajectory_daily.parquet").stat().st_mtime_ns, trajectory_mtime)
            self.assertEqual((output / "imu_daily.parquet").stat().st_mtime_ns, imu_mtime)
            con = duckdb.connect()
            bags = con.execute(f"SELECT gpsno, anchor_date, label, label_status FROM read_parquet('{(output / 'bag_index.parquet').as_posix()}') WHERE anchor_date IN (DATE '2026-06-20', DATE '2026-07-30') ORDER BY gpsno,anchor_date").fetchall()
            self.assertEqual([x[2] for x in bags], [1, None, 0, None, None, None, None, None, 1, None, 0, None])
            self.assertEqual(bags[4][3], "unlabeled_no_event_feed")
            self.assertEqual(bags[6][3], "unlabeled_no_event_feed")
            report = json.loads((output / "quality_events.json").read_text(encoding="utf-8"))
            self.assertEqual(report["after_rule_end_rows"], 2)
            self.assertEqual(report["accident_rows"], 2)
            trajectory_report = json.loads((output / "quality_trajectory.json").read_text(encoding="utf-8"))
            imu_report = json.loads((output / "quality_imu.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory_report["cross_part_vehicles"], 2)
            self.assertEqual(imu_report["cross_part_vehicles"], 2)
            self.assertEqual(imu_report["valid_windows"], 2)
            imu = con.execute(f"SELECT ems_speed,gps_speed FROM read_parquet('{(output / 'imu_canonical' / '*.parquet').as_posix()}') WHERE gpsno='101' ORDER BY ts LIMIT 1").fetchone()
            self.assertIsNone(imu[0])
            self.assertEqual(imu[1], 4.0)
            daily_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{(output / 'daily_features.parquet').as_posix()}')").fetchone()[0]
            self.assertEqual(daily_count, 360)
            future_only_feed = con.execute(f"""
                SELECT day, has_event_feed, event_total_count
                FROM read_parquet('{(output / 'daily_features.parquet').as_posix()}')
                WHERE gpsno='105' AND day IN (DATE '2026-06-20', DATE '2026-06-22')
                ORDER BY day
            """).fetchall()
            self.assertEqual([(row[1], row[2]) for row in future_only_feed], [(False, None), (True, 1)])
            model_output = output / "model_inputs"
            summary = build_model_inputs(output, model_output, 2, 2026)
            self.assertEqual(summary["eligible_vehicles"], 4)
            self.assertEqual(summary["submission_bags"], 6)
            with np.load(model_output / "final_submission.npz") as pack:
                self.assertEqual(pack["x"].shape[0], 6)
                self.assertEqual(pack["x"].shape[1], 20)
                self.assertTrue(np.all(pack["label"] == -1))
            for fold in range(2):
                with np.load(model_output / f"fold_{fold}_train.npz") as train, np.load(model_output / f"fold_{fold}_val.npz") as val:
                    self.assertFalse(set(train["gpsno"]) & set(val["gpsno"]))
                    self.assertEqual(len(val["gpsno"]), 2)


if __name__ == "__main__":
    unittest.main()
