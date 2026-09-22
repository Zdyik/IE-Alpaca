import unittest

import numpy as np
import pandas as pd
import torch

from ie_alpaca.training.event_v5 import inner_vehicle_split, select_epochs


def _pack(prefix: str, n: int) -> dict:
    rng = np.random.default_rng(n)
    events = rng.random((n, 6, 24, 9), dtype=np.float32)
    present = (rng.random((n, 6, 24)) > .5).astype(np.float32)
    event_day = np.zeros((n, 6), dtype=np.float32)
    event_day[::4, 1] = 5
    horizon = np.tile(np.array([40, 40, 30, 20, 10, 7], dtype=np.float32), (n, 1))
    return {"gpsno": [f"{prefix}{i}" for i in range(n)], "events": events,
            "present": present, "context": np.zeros((n, 6, 34), dtype=np.float32),
            "event_day": event_day, "horizon": horizon,
            "label": (event_day > 0).astype(np.int8)}


class V5EpochSelectionContracts(unittest.TestCase):
    def test_inner_split_uses_whole_vehicles_and_is_repeatable(self):
        ids = {f"v{i}" for i in range(40)}
        labels = pd.Series({f"v{i}": i % 2 for i in range(40)})
        train, val = inner_vehicle_split(ids, labels, seed=2026, validation_fraction=.2)
        self.assertFalse(train & val)
        self.assertEqual(train | val, ids)
        self.assertEqual(len(val), 8)
        self.assertEqual(set(labels.loc[list(val)]), {0, 1})
        self.assertEqual((train, val), inner_vehicle_split(ids, labels, seed=2026, validation_fraction=.2))

    def test_epoch_choice_uses_only_inner_pack(self):
        config = {"max_epochs": 4, "min_epochs": 2, "patience": 2, "min_delta": 0.0,
                  "dropout": .1, "learning_rate": .001, "weight_decay": .001,
                  "batch_size": 8, "type_penalty": .02}
        train, val = _pack("train", 20), _pack("val", 8)
        selected, history = select_epochs(train, val, config, learned_weights=True,
                                          seed=2026, device=torch.device("cpu"))
        self.assertTrue(1 <= selected <= len(history) <= 4)
        self.assertEqual(selected, 1 + np.argmin([x["inner_val_nll"] for x in history]))
        val["gpsno"][0] = train["gpsno"][0]
        with self.assertRaises(ValueError):
            select_epochs(train, val, config, learned_weights=True,
                          seed=2026, device=torch.device("cpu"))
