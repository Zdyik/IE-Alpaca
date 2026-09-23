import unittest
from functools import partial

import torch

from ie_alpaca.models.event_hazard_v8 import EventHazardNetV8
from ie_alpaca.training.event_v4 import fit_network
from ie_alpaca.training.event_v5 import select_epochs
from tests.test_train_v5 import _pack


class V8DepthFusionContracts(unittest.TestCase):
    def test_architectures_have_expected_shapes_and_small_parameter_counts(self):
        event = torch.rand(3, 24, 9)
        present = torch.ones(3, 24)
        context = torch.rand(3, 34)
        counts = {}
        for architecture in ("deep_additive", "deep_fusion"):
            model = EventHazardNetV8(34, architecture=architecture, learned_weights=True)
            self.assertEqual(model(event, present, context).shape, (3,))
            self.assertEqual(len(model.weight_report()), 24)
            counts[architecture] = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(counts, {"deep_additive": 2000, "deep_fusion": 2383})

    def test_absent_events_do_not_change_prediction(self):
        for architecture in ("deep_additive", "deep_fusion"):
            torch.manual_seed(2026)
            model = EventHazardNetV8(34, architecture=architecture, learned_weights=True).eval()
            present = torch.zeros(2, 24)
            context = torch.rand(2, 34)
            with torch.inference_mode():
                baseline = model(torch.zeros(2, 24, 9), present, context)
                changed = model(torch.full((2, 24, 9), 1e6), present, context)
            self.assertTrue(torch.allclose(baseline, changed), architecture)

    def test_both_architectures_use_v5_inner_selection_and_refit(self):
        config = {"max_epochs": 3, "min_epochs": 2, "patience": 2, "min_delta": 0.0,
                  "dropout": .1, "learning_rate": .001, "weight_decay": .001,
                  "batch_size": 8, "type_penalty": .02}
        train, val = _pack("train", 20), _pack("val", 8)
        for architecture in ("deep_additive", "deep_fusion"):
            factory = partial(EventHazardNetV8, architecture=architecture)
            selected, _ = select_epochs(train, val, config, learned_weights=True,
                                        seed=2026, device=torch.device("cpu"),
                                        network_factory=factory)
            net, history = fit_network(
                train, {**config, "epochs": selected}, learned_weights=True,
                seed=2026, device=torch.device("cpu"), network_factory=factory,
            )
            self.assertEqual(len(history), selected)
            self.assertEqual(net.architecture, architecture)


if __name__ == "__main__":
    unittest.main()
