import unittest
from functools import partial

import torch

from ie_alpaca.models.event_hazard_v4 import EventHazardNet
from ie_alpaca.models.event_hazard_v6 import EventHazardNetV6
from ie_alpaca.training.event_v4 import fit_network
from ie_alpaca.training.event_v5 import select_epochs
from tests.test_train_v5 import _pack


class V6CapacityContracts(unittest.TestCase):
    def test_only_hidden_widths_change_parameter_count(self):
        base = EventHazardNet(34, learned_weights=True)
        medium = EventHazardNetV6(34, learned_weights=True, dropout=.1,
                                  base_daily_hazard=.01, event_hidden=16, context_hidden=32)
        large = EventHazardNetV6(34, learned_weights=True, dropout=.1,
                                 base_daily_hazard=.01, event_hidden=32, context_hidden=64)
        count = lambda net: sum(x.numel() for x in net.parameters())
        self.assertEqual([count(x) for x in (base, medium, large)], [696, 1360, 2688])
        self.assertEqual(set(base.state_dict()), set(medium.state_dict()))
        event = torch.zeros(2, 24, 9)
        present = torch.zeros(2, 24)
        context = torch.zeros(2, 34)
        self.assertEqual(medium(event, present, context).shape, (2,))

    def test_inner_selection_and_refit_use_requested_width(self):
        config = {"max_epochs": 3, "min_epochs": 2, "patience": 2, "min_delta": 0.0,
                  "dropout": .1, "learning_rate": .001, "weight_decay": .001,
                  "batch_size": 8, "type_penalty": .02}
        factory = partial(EventHazardNetV6, event_hidden=16, context_hidden=32)
        train, val = _pack("train", 20), _pack("val", 8)
        selected, _ = select_epochs(train, val, config, learned_weights=True,
                                    seed=2026, device=torch.device("cpu"), network_factory=factory)
        net, history = fit_network(train, {**config, "epochs": selected}, learned_weights=True,
                                   seed=2026, device=torch.device("cpu"), network_factory=factory)
        self.assertEqual(sum(x.numel() for x in net.parameters()), 1360)
        self.assertEqual(len(history), selected)
