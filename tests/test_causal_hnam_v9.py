import unittest

import torch

from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_METRICS
from ie_alpaca.models.causal_hnam_v9 import CausalHNAMV9


class CausalHNAMTest(unittest.TestCase):
    def test_all_hnam_variants_return_one_logit_per_landmark(self):
        events = torch.rand(3, 6, len(EVENT_CODES), len(EVENT_METRICS))
        present = torch.ones(3, 6, len(EVENT_CODES))
        context = torch.rand(3, 6, 34)
        for variant in ("e1a_flat", "e1b_hierarchical", "e1c_chronic_acute", "e1d_interactions"):
            net = CausalHNAMV9(34, variant=variant, learned_weights=True)
            output = net(events, present, context)
            self.assertEqual(output.shape, (3, 6))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(len(net.weight_report()), len(EVENT_CODES))
