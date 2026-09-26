import unittest

import numpy as np
import torch

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.daily_v11 import EVENT_BUNDLE_COLUMNS, apply_masks, event_bundle_mask
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.representation_v11 import RepresentationEncoderV11, RiskNetV11
from ie_alpaca.models.ssl_heads_v11 import JEPAV11


class V11Contracts(unittest.TestCase):
    def test_event_formula_bundle_is_masked_together(self):
        rng = np.random.default_rng(7); events = np.ones((2, 20, len(EVENT_CODES), len(EVENT_FEATURE_NAMES)), np.float32)
        context = np.ones((2, 20, len(CONTEXT_COLUMNS)), np.float32)
        mask = event_bundle_mask(events.shape[:3], rng); cm = np.zeros(context.shape, bool)
        altered, _ = apply_masks(events, context, mask, cm)
        for column in EVENT_BUNDLE_COLUMNS:
            self.assertTrue(np.all(altered[..., column][mask] == 0))
        self.assertTrue(np.all(altered[..., 5:][mask] == 1))

    def test_anchor_is_prefix_invariant(self):
        torch.manual_seed(8); net = RiskNetV11(len(CONTEXT_COLUMNS), dropout=0).eval()
        event = torch.rand(2, 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES)); context = torch.rand(2, 60, len(CONTEXT_COLUMNS))
        changed_event, changed_context = event.clone(), context.clone(); changed_event[:, 20:] = 999; changed_context[:, 20:] = 999
        with torch.inference_mode():
            a = net(event, context, (20,))[0]; b = net(changed_event, changed_context, (20,))[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_jepa_target_has_no_gradient_and_student_prefix_isolated(self):
        encoder = RepresentationEncoderV11(len(CONTEXT_COLUMNS)); model = JEPAV11(encoder, 32).eval()
        self.assertTrue(all(not p.requires_grad for p in model.target.parameters()))
        event = torch.rand(2, 20, len(EVENT_CODES), len(EVENT_FEATURE_NAMES)); context = torch.rand(2, 20, len(CONTEXT_COLUMNS))
        changed_event = event.clone(); changed_event[:, 10:] = 1000
        with torch.no_grad():
            a = model.online(event[:, :10], context[:, :10])["public"]
            b = model.online(changed_event[:, :10], context[:, :10])["public"]
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__": unittest.main()
