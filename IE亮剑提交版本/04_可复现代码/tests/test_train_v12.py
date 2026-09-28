import unittest

import numpy as np
import torch

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.daily_v12 import (
    PRIMARY_STATE, RISK_GROUP_POSITIONS, STATE_EVENT_CODES, STATE_NAMES,
    shuffled_primary_state, state_block_mask, state_masks_to_inputs,
)
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.state_encoder_v12 import RiskNetV12, TransitionPredictorV12
from ie_alpaca.models.state_mae_v12 import SSLModelV12
from ie_alpaca.training.pretrain_v12 import _future_targets, pretrain_v12


class V12Contracts(unittest.TestCase):
    def test_state_groups_partition_every_event_once(self):
        flattened = [code for codes in STATE_EVENT_CODES.values() for code in codes]
        self.assertEqual(len(flattened), len(EVENT_CODES))
        self.assertEqual(set(flattened), set(EVENT_CODES))
        shuffled = shuffled_primary_state(9)
        self.assertEqual(np.bincount(shuffled, minlength=6).tolist(),
                         np.bincount(PRIMARY_STATE, minlength=6).tolist())

    def test_state_block_expands_to_complete_event_groups(self):
        rng = np.random.default_rng(7)
        state = state_block_mask((3, 20, len(STATE_NAMES)), rng)
        event, context = state_masks_to_inputs(state)
        self.assertEqual(event.shape, (3, 20, len(EVENT_CODES)))
        self.assertEqual(context.shape, (3, 20, len(CONTEXT_COLUMNS)))
        for event_index, state_index in enumerate(PRIMARY_STATE):
            self.assertTrue(np.array_equal(event[..., event_index], state[..., state_index]))

    def test_future_ssl_targets_exclude_severe_events(self):
        events = torch.zeros(2, 12, len(EVENT_CODES), len(EVENT_FEATURE_NAMES))
        severe_positions = [EVENT_CODES.index(11803), EVENT_CODES.index(11804)]
        events[:, 5, severe_positions, 4] = 1
        occurrence, magnitude = _future_targets(events, 7)
        self.assertEqual(tuple(occurrence.shape), (2, 5, len(RISK_GROUP_POSITIONS)))
        self.assertTrue(torch.equal(occurrence, torch.zeros_like(occurrence)))
        self.assertTrue(torch.equal(magnitude, torch.zeros_like(magnitude)))

    def test_risk_prediction_is_prefix_invariant(self):
        torch.manual_seed(3)
        net = RiskNetV12(len(CONTEXT_COLUMNS), "hrc", dropout=0).eval()
        events = torch.rand(2, 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES))
        context = torch.rand(2, 60, len(CONTEXT_COLUMNS))
        changed_events, changed_context = events.clone(), context.clone()
        changed_events[:, 20:] = 999
        changed_context[:, 20:] = 999
        with torch.inference_mode():
            before = net(events, context, (20,))[0]
            after = net(changed_events, changed_context, (20,))[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_teacher_has_no_gradient_and_chain_controls_match(self):
        model = SSLModelV12(len(CONTEXT_COLUMNS), "hrc")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.target.parameters()))
        counts = []
        for mode in ("hrc", "hrc_no_prior", "hrc_shuffle"):
            predictor = TransitionPredictorV12(32, mode)
            counts.append(sum(parameter.numel() for parameter in predictor.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_one_step_pretraining_and_downstream_shapes(self):
        rng = np.random.default_rng(11)
        events = rng.random((8, 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES)), dtype=np.float32)
        events[..., 4] = (events[..., 4] > .85).astype(np.float32)
        context = rng.normal(size=(8, 60, len(CONTEXT_COLUMNS))).astype(np.float32)
        pack = {"events": events, "context": context}
        config = {"width": 16, "dropout": 0.0, "ssl_learning_rate": 3e-4,
                  "ssl_weight_decay": 1e-3, "ssl_batch_size": 4, "ssl_check_every": 1}
        state, history, diagnostics = pretrain_v12(pack, config, "hrc", 1, 19, torch.device("cpu"))
        self.assertEqual(len(history), 1)
        self.assertEqual(set(diagnostics["per_state"]), set(STATE_NAMES))
        net = RiskNetV12(len(CONTEXT_COLUMNS), "hrc", width=16, dropout=0)
        net.backbone.load_state_dict(state)
        with torch.inference_mode():
            logits, auxiliary, details = net(torch.as_tensor(events[:2]), torch.as_tensor(context[:2]), (20, 53))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(auxiliary.shape), (2, 60, 3))
        self.assertEqual(tuple(details["edge_weights"].shape), (3, 3, 3))


if __name__ == "__main__":
    unittest.main()

