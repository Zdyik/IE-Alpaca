import unittest

import torch

from ie_alpaca.models.exposure_v9 import ExposureRiskNetV9


class ExposureModelTest(unittest.TestCase):
    def test_exposure_variants_return_valid_daily_hazards(self):
        inputs = torch.randn(4, 6, 20)
        for variant in ("e4a_direct", "e4b_auxiliary", "e4c_factorized"):
            net = ExposureRiskNetV9(20, variant=variant, dropout=.1,
                                    base_daily_hazard=.01, base_log_exposure=2.0)
            q, log_exposure = net(inputs)
            self.assertEqual(q.shape, (4, 6))
            self.assertEqual(log_exposure.shape, (4, 6))
            self.assertTrue(torch.all((q > 0) & (q < 1)))
            self.assertTrue(torch.all(log_exposure >= 0))
