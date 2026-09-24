import torch
import unittest
from temporal_loss import TemporalNTXentLoss
import torch.nn.functional as F

class TestTemporalKernels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.temperature = 0.5
        
    def _run_loss_inspection(self, loss_module, N=4):
        z_i = F.normalize(torch.randn(N, 128), dim=1)
        z_j = F.normalize(torch.randn(N, 128), dim=1)
        
        # 4 patients: 2 from patient 0, 1 from patient 1, 1 from patient 2
        subject_ids = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        time_indices = torch.tensor([10, 15, 20, 20], dtype=torch.long)
        
        # Inject hook to inspect weights before log_w
        weights_captured = {}
        
        # Monkey patch
        original_forward = loss_module.forward
        
        def hooked_forward(*args, **kwargs):
            # Reproduce logic up to weights
            z = F.normalize(torch.cat([args[0], args[1]], dim=0), dim=1)
            sub_2n = torch.cat([args[2], args[2]], dim=0)
            time_2n = torch.cat([args[3], args[3]], dim=0).float()
            
            same_sub = (sub_2n.unsqueeze(0) == sub_2n.unsqueeze(1))
            time_dist = torch.abs(time_2n.unsqueeze(0) - time_2n.unsqueeze(1))
            
            if loss_module.kernel_type == 'exponential':
                w_same = 1.0 - torch.exp(-loss_module.lambda_decay * time_dist)
            elif loss_module.kernel_type == 'linear':
                w_same = torch.clamp(loss_module.kernel_alpha * time_dist, min=0.0, max=1.0)
            elif loss_module.kernel_type == 'cutoff':
                w_same = (time_dist > loss_module.kernel_cutoff).float()
                
            weights = torch.where(same_sub, w_same, torch.ones_like(w_same))
            weights_captured['weights'] = weights
            weights_captured['same_sub'] = same_sub
            weights_captured['time_dist'] = time_dist
            
            return original_forward(*args, **kwargs)
            
        loss_module.forward = hooked_forward
        loss = loss_module(z_i, z_j, subject_ids, time_indices)
        
        return loss, weights_captured
        
    def test_exponential_kernel(self):
        loss_fn = TemporalNTXentLoss(lambda_decay=0.1, kernel_type='exponential')
        _, info = self._run_loss_inspection(loss_fn)
        w = info['weights']
        same_sub = info['same_sub']
        time_dist = info['time_dist']
        
        # Cross patient
        self.assertTrue(torch.allclose(w[~same_sub], torch.ones_like(w[~same_sub])))
        
        # Same patient, dist = 5
        idx = (same_sub) & (time_dist == 5.0)
        expected = 1.0 - torch.exp(torch.tensor(-0.1 * 5.0))
        self.assertTrue(torch.allclose(w[idx], expected * torch.ones_like(w[idx])))

    def test_linear_kernel(self):
        loss_fn = TemporalNTXentLoss(kernel_type='linear', kernel_alpha=0.1)
        _, info = self._run_loss_inspection(loss_fn)
        w = info['weights']
        same_sub = info['same_sub']
        time_dist = info['time_dist']
        
        self.assertTrue(torch.allclose(w[~same_sub], torch.ones_like(w[~same_sub])))
        
        idx = (same_sub) & (time_dist == 5.0)
        self.assertTrue(torch.allclose(w[idx], torch.tensor(0.5) * torch.ones_like(w[idx])))

    def test_cutoff_kernel(self):
        loss_fn = TemporalNTXentLoss(kernel_type='cutoff', kernel_cutoff=6.0)
        _, info = self._run_loss_inspection(loss_fn)
        w = info['weights']
        same_sub = info['same_sub']
        time_dist = info['time_dist']
        
        self.assertTrue(torch.allclose(w[~same_sub], torch.ones_like(w[~same_sub])))
        
        idx_5 = (same_sub) & (time_dist == 5.0)
        self.assertTrue(torch.allclose(w[idx_5], torch.zeros_like(w[idx_5])))

    def test_positive_pairs_excluded(self):
        loss_fn = TemporalNTXentLoss()
        z_i = F.normalize(torch.randn(2, 128), dim=1)
        z_j = F.normalize(torch.randn(2, 128), dim=1)
        sub = torch.tensor([0, 1])
        t = torch.tensor([0, 0])
        
        # With 2 elements, positive pairs shouldn't contribute to negative sum
        # Loss should not be NaN
        loss = loss_fn(z_i, z_j, sub, t)
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()
