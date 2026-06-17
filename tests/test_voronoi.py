from __future__ import annotations

import torch

from baselines.common.voronoi import voronoi_fill


def test_voronoi_fill_multi_channel():
    masked = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(2, 4, 4)
    masked[:, :, 0, 0] = torch.tensor([1.0, 2.0])
    masked[:, :, 3, 3] = torch.tensor([3.0, 4.0])
    mask[:, 0, 0] = 1
    mask[:, 3, 3] = 1
    filled = voronoi_fill(masked, mask)
    assert filled.shape == masked.shape
    assert torch.allclose(filled[0, :, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(filled[0, :, 3, 3], torch.tensor([3.0, 4.0]))

