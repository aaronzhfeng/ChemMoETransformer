import torch

from models.moe_layer import MoEFFN


def test_moe_shapes_and_loss():
    ffn = MoEFFN(d_model=64, d_ff=128, n_experts=4)
    x = torch.randn(10, 3, 64)  # (L=10, B=3, D=64)
    y = ffn(x)
    assert y.shape == x.shape
    # auxiliary loss should be positive
    assert ffn.aux_loss.item() >= 0.0
