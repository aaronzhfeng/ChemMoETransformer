from types import SimpleNamespace

import torch

from models import build_model


def _mini_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            n_layer=2,
            d_model=64,
            n_head=4,
            d_ff=128,
            dropout=0.1,
            moe_layers=[2],  # only 2nd layer is MoE
            num_experts=2,
        ),
    )


def test_forward_end_to_end():
    cfg = _mini_cfg()
    vocab_size = 20
    model = build_model(cfg, vocab_size)

    B = 2
    src = torch.randint(4, vocab_size, (5, B))
    tgt_in = torch.randint(4, vocab_size, (7, B))
    pad_mask_src = src.eq(0).t()
    pad_mask_tgt = tgt_in.eq(0).t()

    logits, aux = model(src, tgt_in, pad_mask_src, pad_mask_tgt)
    assert logits.shape == (7, B, vocab_size)
    assert aux >= 0
