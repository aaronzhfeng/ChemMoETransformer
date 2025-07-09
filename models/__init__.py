from models.moe_layer import MoEFFN          # noqa: F401
from models.chemtransformer_model import ChemMoETransformer

def build_model(cfg, vocab_size: int) -> ChemMoETransformer:
    return ChemMoETransformer(
        vocab_size=vocab_size,
        n_layer=cfg.model.n_layer,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        moe_layers=list(cfg.model.moe_layers),
        num_experts=cfg.model.num_experts,
    )
