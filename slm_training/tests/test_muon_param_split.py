"""The Muon/AdamW parameter split must cover every trainable tensor exactly once.

torch.optim.Muon raises on any 1-D tensor, so a routing slip is a hard crash
several minutes into a GPU job rather than a silent slowdown.

    python slm_training/tests/test_muon_param_split.py
"""
import torch
import torch.nn as nn

EMB_KEYS = ("emb", "wte", "wpe", "lm_head", "head.weight")


def split(model):
    """Mirrors the routing in run_small_lm.py."""
    muon, adamw_decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "ln" in name.lower() or "norm" in name.lower():
            no_decay.append((name, p))
        elif any(k in name.lower() for k in EMB_KEYS):
            adamw_decay.append((name, p))
        else:
            muon.append((name, p))
    return muon, adamw_decay, no_decay


class _TiedGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 512)
        self.wpe = nn.Embedding(256, 512)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(512, 2048), nn.LayerNorm(2048), nn.Linear(2048, 512))
            for _ in range(6)
        )
        self.ln_f = nn.LayerNorm(512)
        self.lm_head = nn.Linear(512, 50257, bias=False)
        self.lm_head.weight = self.wte.weight  # tied


def main():
    m = _TiedGPT()
    muon, adamw_decay, no_decay = split(m)

    assert all(p.ndim >= 2 for _, p in muon), "torch.optim.Muon rejects 1-D tensors"
    ids = lambda g: {id(p) for _, p in g}
    assert not (ids(muon) & ids(adamw_decay)), "tensor routed to two optimizers"
    assert ids(muon) | ids(adamw_decay) | ids(no_decay) == {id(p) for p in m.parameters()}, \
        "some trainable tensor is not routed to any optimizer"
    assert not any(any(k in n.lower() for k in EMB_KEYS) for n, _ in muon), \
        "embedding/head leaked into Muon"
    assert muon, "nothing routed to Muon"

    # the real thing must accept the group
    torch.optim.Muon([p for _, p in muon], lr=0.02)
    print(f"ok: muon={len(muon)} adamw_decay={len(adamw_decay)} no_decay={len(no_decay)}")


if __name__ == "__main__":
    main()
