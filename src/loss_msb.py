"""
MSB-weighted digit cross-entropy for HRM-Sapient.

Numbers are encoded MSB-first with optional NEG token at position 0:
    400  -> ['4', '0', '0', EOS, PAD, PAD, PAD, PAD]
    1840 -> ['1', '8', '4', '0', EOS, PAD, PAD, PAD]

The training failures observed at epoch 500 are *leading-digit* errors
(100 vs 400, 1840 vs 1200, 124 vs 624). The standard mean cross-entropy
treats every position equally, so a wrong leading digit and a wrong
trailing digit contribute the same loss — which is wrong: the leading
digit determines the magnitude of the answer.

This loss puts more weight on the first non-PAD digits, then decays.
"""
import torch
import torch.nn.functional as F

DIGIT_PAD = 0
MAX_DIGITS = 8


def msb_position_weights(max_digits: int = MAX_DIGITS,
                         w_leading: float = 3.0,
                         w_tail: float = 1.0,
                         decay: float = 0.6) -> torch.Tensor:
    """
    Geometric decay from leading position to trailing.

    Defaults give weights:  [3.00, 2.20, 1.72, 1.43, 1.26, 1.16, 1.09, 1.05]
    => leading digit loss is ~3x trailing.
    """
    w = []
    for i in range(max_digits):
        w.append(w_tail + (w_leading - w_tail) * (decay ** i))
    return torch.tensor(w, dtype=torch.float32)


def final_node_digit_loss_msb(digit_logits: torch.Tensor,
                              final_digit_tgt: torch.Tensor,
                              num_real_nodes: torch.Tensor,
                              pos_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    digit_logits   : (B, N, D, V)
    final_digit_tgt: (B, D)
    num_real_nodes : (B,)
    pos_weights    : (D,) — leading-digit-heavier weights. If None, uses default.

    Returns scalar loss.
    """
    B, N, D, V = digit_logits.shape

    last_idx = (num_real_nodes - 1).clamp(0, N - 1)
    idx = last_idx.view(B, 1, 1, 1).expand(B, 1, D, V)
    final_logits = digit_logits.gather(1, idx).squeeze(1)            # (B, D, V)

    pad_mask = (final_digit_tgt != DIGIT_PAD).float()                # (B, D)

    if pos_weights is None:
        pos_weights = msb_position_weights(D).to(digit_logits.device)
    else:
        pos_weights = pos_weights.to(digit_logits.device)
    pw = pos_weights.view(1, D)                                      # (1, D)

    log_probs = F.log_softmax(final_logits, dim=-1)
    tgt_lp = log_probs.gather(-1, final_digit_tgt.unsqueeze(-1)).squeeze(-1)  # (B, D)

    weighted = -tgt_lp * pad_mask * pw
    denom = (pad_mask * pw).sum().clamp(min=1.0)
    return weighted.sum() / denom


if __name__ == "__main__":
    w = msb_position_weights()
    print("MSB position weights:", [f"{x:.3f}" for x in w.tolist()])
    print("Leading vs trailing ratio:", f"{w[0].item() / w[-1].item():.2f}x")
