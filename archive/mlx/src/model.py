"""
HRMForMath — Hierarchical Reasoning Model for GSM8K.

Faithful implementation of the HRM paper's architecture adapted for math:
  1. GraphAwareBridge (GAT): op tokens + numerical args → embeddings (B, N, D)
  2. H-level (planner): slow, abstract reasoning — updates global context
  3. L-level (executor): fast, detailed computation — processes each step
  4. Digit classification head: predicts each digit of the answer (like Sudoku)
  5. ACT Q-head: adaptive halting (learns when to stop reasoning)

Key design choices (matching the paper):
  - 1-step gradient approximation: only the final H/L iteration gets gradients
  - All inner H/L iterations use stop_gradient for O(1) memory
  - Cross-entropy over digit tokens (not regression!)
  - Input injection: L_level receives z_H + input_embeddings at each iteration
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional
import math

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, CastedLinear, CastedEmbedding
)
from src.graph_encoder import GraphAwareBridge
from src.dataset import DIGIT_VOCAB_SIZE, MAX_DIGITS


# ─── HRM Reasoning Module (from paper) ─────────────────────────────────────

class HRMBlock(nn.Module):
    """
    Single transformer block used inside H-level and L-level.
    Post-norm (add-then-norm) as in the original HRM paper.
    """
    def __init__(self, d_model: int, n_heads: int, expansion: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rms_norm_eps = 1e-5
        
        self.self_attn = Attention(
            hidden_size=d_model,
            head_dim=d_model // n_heads,
            num_heads=n_heads,
            num_key_value_heads=n_heads,
            causal=False  # Non-causal — HRM uses bidirectional attention
        )
        self.mlp = SwiGLU(d_model, expansion)
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:
        # Post-norm: residual + norm (paper-faithful)
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=None, hidden_states=hidden_states, attention_mask=attention_mask),
            variance_epsilon=self.rms_norm_eps
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.rms_norm_eps
        )
        return hidden_states


class HRMReasoningModule(nn.Module):
    """
    Reasoning module — processes sequences through multiple transformer blocks.
    Used for both H-level (planner) and L-level (executor).
    
    The input_injection is added to hidden_states before processing,
    exactly as in the paper (line 94: hidden_states = hidden_states + input_injection).
    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, expansion: float = 2.0):
        super().__init__()
        self.layers = [HRMBlock(d_model, n_heads, expansion) for _ in range(n_layers)]
    
    def __call__(self, hidden_states: mx.array, input_injection: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Input injection (add) — paper line 94
        hidden_states = hidden_states + input_injection
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        return hidden_states


# ─── Main Model ─────────────────────────────────────────────────────────────

class HRMForMath(nn.Module):
    """
    Full HRM adapted for GSM8K math reasoning.
    
    Architecture:
      Bridge(node_ids, values, adj) → embeddings (B, N, D)
                ↓
      H_level(z_H, z_L)  ←→  L_level(z_L, z_H + embeddings)
           [H_cycles × L_cycles, with stop_gradient]
                ↓
      digit_head(z_H) → (B, N, MAX_DIGITS, VOCAB) → per-digit classification
    """
    def __init__(
        self,
        vocab_size: int = 9,       # OP_VOCAB size
        d_model: int = 256,
        n_heads: int = 8,
        H_cycles: int = 2,        # Number of high-level reasoning iterations
        L_cycles: int = 4,        # Number of low-level iterations per H-cycle
        H_layers: int = 4,        # Transformer blocks in H-level
        L_layers: int = 4,        # Transformer blocks in L-level
        expansion: float = 2.0,
        seq_len: int = 50,        # Max nodes
        max_digits: int = MAX_DIGITS,
        digit_vocab: int = DIGIT_VOCAB_SIZE,
        **kwargs,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.max_digits = max_digits
        self.digit_vocab = digit_vocab
        self.embed_scale = math.sqrt(d_model)
        
        # 1. Graph-Aware Bridge: encodes op types + numerical args via GAT
        self.bridge = GraphAwareBridge(
            vocab_size=vocab_size,
            d_model=d_model,
            num_value_features=2,
            gat_hidden=128,
            gat_layers=3,
            heads=4,
        )
        
        # 2. Positional encoding (learned)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_pos = CastedEmbedding(
            seq_len, d_model, init_std=embed_init_std, cast_to=mx.float32
        )
        
        # 3. H-level reasoning (planner) — slow, abstract
        self.H_level = HRMReasoningModule(H_layers, d_model, n_heads, expansion)
        
        # 4. L-level reasoning (executor) — fast, detailed
        self.L_level = HRMReasoningModule(L_layers, d_model, n_heads, expansion)
        
        # 5. Initial states (learned, from paper)
        self.H_init = trunc_normal_init_(shape=(d_model,), std=1.0)
        self.L_init = trunc_normal_init_(shape=(d_model,), std=1.0)
        
        # 6. Digit classification head (replaces regression!)
        # Maps d_model → max_digits * digit_vocab
        self.digit_head = CastedLinear(d_model, max_digits * digit_vocab, bias=True)
        
        # 7. ACT Q-head: [halt_logit, continue_logit]
        self.q_head = CastedLinear(d_model, 2, bias=True)
        # Init Q-head to favor continuing (paper: bias = [-5, -5])
        self.q_head.weight = mx.zeros_like(self.q_head.weight)
        self.q_head.bias = mx.array([-5.0, -5.0])
        
        self._is_training = True
    
    def train(self):
        self._is_training = True
        return self
    
    def eval(self):
        self._is_training = False
        return self
    
    def __call__(self, batch: Dict[str, mx.array]) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass.
        
        Returns:
            digit_logits: (B, N, MAX_DIGITS, VOCAB) — per-node digit predictions
            q_halt: (B,) — halt logits for ACT
            q_continue: (B,) — continue logits for ACT
        """
        inputs = batch["inputs"]          # (B, N) — op token IDs
        node_values = batch["node_values"]  # (B, N, 2) — log1p args
        adj_mask = batch["adj_mask"]        # (B, N, N) — adjacency
        
        B, N = inputs.shape
        
        # ── Step 1: Bridge Encoding ───────────────────────────────────────────
        # GAT processes the computation graph → node embeddings
        graph_ctx = self.bridge(inputs, node_values, adj_mask)
        
        # Add positional embeddings (node order = execution order)
        pos_emb = self.embed_pos.embedding_weight.astype(mx.float32)  # (seq_len, D)
        input_embeddings = 0.707106781 * (graph_ctx + pos_emb[None, :N, :])
        input_embeddings = self.embed_scale * input_embeddings  # (B, N, D)
        
        # Create attention mask to ignore PAD nodes
        # Shape: (B, 1, 1, N) so it broadcasts across heads and query positions
        pad_mask = (inputs == 0)  # OP_VOCAB["PAD"] is 0
        attention_mask = mx.where(pad_mask[:, None, None, :], mx.array(-1e9), mx.array(0.0))
        
        # ── Step 2: HRM H/L Reasoning (paper-faithful) ───────────────────────
        # Initialize reasoning states
        z_H = mx.broadcast_to(self.H_init, (B, N, self.d_model))
        z_L = mx.broadcast_to(self.L_init, (B, N, self.d_model))
        
        # Inner loops with stop_gradient — 1-step grad approximation
        # Paper: "We detach all but the final iteration from the computation graph"
        for h_step in range(self.H_cycles):
            for l_step in range(self.L_cycles):
                # All iterations except the very last get stop_gradient
                if not (h_step == self.H_cycles - 1 and l_step == self.L_cycles - 1):
                    z_L = mx.stop_gradient(
                        self.L_level(z_L, z_H + input_embeddings, attention_mask=attention_mask)
                    )
            
            if h_step != self.H_cycles - 1:
                z_H = mx.stop_gradient(
                    self.H_level(z_H, z_L, attention_mask=attention_mask)
                )
        
        # Final iteration WITH gradient (this is where learning happens)
        z_L = self.L_level(z_L, z_H + input_embeddings, attention_mask=attention_mask)
        z_H = self.H_level(z_H, z_L, attention_mask=attention_mask)
        
        # ── Step 3: Digit Classification ──────────────────────────────────────
        # z_H contains the final reasoning state for each node
        digit_flat = self.digit_head(z_H)  # (B, N, max_digits * vocab)
        digit_logits = digit_flat.reshape(B, N, self.max_digits, self.digit_vocab)
        
        # ── Step 4: ACT Q-values ─────────────────────────────────────────────
        # Use first token's representation for halt decision (paper convention)
        q_logits = self.q_head(z_H[:, 0]).astype(mx.float32)  # (B, 2)
        q_halt = q_logits[..., 0]      # (B,)
        q_continue = q_logits[..., 1]  # (B,)
        
        return digit_logits, q_halt, q_continue
