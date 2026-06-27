"""Generate a top-level architecture diagram for HRM-v3.1.

Shows each component as a labelled box with:
  - input type + shape
  - process
  - output type + shape

A worked example ("If a tree has 47 apples and 9 are removed, then 5 are
added back, how many remain?") flows through the diagram so the reader
can see concrete values at each stage.

Run:  python3 build_arch_diagram.py
Output: hrm_architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ===================================================================
# Colors
COL_INPUT   = "#FFE8CC"   # peach
COL_PARSE   = "#FFD6A5"
COL_BRIDGE  = "#CDEAFA"   # light blue
COL_HRM     = "#D6F5D6"   # light green
COL_ACT     = "#F8D7F0"   # pink
COL_HEAD    = "#FFF3B0"   # yellow
COL_OUTPUT  = "#E0E0E0"   # gray
COL_BORDER  = "#444"
COL_EXAMPLE = "#7A4D00"   # brown

PROCESS_COL = "#1A4F8C"
SHAPE_COL   = "#666"

fig, ax = plt.subplots(figsize=(18, 24))
ax.set_xlim(0, 100)
ax.set_ylim(0, 130)
ax.axis("off")


def box(x, y, w, h, color, label, process, shape_in, shape_out, example):
    """Draw a labelled box with input shape, process, output shape, and example."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                       facecolor=color, edgecolor=COL_BORDER, linewidth=1.5)
    ax.add_patch(p)
    cx = x + w / 2
    # Title
    ax.text(cx, y + h - 1.0, label, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#222")
    # Input shape
    ax.text(x + 1.2, y + h - 3.5, f"in:  {shape_in}",
            ha="left", va="top", fontsize=8.5, color=SHAPE_COL,
            family="monospace")
    # Process
    ax.text(x + 1.2, y + h - 5.0, process,
            ha="left", va="top", fontsize=8.8, color=PROCESS_COL,
            wrap=True)
    # Output shape
    ax.text(x + 1.2, y + 2.6, f"out: {shape_out}",
            ha="left", va="bottom", fontsize=8.5, color=SHAPE_COL,
            family="monospace")
    # Example
    if example:
        ax.text(x + 1.2, y + 0.8, example,
                ha="left", va="bottom", fontsize=8.0, color=COL_EXAMPLE,
                style="italic", family="monospace")


def arrow(x1, y1, x2, y2, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=20,
                        linewidth=1.8, color="#444")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, label,
                fontsize=8.5, color="#222", style="italic")


def section_band(y, label, color):
    """Horizontal band marking a major section."""
    ax.add_patch(FancyBboxPatch((1, y), 98, 1.5, boxstyle="round,pad=0",
                                 facecolor=color, edgecolor="none", alpha=0.5))
    ax.text(2, y + 0.75, label, ha="left", va="center",
            fontsize=10, fontweight="bold", color="#111")


# ===================================================================
# TITLE
ax.text(50, 128.5, "HRM-v3.1 Architecture — Input → Process → Output",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#111")
ax.text(50, 126.5,
        "Worked example: \"Tree has 47 apples, 9 removed, then 5 added. How many remain?\"",
        ha="center", va="center", fontsize=10, style="italic", color=COL_EXAMPLE)

# ===================================================================
# 1. INPUT — Symbolic Trace (the parsed math problem)
section_band(122, "STAGE 1 — INPUT  (parsed symbolic trace)", COL_INPUT)

box(15, 110, 70, 10, COL_INPUT,
    "Parsed Math Trace (JSON)",
    "A list of operation steps + final answer key.\n"
    "Each step has: op, arg1, arg2, result, result_value.\n"
    "arg1/arg2 are either literal numbers or string\n"
    "references to earlier results (e.g. \"v0\").",
    "JSON dict",
    "JSON dict",
    "{\"steps\": [{\"op\":\"sub\", \"arg1\":47, \"arg2\":9, \"result\":\"v0\", \"result_value\":38},\n"
    "           {\"op\":\"add\", \"arg1\":\"v0\", \"arg2\":5, \"result\":\"v1\", \"result_value\":43}],\n"
    " \"final_answer\": \"v1\"}     # target = 43")

arrow(50, 110, 50, 107)

# ===================================================================
# 2. PARSE_GRAPH — Convert to tensors
section_band(104, "STAGE 2 — TENSORIZE  (parse_graph)", COL_PARSE)

box(8, 86, 38, 16, COL_PARSE,
    "Node IDs & Operation Types",
    "Map each step's op to a small int via\n"
    "OP_VOCAB. Final 'final_answer' node appended.\n"
    "Pad to max_nodes=50 with PAD=0.",
    "JSON steps",
    "LongTensor [50]",
    "node_ids = [2, 1, 8, 0, 0, ...]\n"
    "          [sub, add, final, PAD, PAD, ...]")

box(54, 86, 38, 16, COL_PARSE,
    "Node Values & Adjacency",
    "Per node: 4 features\n"
    "  [log(|arg1|)·sign, is_ref1,\n"
    "   log(|arg2|)·sign, is_ref2]\n"
    "Adjacency: 50x50 mask, edge if arg refs an\n"
    "earlier node's result (self-loop always on).",
    "JSON steps",
    "Float [50, 4]; Float [50, 50]",
    "node_values[0] = [log(47), 0, log(9), 0]\n"
    "adj[0,1] = 1.0   # v0 feeds into step 1")

arrow(27, 86, 27, 79)
arrow(73, 86, 73, 79)

# ===================================================================
# 3. BRIDGE — Graph-Aware Encoder
section_band(76, "STAGE 3 — GRAPH-AWARE BRIDGE  (3× Dense GAT)", COL_BRIDGE)

box(20, 58, 60, 16, COL_BRIDGE,
    "GraphAwareBridge\n(Op embedding + 3 stacked GAT layers)",
    "1. Embed op IDs to (d − 4)-dim vectors.\n"
    "2. Concatenate with the 4 numeric features → d-dim.\n"
    "3. Apply 3 Dense GAT layers (multi-head attention\n"
    "   over the adjacency mask) to propagate info\n"
    "   between connected operation nodes.",
    "ids [B,50]; vals [B,50,4]; adj [B,50,50]",
    "FloatTensor [B, 50, d=256]",
    "x_t[0] now encodes 'sub of 47 and 9' as a\n"
    "256-dim vector aware of its downstream node.")

arrow(50, 58, 50, 55)

# ===================================================================
# 4. HRM CORE — Hierarchical reasoning with H/L cycles
section_band(52, "STAGE 4 — HRM CORE  (H + L modules, hierarchical convergence)", COL_HRM)

box(8, 26, 40, 24, COL_HRM,
    "HRM Cycles  (one ACT segment)",
    "For Hcycles=3 outer loops:\n"
    "  For Lcycles=4 inner loops:\n"
    "    zL ← L_module(zL, zH + x_t)\n"
    "  zH ← H_module(zH, zL)\n\n"
    "Only the FINAL H/L iteration backprops\n"
    "(one-step gradient). Prior 11 iterations\n"
    "run under no_grad → cheap deep inference.",
    "x_t [B,50,d]; zH, zL [B,50,d]",
    "zH, zL [B, 50, d]",
    "12 H/L iterations per segment.\n"
    "Latent state slowly converges to a\n"
    "representation that 'computes' the trace.")

box(54, 26, 38, 24, COL_HRM,
    "Why this works",
    "• L_module = fast local refinement\n"
    "• H_module = slow strategic update\n"
    "• Repeating H/L lets the latent state\n"
    "  iteratively compute (sub then add)\n"
    "  in the latent space.\n"
    "• 4 ACT segments per training pass →\n"
    "  48 reasoning steps total.",
    "(architectural intuition)",
    "zH at position nr-1 (final_answer node) carries\n"
    "the computed answer in its 256-dim embedding.",
    "")

arrow(28, 26, 28, 23)
# Branch: zH feeds into both heads
arrow(50, 24, 35, 18)
arrow(50, 24, 65, 18)

# ===================================================================
# 5. TWO HEADS
section_band(20, "STAGE 5 — OUTPUT HEADS  (Q-head for ACT halting + Digit head for answer)", COL_HEAD)

box(8, 4, 38, 14, COL_ACT,
    "Q-Head (ACT halt/continue)",
    "Reads zH[:, 0] (the first node's H-state)\n"
    "through LayerNorm + Linear(d→2).\n"
    "Outputs two logits: q_halt and q_continue.\n"
    "If q_halt > q_continue (and segment ≥ act_min),\n"
    "the model stops computing.",
    "zH[:, 0]  [B, d=256]",
    "(q_halt, q_continue)  [B], [B]",
    "Halts when confident; trained with Q-learning\n"
    "(BCE vs correctness; bootstrapped TD target).")

box(54, 4, 38, 14, COL_HEAD,
    "Digit Head (final answer)",
    "Linear(d → MAX_DIGITS × DIGIT_VOCAB_SIZE)\n"
    "applied per-node, reshaped to [B, 50, 8, 13].\n"
    "At node nr-1 (final_answer position),\n"
    "take argmax over the 13-vocab to get 8 digit\n"
    "tokens, then decode_digits → integer.",
    "zH  [B, 50, d=256]",
    "Logits [B, 50, 8, 13]  → int answer",
    "Digit tokens [4, 4, 12, 0, 0, 0, 0, 0]\n"
    "= ['4', '3', EOS, PAD, ...] → decode → 43 ✓")

# Final result arrow
arrow(73, 4, 73, 1.5)
ax.text(73, 0.8, "ANSWER:  43", ha="center", va="top",
        fontsize=12, fontweight="bold", color="#0A6E0A")

# ===================================================================
# Right-side annotation: ACT loop arrow
# Draw a loopback arrow indicating "if q_continue, repeat for next ACT segment"
loop = FancyArrowPatch((85, 11), (95, 11), connectionstyle="arc3,rad=-1.2",
                       arrowstyle="-|>", mutation_scale=15,
                       linewidth=1.4, color="#882288", linestyle="--")
ax.add_patch(loop)
ax.text(96, 30, "ACT loop:\n"
                "if not halted,\n"
                "carry zH, zL,\n"
                "do another\n"
                "segment\n"
                "(max 4)",
        ha="left", va="center", fontsize=8.5, color="#882288",
        style="italic")
# A vertical arrow from the loop point back up to the HRM core
arr2 = FancyArrowPatch((94, 13), (94, 38), arrowstyle="-|>",
                       mutation_scale=15, linewidth=1.4,
                       color="#882288", linestyle="--")
ax.add_patch(arr2)

# ===================================================================
# Legend at the bottom
legend_patches = [
    mpatches.Patch(facecolor=COL_INPUT,  edgecolor=COL_BORDER, label="Input"),
    mpatches.Patch(facecolor=COL_PARSE,  edgecolor=COL_BORDER, label="Tensor encoding"),
    mpatches.Patch(facecolor=COL_BRIDGE, edgecolor=COL_BORDER, label="Graph encoder (GAT)"),
    mpatches.Patch(facecolor=COL_HRM,    edgecolor=COL_BORDER, label="HRM core (H+L cycles)"),
    mpatches.Patch(facecolor=COL_ACT,    edgecolor=COL_BORDER, label="Q-head (ACT)"),
    mpatches.Patch(facecolor=COL_HEAD,   edgecolor=COL_BORDER, label="Digit head"),
]
ax.legend(handles=legend_patches, loc="upper center",
          bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=9)

plt.tight_layout()
out_path = "/Users/mohamedshamil/Desktop/HRM-MLX/docs/hrm_architecture.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
