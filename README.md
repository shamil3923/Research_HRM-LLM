# HRM-MLX — Hierarchical Reasoning for GSM8K Math

A hybrid **HRM + LLM** system for multi-step math word problems (GSM8K).

- The **HRM genuinely learns integer arithmetic** (add / sub / mul / div) from a
  digit-aligned representation — it computes, it does not memorize or call a calculator.
- An **LLM acts only as a translator**: it parses a word problem into a structured
  JSON trace of arithmetic steps. It never does the math.
- A **graph-aware bridge** maps that JSON trace into tensors and executes each step
  through the trained HRM primitives, chaining them to a final answer.

## Pipeline

```
word problem ──(LLM translator)──▶ JSON trace ──(bridge)──▶ tensors ──(HRM)──▶ answer
```

Training is staged: **pretrain** on synthetic arithmetic → **curriculum** on faithful
GSM8K traces → **finetune**.

## Layout

```
hrm_arith.py, faithful_hrm.py   core HRM arithmetic + data generators (inlined into notebooks)
notebooks/                      current Kaggle notebooks (built by builders/)
builders/                       scripts that assemble the notebooks and parse datasets
data/                           GSM8K traces (faithful, openai, qwen)
ckpt_faithful/                  model weights for the current pipeline
docs/                           progress report + architecture diagram
archive/                        superseded notebooks/builders/debug + the old MLX impl (archive/mlx/)
```

> The original MLX implementation (`src/`, `models/`, training scripts, old checkpoints)
> is not used by the current pipeline and now lives under `archive/mlx/`.

## Datasets (`data/`)

| Set | Source | Grounding | Use |
|-----|--------|-----------|-----|
| `gsm8k_faithful_*`        | extracted from GSM8K `<<>>` annotations | ~73% | primary training |
| `gsm8k_llm_traces_openai_*` | gpt-4o-mini, validation-gated | ~76% | LLM-parsed arm / robustness |
| `gsm8k_*_parsed/split`    | Qwen (legacy)                 | ~27% | discredited baseline only |

Traces are kept only if they re-execute to the dataset's gold answer; number-grounding
(operands actually present in the problem) is reported to detect confabulation.

## Usage

Run builders from the repo root:

```bash
# build the arithmetic-HRM notebook
venv/bin/python3 builders/build_arith_notebook.py

# build the faithful training notebook
venv/bin/python3 builders/build_faithful_notebook.py

# regenerate faithful GSM8K traces
venv/bin/python3 builders/build_faithful_gsm8k.py

# parse GSM8K with the OpenAI Batch API (needs OPENAI_API_KEY)
venv/bin/python3 builders/build_llm_traces_openai.py
```

The notebooks are self-contained (core code is inlined) and run on Kaggle GPUs.

## Notes

- Exact integer arithmetic requires TF32 **disabled** on Ampere GPUs
  (`torch.backends.cuda.matmul.allow_tf32 = False`).
- Secrets belong in `.env` (gitignored) or Kaggle Secrets — never commit API keys.
