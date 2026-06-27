"""
Cache the GSM8K *test* split (1,319 problems) using the same LLM parser.

The model is never trained on this split. After caching + cleaning,
the model is evaluated on it to produce the test-set accuracy that
the research report needs.

Usage:
    python src/cache_gsm8k_test.py
    python src/clean_data.py --input data/gsm8k_test_parsed.json \
                             --output data/gsm8k_test_clean.json
"""
import os
import sys

# Make the project root importable regardless of where this is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the existing batched parser. Just point it at the test split.
from src.cache_gsm8k import cache_dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cache_dataset(split="test", num_samples=1319)
