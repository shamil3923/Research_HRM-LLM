import os
import json
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

load_dotenv()

api_key = os.environ.get("NVIDIA_API_KEY", "")
if not api_key:
    print("\nERROR: NVIDIA_API_KEY environment variable is missing.")
    print("  export NVIDIA_API_KEY='your-key-here'")
    exit(1)

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b")

client = OpenAI(base_url=BASE_URL, api_key=api_key)
print(f"[parser] Using model: {MODEL_NAME}")

write_lock = threading.Lock()

BATCH_SIZE = 25  # Problems per API call — fewer calls = faster


def parse_batch(problems, max_retries=5):
    """
    Parse MULTIPLE GSM8K problems in a single LLM call.
    Returns a list of parsed traces (one per problem).
    """
    # Build the batched prompt
    prompt_parts = []
    for i, (question, answer) in enumerate(problems):
        prompt_parts.append(f"""
--- PROBLEM {i+1} ---
Question: {question}
Solution: {answer}
""")
    
    all_problems = "\n".join(prompt_parts)
    
    prompt = f"""You are an expert at extracting reasoning traces from math problems.
Parse each of the following {len(problems)} GSM8K problems into structured JSON.

{all_problems}

Return a JSON ARRAY with exactly {len(problems)} objects, one per problem, in order.
Each object should have:
- "steps": array of steps, each with "op" (add/sub/mul/div/const), "arg1", "arg2", "result" (v1, v2, etc.)
- "final_answer": the variable name of the final result

Return ONLY the JSON array. No explanation."""

    base_delay = 5

    # qwen/qwen3.5-* are reasoning models — by default they put output in
    # `reasoning_content` and leave `content` empty. Disable thinking so the
    # JSON lands in `content` where the regex extractor can find it.
    extra_body = {"chat_template_kwargs": {"thinking": False}}

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                extra_body=extra_body,
            )

            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError(
                    f"Model '{MODEL_NAME}' returned empty content. "
                    "Check NVIDIA_MODEL in .env — it may be retired/unavailable."
                )

            # Extract JSON from optional ```json fences
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            elif '```' in content:
                match2 = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if match2:
                    content = match2.group(1)

            parsed = json.loads(content)

            if isinstance(parsed, dict):
                return [parsed]
            return parsed

        except json.JSONDecodeError:
            return []
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            elif "DEGRADED" in error_msg:
                print(f"\n[ERROR] Model DEGRADED. Skipping batch.")
                return []
            else:
                return []

    return []


def extract_target_value(answer_text):
    """Extracts the final numerical answer from GSM8K text format (after ####)."""
    match = re.search(r'####\s*(-?[\d,]+)', answer_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return 0.0


def load_existing_data(out_file):
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            data = json.load(f)
        print(f"Resuming from existing data: {len(data)} samples")
        return data
    return []


def save_data(data, out_file):
    with write_lock:
        os.makedirs("data", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)


def cache_dataset(split="train", num_samples=7500):
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)
    
    out_file = f"data/gsm8k_{split}_parsed.json"
    output_data = load_existing_data(out_file)
    start_idx = len(output_data)  # Resume from where we left off
    
    total = min(len(dataset), num_samples)
    remaining = total - start_idx
    
    if remaining <= 0:
        print(f"Already have {len(output_data)} samples. Done!")
        return
    
    # Create batches of BATCH_SIZE problems each
    batches = []
    batch_items = []
    for i in range(start_idx, total):
        item = dataset[i]
        batch_items.append((i, item))
        if len(batch_items) == BATCH_SIZE:
            batches.append(batch_items)
            batch_items = []
    if batch_items:
        batches.append(batch_items)
    
    num_batches = len(batches)
    print(f"\nTotal to process: {remaining} samples in {num_batches} batches of {BATCH_SIZE}")
    print(f"This means only ~{num_batches} API calls instead of {remaining}!")
    print(f"Saving incrementally every batch to {out_file}\n")
    
    # Process batches with moderate concurrency
    max_workers = 5  # 5 concurrent batches × 10 problems = 50 problems in flight
    success_total = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {}
        for batch in batches:
            problems = [(item["question"], item["answer"]) for _, item in batch]
            future = executor.submit(parse_batch, problems)
            future_to_batch[future] = batch
        
        for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Batches"):
            batch = future_to_batch[future]
            try:
                traces = future.result()
                
                # Match traces back to items
                for j, trace in enumerate(traces):
                    if j < len(batch) and trace and isinstance(trace, dict) and "steps" in trace:
                        _, item = batch[j]
                        target = extract_target_value(item["answer"])
                        with write_lock:
                            output_data.append({
                                "question": item["question"],
                                "trace": trace,
                                "target": target,
                            })
                            success_total += 1
                
                # Save after every batch
                save_data(output_data, out_file)
                
            except Exception as e:
                tqdm.write(f"  Batch error: {e}")
    
    # Final save
    save_data(output_data, out_file)
    
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Total cached: {len(output_data)} samples")
    print(f"  New in this run: {success_total}")
    print(f"  Saved to: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cache_dataset("train", num_samples=7500)
