"""
HRM Pipeline API Server — live inference + pre-computed predictions.

Provides:
  GET  /                    → Serves the UI
  GET  /predictions.json    → Pre-computed predictions
  POST /api/predict          → Live inference from a math question

Usage:
    python src/serve_ui.py
    → Open http://localhost:8765
"""
import os, sys, json, re, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import threading

from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np

from src.dataset import (
    GSM8KGraphDataset, DIGIT_VOCAB, DIGIT_VOCAB_SIZE,
    MAX_DIGITS, decode_digits_to_number, IDX_TO_DIGIT,
    parse_graph_from_json, encode_number_to_digits
)
from src.notebook_model import HRMForMath, OP_VOCAB

IDX_TO_OP = {v: k for k, v in OP_VOCAB.items()}

# ─── Global model (loaded once) ─────────────────────────────────────────────
MODEL = None
LLM_CLIENT = None


def load_model():
    global MODEL
    ckpt = "checkpoints/gsm8k/real_best_model.pt"
    
    if not os.path.exists(ckpt):
        print(f"ERROR: No checkpoint found at {ckpt}!")
        return
    
    MODEL = HRMForMath(
        vsz=len(OP_VOCAB), d=512, heads=8,
        Hc=4, Lc=8, Hl=8, Ll=8, slen=50,
    )
    
    # Load PyTorch state dict
    sd = torch.load(ckpt, map_location='cpu')
    if 'model_state' in sd:
        sd = sd['model_state']
        
    MODEL.load_state_dict(sd)
    MODEL.eval()
    print(f"Model loaded from {ckpt}")


def init_llm_client():
    global LLM_CLIENT
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if api_key:
        from openai import OpenAI
        LLM_CLIENT = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        print("LLM client initialized (NVIDIA API)")
    else:
        print("WARNING: No NVIDIA_API_KEY — live inference will not work")


def parse_question_with_llm(question):
    """Parse a math question into a reasoning trace using the LLM."""
    if not LLM_CLIENT:
        return None, "No LLM API key configured"
    
    prompt = f"""You are an expert at extracting reasoning traces from math problems into a strict JSON format.

Problem: {question}

Solve this step by step, then format the reasoning as a JSON object with:
- "steps": array of steps, each with "op" (add/sub/mul/div/const), "arg1", "arg2", "result" (v1, v2, etc.)
- "final_answer": the variable name of the final result
- "explanation": a brief natural language explanation of the solution

Return ONLY valid JSON."""

    try:
        completion = LLM_CLIENT.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        content = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        elif '```' in content:
            match2 = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if match2:
                content = match2.group(1)
        
        trace = json.loads(content)
        explanation = trace.pop("explanation", "")
        return trace, explanation
    except Exception as e:
        return None, str(e)


def run_inference(trace):
    """Run the HRM model on a parsed trace and return all intermediate outputs."""
    if MODEL is None:
        return None
    
    # Parse graph
    (node_ids, node_values, node_digit_targets, adj_mask,
     num_real, raw_results) = parse_graph_from_json(trace, max_nodes=50)
    
    # Compute ground truth from trace
    ans_var = trace.get("final_answer", "")
    var_to_value = {}
    for step in trace.get("steps", []):
        res = step.get("result", "")
        arg1 = step.get("arg1", 0)
        arg2 = step.get("arg2", 0)
        
        def resolve(a):
            if isinstance(a, (int, float)):
                return float(a)
            elif isinstance(a, str) and a in var_to_value:
                return var_to_value[a]
            try:
                return float(a)
            except:
                return 0.0
        
        v1, v2 = resolve(arg1), resolve(arg2)
        op = step.get("op", "const")
        if op == "add": result = v1 + v2
        elif op == "sub": result = v1 - v2
        elif op == "mul": result = v1 * v2
        elif op == "div" and v2 != 0: result = v1 / v2
        else: result = v1
        
        if res:
            var_to_value[res] = result
    
    computed_answer = int(round(var_to_value.get(ans_var, 0)))
    
    # Build batch (PyTorch)
    batch = {
        "node_ids": torch.tensor([node_ids], dtype=torch.long),
        "node_values": torch.tensor([node_values], dtype=torch.float32),
        "adj_mask": torch.tensor([adj_mask], dtype=torch.float32),
        "node_digit_targets": torch.tensor([node_digit_targets], dtype=torch.long),
        "final_digit_target": torch.tensor([encode_number_to_digits(computed_answer)], dtype=torch.long),
        "raw_target": torch.tensor([computed_answer], dtype=torch.long),
        "num_real_nodes": torch.tensor([num_real], dtype=torch.long),
    }
    
    # Forward pass
    inputs = batch["node_ids"]
    node_vals = batch["node_values"]
    adj = batch["adj_mask"]
    B, N = inputs.shape
    
    with torch.no_grad():
        # Bridge
        graph_ctx = MODEL.bridge(inputs, node_vals, adj)
        
        # Full forward
        digit_logits, q_halt, q_continue = MODEL(batch)
        
        li = min(num_real - 1, N - 1)
        final_logits = digit_logits[0, li]
        probs = torch.softmax(final_logits, dim=-1)
        pred_digits_arr = torch.argmax(probs, dim=-1)
        
        pred_list = [int(pred_digits_arr[d].item()) for d in range(MAX_DIGITS)]
        pred_int = decode_digits_to_number(pred_list)
        
        # Build digit distributions
        digit_probs = []
        for d in range(MAX_DIGITS):
            dist = {}
            for v in range(DIGIT_VOCAB_SIZE):
                label = IDX_TO_DIGIT.get(v, '?')
                dist[label] = round(float(probs[d, v].item()), 4)
            digit_probs.append({
                "position": d,
                "predicted": IDX_TO_DIGIT.get(pred_list[d], '?'),
                "distribution": dist
            })
        
        # Build nodes
        nodes = []
        for n in range(num_real):
            op_id = node_ids[n]
            op_name = IDX_TO_OP.get(op_id, "PAD")
            v1 = round(node_values[n][0], 3)
            v2 = round(node_values[n][1], 3)
            
            node_logits = digit_logits[0, n]
            node_pred = torch.argmax(node_logits, dim=-1)
            node_pred_list = [int(node_pred[d].item()) for d in range(MAX_DIGITS)]
            node_pred_int = decode_digits_to_number(node_pred_list)
            
            node_target = decode_digits_to_number(node_digit_targets[n])
            emb_norm = round(float(torch.sqrt(torch.sum(graph_ctx[0, n] ** 2)).item()), 3)
            
            nodes.append({
                "id": n,
                "op": op_name,
                "arg1_norm": v1,
                "arg2_norm": v2,
                "embedding_norm": emb_norm,
                "predicted": node_pred_int,
                "target": node_target,
                "correct": node_pred_int == node_target
            })
        
        # Edges
        edges = []
        adj_np = np.array(adj_mask)
        for i in range(num_real):
            for j in range(num_real):
                if adj_np[i][j] > 0.5:
                    edges.append({"source": i, "target": j})
        
        return {
            "predicted_answer": pred_int,
            "computed_answer": computed_answer,
            "correct": pred_int == computed_answer,
            "num_nodes": num_real,
            "graph": {"nodes": nodes, "edges": edges},
            "digit_probs": digit_probs,
            "pred_digits": [IDX_TO_DIGIT.get(d, '?') for d in pred_list],
            "q_halt": round(float(q_halt[0, 0].item()), 4),
            "q_continue": round(float(q_continue[0, 0].item()), 4),
        }


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class HRMHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="ui", **kwargs)
    
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            question = data.get("question", "")
            if not question:
                self._json_response({"error": "No question provided"}, 400)
                return
            
            # Step 1: Parse with LLM
            trace, explanation = parse_question_with_llm(question)
            if not trace:
                self._json_response({"error": f"LLM parsing failed: {explanation}"}, 500)
                return
            
            # Step 2: Run HRM inference
            result = run_inference(trace)
            if not result:
                self._json_response({"error": "Model inference failed"}, 500)
                return
            
            result["question"] = question
            result["explanation"] = explanation
            result["trace"] = trace
            
            self._json_response(result)
        else:
            self.send_error(404)
    
    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        if '/api/' in str(args[0]):
            print(f"  API: {args[0]}")


def main():
    load_model()
    init_llm_client()
    
    port = 8765
    server = HTTPServer(('', port), HRMHandler)
    print(f"\n{'='*50}")
    print(f"  HRM Pipeline Visualizer")
    print(f"  Open: http://localhost:{port}")
    print(f"{'='*50}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

