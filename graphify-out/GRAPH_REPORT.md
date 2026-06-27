# Graph Report - .  (2026-05-14)

## Corpus Check
- Corpus is ~43,179 words - fits in a single context window. You may not need a graph.

## Summary
- 624 nodes · 923 edges · 57 communities (45 shown, 12 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 149 edges (avg confidence: 0.71)
- Token cost: 187,821 input · 46,953 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Dual AdamATan2 Optimizer|Dual AdamATan2 Optimizer]]
- [[_COMMUNITY_Config & Data Loading|Config & Data Loading]]
- [[_COMMUNITY_v3.2 Predictions Export|v3.2 Predictions Export]]
- [[_COMMUNITY_Training Step & Collation|Training Step & Collation]]
- [[_COMMUNITY_Core Training Concepts|Core Training Concepts]]
- [[_COMMUNITY_GSM8K Dataset Module|GSM8K Dataset Module]]
- [[_COMMUNITY_Pipeline Architecture (Paper)|Pipeline Architecture (Paper)]]
- [[_COMMUNITY_MLX HRM Evaluator|MLX HRM Evaluator]]
- [[_COMMUNITY_Optimizer Utilities|Optimizer Utilities]]
- [[_COMMUNITY_Graph-Aware Bridge (GAT)|Graph-Aware Bridge (GAT)]]
- [[_COMMUNITY_Explanation HTTP Server|Explanation HTTP Server]]
- [[_COMMUNITY_Predictions HTTP Server|Predictions HTTP Server]]
- [[_COMMUNITY_AdamATan2 Implementations|AdamATan2 Implementations]]
- [[_COMMUNITY_HRM Carry State|HRM Carry State]]
- [[_COMMUNITY_Smoke Test Pipeline|Smoke Test Pipeline]]
- [[_COMMUNITY_Sudoku Configs & README|Sudoku Configs & README]]
- [[_COMMUNITY_HRMConfig Loader|HRMConfig Loader]]
- [[_COMMUNITY_HRM Inner HL Cycles|HRM Inner H/L Cycles]]
- [[_COMMUNITY_Kaggle Training Guide|Kaggle Training Guide]]
- [[_COMMUNITY_Init & Casted Layers|Init & Casted Layers]]
- [[_COMMUNITY_LLM Explanation Module|LLM Explanation Module]]
- [[_COMMUNITY_MLX GAT Bridge|MLX GAT Bridge]]
- [[_COMMUNITY_HRM Transformer Blocks|HRM Transformer Blocks]]
- [[_COMMUNITY_HRM Reasoning Blocks|HRM Reasoning Blocks]]
- [[_COMMUNITY_Training Curves (v3.1)|Training Curves (v3.1)]]
- [[_COMMUNITY_GSM8K Evaluation Concepts|GSM8K Evaluation Concepts]]
- [[_COMMUNITY_MLX Attention & Rotary|MLX Attention & Rotary]]
- [[_COMMUNITY_Export Predictions (orig)|Export Predictions (orig)]]
- [[_COMMUNITY_REINFORCE RL Fine-tune|REINFORCE RL Fine-tune]]
- [[_COMMUNITY_Training Stats Builder|Training Stats Builder]]
- [[_COMMUNITY_HRM ACT Wrapper|HRM ACT Wrapper]]
- [[_COMMUNITY_HRM Common Utilities|HRM Common Utilities]]
- [[_COMMUNITY_Predict CLI Script|Predict CLI Script]]
- [[_COMMUNITY_Curriculum Learning|Curriculum Learning]]
- [[_COMMUNITY_PyTorch Checkpoint Tooling|PyTorch Checkpoint Tooling]]
- [[_COMMUNITY_Sparse Embedding|Sparse Embedding]]
- [[_COMMUNITY_Implementation Stack (Paper)|Implementation Stack (Paper)]]
- [[_COMMUNITY_Data Cleaning|Data Cleaning]]
- [[_COMMUNITY_Data Diagnostics|Data Diagnostics]]
- [[_COMMUNITY_GSM8K Test Caching|GSM8K Test Caching]]
- [[_COMMUNITY_Per-Cycle Norm Trace|Per-Cycle Norm Trace]]
- [[_COMMUNITY_Load YAML Config|Load YAML Config]]
- [[_COMMUNITY_Args - Config|Args -> Config]]
- [[_COMMUNITY_Combined Optimizer State|Combined Optimizer State]]
- [[_COMMUNITY_OP_VOCAB constants|OP_VOCAB constants]]
- [[_COMMUNITY_DIGIT_VOCAB constants|DIGIT_VOCAB constants]]
- [[_COMMUNITY_Sudoku Loader|Sudoku Loader]]
- [[_COMMUNITY_PyTorch DataLoader|PyTorch DataLoader]]
- [[_COMMUNITY_Kaggle T4 GPU|Kaggle T4 GPU]]

## God Nodes (most connected - your core abstractions)
1. `HRMForMath` - 28 edges
2. `HierarchicalReasoningModel_Inner` - 21 edges
3. `HierarchicalReasoningModel` - 21 edges
4. `CastedLinear` - 20 edges
5. `Attention` - 17 edges
6. `CastedEmbedding` - 16 edges
7. `SwiGLU` - 16 edges
8. `HRMTrainer` - 15 edges
9. `HRMTransformerBlock` - 14 edges
10. `HRMReasoningModule` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Kaggle GPU training guide` --semantically_similar_to--> `train()`  [INFERRED] [semantically similar]
  kaggle/KAGGLE_GUIDE.md → src/train.py
- `config.yaml (root)` --references--> `HRMForMath`  [INFERRED]
  config.yaml → src/model.py
- `clean_data script` --semantically_similar_to--> `parse_graph_from_json()`  [INFERRED] [semantically similar]
  src/clean_data.py → kaggle/hrm_gsm8k_pytorch.py
- `trace_to_lines()` --semantically_similar_to--> `parse_graph_from_json()`  [INFERRED] [semantically similar]
  src/explainer.py → kaggle/hrm_gsm8k_pytorch.py
- `DenseGATLayer` --semantically_similar_to--> `DenseGATLayer`  [INFERRED] [semantically similar]
  kaggle/hrm_gsm8k_pytorch.py → src/graph_encoder.py

## Hyperedges (group relationships)
- **MLX Sudoku training stack** — pretrain, train_yaml, dual_optimizer_dualadamatan2, mlx_adam_atan2_exact_adamatan2exact, lr_scheduler_cosineschedulewithwarmup, load_official_sudoku_fn [INFERRED 0.85]
- **v3.2 GSM8K inference + explanation pipeline** — export_predictions_v32, serve_v32, regenerate_explanations, build_training_stats [INFERRED 0.85]
- **PyTorch checkpoint inspection and conversion tools** — convert_pt, inspect_pt, find_pt, create_dummy, test_load [INFERRED 0.85]
- **HRM hierarchical H/L reasoning with ACT halting** — hrm_hrm_act_v1_hierarchicalreasoningmodel, hrm_hrm_act_v1_hierarchicalreasoningmodel_inner, hrm_hrm_act_v1_hrmreasoningmodule, concept_hierarchical_reasoning, concept_act_halting, concept_one_step_gradient [INFERRED 0.95]
- **GSM8K LLM-parsed trace pipeline (cache -> clean -> train)** — src_cache_gsm8k_cache_dataset, src_cache_gsm8k_parse_batch, src_clean_data_script, kaggle_hrm_gsm8k_pytorch_parse_graph_from_json, kaggle_hrm_gsm8k_pytorch_gsm8kdataset [INFERRED 0.85]
- **Graph-Aware Bridge: GAT-based op-token + numeric value embedding** — src_graph_encoder_graphawarebridge, src_graph_encoder_densegatlayer, kaggle_hrm_gsm8k_pytorch_graphawarebridge, kaggle_hrm_gsm8k_pytorch_densegatlayer [INFERRED 0.95]
- **HRM forward pipeline: bridge -> H/L cycles -> digit head** — src_model_hrmformath, src_model_hrmreasoningmodule, src_model_hrmblock, src_model_digit_head [EXTRACTED 1.00]
- **Training stack: model + dataset + optimizer + loss + curriculum** — src_train_train, src_model_hrmformath, src_dataset_gsm8kgraphdataset, src_optimizer_trainingoptimizer, src_train_final_node_digit_loss, src_train_curriculumdataset [EXTRACTED 1.00]
- **REINFORCE fine-tune pipeline** — src_rl_finetune_train, src_rl_finetune_reinforce_loss, src_rl_finetune_compute_rewards, src_train_final_node_digit_loss, src_model_hrmformath [EXTRACTED 1.00]
- **Six-Component Hybrid Pipeline Architecture** — paper_llm_parser_module, paper_bridge_module, paper_hrm_core, paper_post_processor_module, paper_llm_explanation_module, paper_final_output_component [EXTRACTED 1.00]
- **HRM Core Two-Level Reasoning** — paper_hrm_core, paper_h_module, paper_l_module, paper_adaptive_computation_time, paper_halt_gate [EXTRACTED 1.00]
- **Evaluation Metrics for GSM8K** — paper_exact_match_accuracy, paper_mae_metric, paper_medae_metric, paper_explanation_coherence, paper_gsm8k_dataset [EXTRACTED 1.00]

## Communities (57 total, 12 thin omitted)

### Community 0 - "Dual AdamATan2 Optimizer"
Cohesion: 0.05
Nodes (25): DualAdamATan2, Dual optimizer support for separate embedding learning rates Matches the origina, Update parameters using both optimizers, Separate gradients into main and embedding groups, Dual optimizer setup matching original PyTorch HRM:     - AdamATan2 for main par, Update learning rate for main optimizer (for LR scheduler compatibility), Initialize both optimizers with separated parameter groups, Separate parameters into main and embedding groups (+17 more)

### Community 1 - "Config & Data Loading"
Cohesion: 0.07
Nodes (28): Config Loader module, Dual Optimizer module, analyze_dataset(), load_official_sudoku_data(), Load official Sudoku-Extreme dataset from HuggingFace, Analyze the dataset to understand difficulty distribution, Load Sudoku data from official CSV format          CSV format: source,question,a, Apply valid Sudoku transformations that preserve correctness     Based on offici (+20 more)

### Community 2 - "v3.2 Predictions Export"
Cohesion: 0.1
Nodes (21): decode_digits(), DenseGATLayer, encode_number(), GraphAwareBridge, HRMBlock, HRMForMath, HRMModule, main() (+13 more)

### Community 3 - "Training Step & Collation"
Cohesion: 0.1
Nodes (18): 1-step gradient approximation, decode_digits_to_number(), DenseGATLayer, evaluate(), final_node_digit_loss(), GraphAwareBridge, HRMBlock, HRMForMath (+10 more)

### Community 4 - "Core Training Concepts"
Cohesion: 0.1
Nodes (29): Build Training Stats, AdamATan2 optimizer algorithm, Cosine LR with warmup schedule, GSM8K reasoning pipeline (v3.2), Hierarchical Reasoning Model (HRM), ui/predictions.json artifact, Sudoku-Extreme training pipeline, HRMConfig dataclass (+21 more)

### Community 5 - "GSM8K Dataset Module"
Cohesion: 0.1
Nodes (22): Dataset, DIGIT_VOCAB, encode_number_to_digits(), GSM8KDataset, OP_VOCAB, parse_graph_from_json(), cache_dataset(), extract_target_value() (+14 more)

### Community 6 - "Pipeline Architecture (Paper)"
Cohesion: 0.11
Nodes (24): Adaptive Computation Time (ACT), Bridge Module (Representation Mapping), Design Rationale, Deterministic Fallback Parser, File-based JSON Cache, Final Output (Answer + Explanation), H-module (Slow Planner), Halt Gate (Explicit Halting) (+16 more)

### Community 7 - "MLX HRM Evaluator"
Cohesion: 0.12
Nodes (17): Adaptive Computation Time (ACT), HRMEvaluator, main(), Evaluator for HRM model, Load model checkpoint, Evaluate model on dataset, binary_cross_entropy_with_logits(), compute_act_loss() (+9 more)

### Community 8 - "Optimizer Utilities"
Cohesion: 0.12
Nodes (11): clip_grad_norm(), Optimizer utilities for HRM-MLX GSM8K training.  Provides a thin wrapper around, Restart LR schedule at current step (call at phase boundaries)., Apply one optimizer step:           1. Clip gradients           2. Update LR fro, Clip gradients by global L2 norm.     Returns (clipped_grads, grad_norm_before_c, Linear warmup for `warmup_steps` steps, then holds at `peak_lr`.     Call `resta, Linear warmup then cosine decay.     Call `restart(current_step)` at curriculum, Wrapper combining Adam optimizer + LR schedule + gradient clipping.     Tracks s (+3 more)

### Community 9 - "Graph-Aware Bridge (GAT)"
Cohesion: 0.16
Nodes (7): DenseGATLayer, GraphAwareBridge, HRMBlock, HRMForMath, HRMModule, rms_norm(), SwiGLU

### Community 10 - "Explanation HTTP Server"
Cohesion: 0.17
Nodes (12): _call_with_timeout(), generate_explanation(), Handler, init_llm(), init_model(), main(), parse_question(), Combined static file server + live-inference backend for the v3.2 PyTorch HRM. (+4 more)

### Community 11 - "Predictions HTTP Server"
Cohesion: 0.16
Nodes (11): main(), SimpleHTTPRequestHandler, HRMHandler, init_llm_client(), load_model(), main(), parse_question_with_llm(), HRM Pipeline API Server — live inference + pre-computed predictions.  Provides: (+3 more)

### Community 12 - "AdamATan2 Implementations"
Cohesion: 0.13
Nodes (11): AdamATan2, AdamATan2Scaled, MLX implementation of AdamATan2 optimizer  Based on the paper "Scaling Exponents, Apply gradients to all parameters, Enhanced version with better scaling for the atan2 operation          This versi, Apply scaled AdamATan2 update, AdamATan2 optimizer for MLX          Uses atan2 instead of division for numerica, Test the AdamATan2 implementation (+3 more)

### Community 13 - "HRM Carry State"
Cohesion: 0.15
Nodes (10): HRMCarry, HRMInnerCarry, Create empty inner carry - EXACT match to original lines 168-172, Reset carry - EXACT match to original lines 174-178, Inner carry state for HRM - matches HierarchicalReasoningModel_ACTV1InnerCarry, Initialize carry state - matches original lines 228-238, Complete carry state - matches HierarchicalReasoningModel_ACTV1Carry, Forward pass with ACT - matches original lines 240-283 (+2 more)

### Community 14 - "Smoke Test Pipeline"
Cohesion: 0.16
Nodes (11): main(), Quick smoke test for the HRM pipeline., DIGIT_VOCAB, encode_number_to_digits(), GSM8KGraphDataset, OP_VOCAB, parse_graph_from_json(), GSM8K dataset with digit-level classification targets.  Each node's arithmetic r (+3 more)

### Community 15 - "Sudoku Configs & README"
Cohesion: 0.12
Nodes (8): README, Hierarchical reasoning concept, ACT Q-head halting, Digit classification head, HRMForMath, 1-step gradient approximation, Forward pass.                  Returns:             digit_logits: (B, N, MAX_DIG, Full HRM adapted for GSM8K math reasoning.          Architecture:       Bridge(n

### Community 16 - "HRMConfig Loader"
Cohesion: 0.17
Nodes (10): from_yaml(), HRMConfig, load_config(), Configuration loader for HRM training Supports YAML configuration files like the, Convert to dictionary, Save configuration to YAML file, Load configuration with priority: command line args > YAML config > defaults, Configuration dataclass for HRM training (+2 more)

### Community 17 - "HRM Inner H/L Cycles"
Cohesion: 0.19
Nodes (9): Hierarchical Reasoning (H/L cycles), HierarchicalReasoningModel_Inner, Initialize Q-head exactly as in official code, Input embeddings - EXACT match to original lines 146-166, Forward pass - EXACTLY matching original lines 180-213, Inner HRM model - matches HierarchicalReasoningModel_ACTV1_Inner, trunc_normal_init_, CastedEmbedding (+1 more)

### Community 18 - "Kaggle Training Guide"
Cohesion: 0.2
Nodes (13): config.yaml (root), Kaggle GPU training guide, digit_cross_entropy_loss(), evaluate(), final_node_digit_loss(), parse_args(), Training script for HRMForMath on GSM8K.  Trains the HRM with digit-level classi, Evaluate exact-match accuracy and per-digit accuracy.          Returns: (exact_m (+5 more)

### Community 19 - "Init & Casted Layers"
Cohesion: 0.21
Nodes (7): _find_multiple, Truncated normal initialization - EXACT match to original common.py:7-30, trunc_normal_init_(), CastedLinear, SwiGLU activation - EXACT match to original lines 139-149, Linear layer with dtype casting - EXACT match to original, SwiGLU

### Community 20 - "LLM Explanation Module"
Cohesion: 0.29
Nodes (10): explain(), explain_batch(), _get_client(), main(), LLM Explanation Module — converts an HRM reasoning trace into a step-by-step nat, Generate a natural-language explanation for a single problem.      Returns a cle, items: list of {"question", "trace", "predicted_answer"}     Returns each item w, Render the symbolic trace into ordered, human-readable arithmetic lines.      Ex (+2 more)

### Community 21 - "MLX GAT Bridge"
Cohesion: 0.24
Nodes (6): DenseGATLayer, GraphAwareBridge, node_ids: (B, N) - Integer tokens representing operations         node_values: (, x: (B, N, in_features) - Node features         adj_mask: (B, N, N) - Adjacency m, Graph Attention Network layer using dense adjacency matrices for MLX.     Optimi, Bridge module that processes structured reasoning traces (JSON graphs)      into

### Community 22 - "HRM Transformer Blocks"
Cohesion: 0.22
Nodes (6): HRMReasoningModule, HRMTransformerBlock, HRM Transformer block - matches HierarchicalReasoningModel_ACTV1Block, Reasoning module - matches HierarchicalReasoningModel_ACTV1ReasoningModule, RMS normalization - EXACT match to original layers.py lines 152-158, rms_norm()

### Community 23 - "HRM Reasoning Blocks"
Cohesion: 0.25
Nodes (5): HRMBlock, HRMReasoningModule, HRMForMath — Hierarchical Reasoning Model for GSM8K.  Faithful implementation of, Single transformer block used inside H-level and L-level.     Post-norm (add-the, Reasoning module — processes sequences through multiple transformer blocks.

### Community 24 - "Training Curves (v3.1)"
Cohesion: 0.22
Nodes (11): Adaptive Computation Time (ACT), Adam-atan2 Optimizer, HRM v3.1 Training Curves (ACT + Adam-atan2), Combined Loss (decreasing from ~6.7 to ~3.0), Convergence Plateau Trend, Digit-Level Accuracy (plateau ~73%), 500 Training Epochs, HRM v3.1 Model (+3 more)

### Community 25 - "GSM8K Evaluation Concepts"
Cohesion: 0.2
Nodes (10): Chain-of-Thought Answer Format, Evaluation Design, Exact-Match Accuracy, Explanation Coherence Metric, GSM8K Benchmark Dataset, HuggingFace Datasets Library, log1p Normalisation, Mean Absolute Error (MAE) (+2 more)

### Community 26 - "MLX Attention & Rotary"
Cohesion: 0.31
Nodes (7): apply_rotary_pos_emb(), Attention, HRM Layer implementations in MLX Exact match to original HRM/models/layers.py, Attention module - adapted for MLX (no FlashAttention), Rotates half the hidden dims of the input., Apply rotary position embeddings - EXACT match to original layers.py lines 30-40, rotate_half()

### Community 27 - "Export Predictions (orig)"
Cohesion: 0.28
Nodes (8): decode_digits_to_number(), Decode a digit token sequence back to an integer.          Examples:         [8,, export_sample(), load_model(), main(), Export HRM pipeline intermediate outputs for UI visualization. Generates a JSON, Run model and capture intermediate outputs for one sample., HRM Pipeline Visualizer UI

### Community 28 - "REINFORCE RL Fine-tune"
Cohesion: 0.33
Nodes (8): compute_rewards(), load_model(), parse_args(), RL Fine-tuning script for HRMForMath using REINFORCE.  Applies Policy Gradient (, Compute dense rewards for digit classification.          Args:         pred_digi, Compute REINFORCE policy gradient loss with cross-entropy regularization., reinforce_loss(), train()

### Community 29 - "Training Stats Builder"
Cohesion: 0.25
Nodes (5): _full_question(), Build ui/training_stats.json from the latest training output.  Reads:     output, Look up full question by val-set index; fall back to the truncated string., Return the symbolic trace for this val sample so the UI can render steps., _val_trace()

### Community 30 - "HRM ACT Wrapper"
Cohesion: 0.25
Nodes (5): HierarchicalReasoningModel, Complete HRM with ACT wrapper - matches HierarchicalReasoningModel_ACTV1, Set model to training mode, Set model to evaluation mode, test_model()

### Community 31 - "HRM Common Utilities"
Cohesion: 0.29
Nodes (4): Hierarchical Reasoning Model with ACT (Adaptive Computation Time) MLX implementa, _find_multiple(), Common utilities for HRM implementation Exact match to original HRM/models/commo, Find multiple - EXACT match to original layers.py line 19-20

### Community 32 - "Predict CLI Script"
Cohesion: 0.38
Nodes (5): format_digit_tokens(), load_model(), predict(), Inference script for HRMForMath — run predictions on GSM8K samples.  Usage:, Convert digit token list to human-readable string.

### Community 33 - "Curriculum Learning"
Cohesion: 0.33
Nodes (3): Three-phase curriculum strategy, CurriculumDataset, Three-phase curriculum ordering by computation step count.     Phase 0: ≤2 steps

### Community 34 - "PyTorch Checkpoint Tooling"
Cohesion: 0.47
Nodes (6): PyTorch best_model checkpoint artifact, convert_pt (PyTorch -> NPZ), Create dummy checkpoint, find_pt utility, Inspect PT checkpoint, Test PyTorch model load

### Community 35 - "Sparse Embedding"
Cohesion: 0.33
Nodes (3): CastedSparseEmbedding, Sparse embedding implementation for MLX Simplified version of original sparse_em, Sparse embedding for puzzle identifiers - simplified version     In the original

### Community 36 - "Implementation Stack (Paper)"
Cohesion: 0.5
Nodes (4): Implementation Chapter, Kaggle Notebook Environment, Python 3.12, PyTorch 2.11

## Knowledge Gaps
- **209 isolated node(s):** `Export per-sample predictions from the v3.2 PyTorch checkpoint for the UI.  Mirr`, `Same as step() but returns per-cycle (z_H, z_L) norms for visualisation.`, `Learning rate scheduling for HRM training Implements cosine decay with warmup as`, `Cosine learning rate schedule with linear warmup          Matches the original P`, `Get learning rate for given step` (+204 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HierarchicalReasoningModel` connect `HRM ACT Wrapper` to `Config & Data Loading`, `Sparse Embedding`, `Training Step & Collation`, `MLX HRM Evaluator`, `HRM Carry State`, `HRM Inner H/L Cycles`, `Init & Casted Layers`, `HRM Transformer Blocks`, `MLX Attention & Rotary`, `HRM Common Utilities`?**
  _High betweenness centrality (0.341) - this node is a cross-community bridge._
- **Why does `HRMForMath` connect `Sudoku Configs & README` to `Curriculum Learning`, `Graph-Aware Bridge (GAT)`, `Smoke Test Pipeline`, `HRM Inner H/L Cycles`, `Kaggle Training Guide`, `Init & Casted Layers`, `MLX GAT Bridge`, `HRM Reasoning Blocks`, `MLX Attention & Rotary`, `Export Predictions (orig)`, `REINFORCE RL Fine-tune`?**
  _High betweenness centrality (0.230) - this node is a cross-community bridge._
- **Why does `HRMForMath` connect `Training Step & Collation` to `Predictions HTTP Server`, `HRM ACT Wrapper`?**
  _High betweenness centrality (0.184) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `HRMForMath` (e.g. with `SwiGLU` and `Attention`) actually correct?**
  _`HRMForMath` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `HierarchicalReasoningModel_Inner` (e.g. with `SwiGLU` and `Attention`) actually correct?**
  _`HierarchicalReasoningModel_Inner` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `HierarchicalReasoningModel` (e.g. with `SwiGLU` and `Attention`) actually correct?**
  _`HierarchicalReasoningModel` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `CastedLinear` (e.g. with `HRMInnerCarry` and `HRMCarry`) actually correct?**
  _`CastedLinear` has 9 INFERRED edges - model-reasoned connections that need verification._