# AGENTS.md — Write-Ahead Log

## 2026-02-20: Qwen3-30B MoE Support Investigation

### Goal
Make Helix prototype support dummy Qwen3-30B-A3B weights with real MoE architecture,
reusing the existing pipeline-parallelism division logic (which is model-agnostic).

### Status: ANALYSIS COMPLETE — Ready for Implementation

---

### Investigation Findings

#### Finding 1: vLLM 0.4 Already Has Qwen2MoeForCausalLM (NO UPGRADE NEEDED)

The installed vLLM 0.4.0.post1 already registers three MoE model classes:
- `MixtralForCausalLM`
- `QuantMixtralForCausalLM`
- **`Qwen2MoeForCausalLM`** ← This is our path forward

Source: `vllm/model_executor/models/qwen2_moe.py` (458 lines)

This eliminates the need to upgrade vLLM (which would require vLLM ≥0.8.4 + torch ≥2.6
+ transformers ≥4.56.3 — a massive, risky change).

#### Finding 2: Qwen2MoeDecoderLayer Has IDENTICAL Forward Signature to LlamaDecoderLayer

Both follow the exact same contract:
```python
def forward(self, positions, hidden_states, kv_cache, attn_metadata, residual)
    → (hidden_states, residual)
```

Key difference: `Qwen2MoeDecoderLayer.__init__` requires an extra `layer_idx` parameter:
```python
# Llama:      LlamaDecoderLayer(config, linear_method)
# Qwen2Moe:   Qwen2MoeDecoderLayer(config, layer_idx, linear_method)
```

The `layer_idx` determines MoE vs dense layer via `(layer_idx + 1) % decoder_sparse_step`.
With `decoder_sparse_step=1` (all layers are MoE), every layer uses `Qwen2MoeSparseMoeBlock`.

#### Finding 3: C++ Communication Layer is Size-Agnostic

- `compute_worker.h:345-346`: Tensors treated as opaque FP16 byte buffers
- `msg.h:55-71`: Header only tracks `request_id`, `num_tokens`, routing — no `hidden_size`
- `llm_sys/worker.py:95`: `hidden_size` queried dynamically via `engine.model_config.get_hidden_size()`
- Changing from hidden_size=6656 (LLaMA-30B) to 2048 (Qwen3-30B) requires zero C++ changes

#### Finding 4: KV-Cache Allocation is Model-Agnostic

`LayerwiseWorker.init_cache_engine()` patches `get_num_layers() → 1` to allocate KV cache
for a single layer at a time. Cache block sizes are computed from `num_attention_heads`,
`num_kv_heads`, and `head_dim` — all from the HF config. Works for any model.

#### Finding 5: Pipeline Integration Surface is Minimal

The only Helix-specific code that needs to "know" the model is:
1. `llm_sys/engine/llama.py` → the `LlamaPipelineStage` class (template to replicate)
2. The `_MODELS` registry entry mapping architecture name → pipeline stage class
3. The model config directory (config.json + tokenizer files)

Everything else (model_runner.py, worker.py, exec_engine.py, common.py, C++ comm) is
model-agnostic.

### Resolved Open Questions

| Question | Answer |
|----------|--------|
| Same `(hidden_states, residual)` return signature? | ✅ Yes, identical to LlamaDecoderLayer |
| KV-cache allocation works for MoE layers? | ✅ Yes, attention is same; MoE only changes MLP |
| C++ comm hardcodes tensor sizes? | ✅ No, opaque byte buffers with dynamic sizing |
| Need vLLM upgrade? | ✅ No, Qwen2MoeForCausalLM already in vLLM 0.4 |

### Concrete Migration Plan

**Approach: Create `Qwen3_30b_hackedPipelineStage` using existing vLLM 0.4 `Qwen2MoeDecoderLayer`**

#### Step 1: Create Qwen3 hacked model config directory
- Path: `artifact_evaluation/single_cluster/models/qwen3_30b_hacked/`
- `config.json`: Qwen3-30B dimensions adapted for Qwen2Moe code path
  - `architectures: ["Qwen2MoeForCausalLM"]`, `model_type: "qwen2_moe"`
  - `hidden_size: 2048`, `num_hidden_layers: 48`, `num_experts: 128`,
    `num_experts_per_tok: 8`, `moe_intermediate_size: 768`
  - `shared_expert_intermediate_size: 0` (no shared expert, simplifies init)
  - `vocab_size: 32000` (matching LLaMA tokenizer to avoid mismatch)
  - `decoder_sparse_step: 1` (all layers are MoE)
- Copy `tokenizer.json` and `tokenizer_config.json` from llama30b directory

#### Step 2: Create `llm_sys/engine/qwen3_30b_hacked.py`
- Mirror `LlamaPipelineStage` structure with 3 changes:
  1. Import `Qwen2MoeDecoderLayer` instead of `LlamaDecoderLayer`
  2. Pass `layer_id` as `layer_idx` to decoder layer constructor
  3. Register as `_MODELS["Qwen2MoeForCausalLM"]`

#### Step 3: Import the new module in Helix's engine init
- Add `import llm_sys.engine.qwen3_30b_hacked` alongside existing `import llm_sys.engine.llama`
  in the worker startup path

#### Step 4: Create run configs and test
- Create `local_real_sys_l40s/real_sys_config_qwen3_30b_hacked.txt` with 48-layer split
- Create worker and host scripts
- Run end-to-end with dummy weights

### Risks
1. **128 experts × fused_moe kernel**: vLLM 0.4's Triton `fused_moe` should handle 128
   experts but hasn't been tested at this scale. Fallback: reduce to 64 or 32 experts.
2. **Memory**: 128 experts per layer ≈ 1.2GB. With 24 layers per GPU (48/2), that's
   ~29GB for model weights alone. KV cache needs additional memory. May need to reduce
   layers per GPU or expert count.
3. **head_dim mismatch**: Real Qwen3-30B uses head_dim=128 with hidden_size=2048 (meaning
   attention has an internal expansion). Qwen2Moe code computes head_dim=hidden_size/num_heads
   =64. For dummy weights this is fine; the MoE routing architecture is what we're testing.

---

## 2026-02-20: Implementation Start — Qwen2Moe-backed Qwen3-30B Dummy Run

### Status: IN_PROGRESS — Executing Migration Plan

### Write-Ahead Execution Plan
1. Add `Qwen3_30b_hackedPipelineStage` in `llm_sys/engine/qwen3_30b_hacked.py`, mirroring
   `LlamaPipelineStage` and preserving Helix's existing layerwise execution contract.
2. Register/import this stage in engine startup so architecture
   `Qwen2MoeForCausalLM` resolves to the Helix pipeline stage.
3. Create model config directory `artifact_evaluation/single_cluster/models/qwen3_30b_hacked`
   with Qwen3-30B-like MoE dimensions (48 layers, 128 experts, topk=8, hidden=2048) and
   `shared_expert_intermediate_size=0`.
4. Create/adjust local run configs/scripts for 2xL40S single-node loopback execution.
5. Run end-to-end real-system workflow with dummy weights and collect logs/results.
6. Document full steps, rationale, issues, results, concerns, and realism limits in a
   dedicated markdown file.

### Rationale
- This path reuses existing validated Helix runtime changes from the prior LLaMA run.
- It avoids risky dependency upgrades while still executing real MoE routing/kernels.
- It satisfies the user goal: dummy weights, but real Qwen3-like MoE computation shape.

### Completion Log
- Added `llm_sys/engine/qwen3_30b_hacked.py` and registered `Qwen2MoeForCausalLM` pipeline stage.
- Imported stage module in `llm_sys/worker.py` startup path.
- Added model config directory `artifact_evaluation/single_cluster/models/qwen3_30b_hacked`.
- Added local run files under `local_real_sys_l40s` for 48-layer two-way split.
- Ran end-to-end on 2xL40S with dummy weights and MoE architecture active.

### Runtime Notes (Observed)
1. Initial failure: `model_type: qwen2_moe` unsupported by transformers 4.39.3 AutoConfig.
   - Mitigation: switched to `model_type: llama` while preserving
     `architectures: ["Qwen2MoeForCausalLM"]`.
2. Initial failure: `max_model_len` > `max_num_batched_tokens`.
   - Mitigation: reduced local `max_position_embeddings` to 4096.

### Final Status: SUCCESSFUL E2E RUN
- Host completed prompt+decode flow.
- Route observed: `[1, 2, 0]` with layer partitions `[0,24)` and `[24,48)`.
- Results saved in `local_real_sys_l40s/result_qwen3_30b_hacked_fixed_random`.
- Parsed metrics:
  - Avg prompt latency: `0.060s`
  - Avg decode latency: `0.030s`
  - Throughput: `74.3 Tokens/s`
