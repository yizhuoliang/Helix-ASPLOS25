# Human Run Instructions: Original LLaMA and `qwen3_30b_hacked`

This guide explains how to run Helix real-system workflow for:
- Original dummy LLaMA (`llama30b`)
- Dummy Qwen3-30B simulation (`qwen3_30b_hacked`, implemented via Qwen2MoE backend)

The examples use one machine with 2 GPUs (like 2xL40S), but the steps are not tied to L40S. You can adapt GPU count, VRAM usage, and network addresses.

## 1) Prerequisites

- Linux machine with NVIDIA GPUs and working CUDA driver/runtime.
- Conda env (example name: `helix`) with required Python deps.
- Helix communication module built (`llm_host` / `llm_worker` importable).

Recommended quick checks:

```bash
nvidia-smi
conda run -n helix python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
conda run -n helix python -c "import llm_host, llm_worker; print('comm ok')"
```

## 2) Shared Runtime Setup

Use these environment variables in every shell running host/worker:

```bash
export HELIX_PY=/home/yizhuoliang/miniconda3/envs/helix/bin/python
export PYTHONPATH=/path/to/Helix-ASPLOS25
export HELIX_CONFIG_BROADCAST_ADDR=tcp://127.0.0.1:5600
```

Notes:
- If port is occupied, pick another one (for example `5601`, `5602`, ...).
- For multi-process single-node runs, assign distinct loopback IPs via `HELIX_LOCAL_IP`.

## 3) Run Original LLaMA Dummy Workflow

### 3.1 Config and scripts

- Config: `local_real_sys_l40s/real_sys_config.txt`
- Worker: `local_real_sys_l40s/run_worker_random.py`
- Host: `local_real_sys_l40s/run_host_fixed_random.py`

### 3.2 Launch workers + host

```bash
CUDA_VISIBLE_DEVICES=0 HELIX_LOCAL_IP=127.0.0.2 $HELIX_PY -u local_real_sys_l40s/run_worker_random.py --vram-usage 0.72 > local_real_sys_l40s/worker1_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 HELIX_LOCAL_IP=127.0.0.3 $HELIX_PY -u local_real_sys_l40s/run_worker_random.py --vram-usage 0.72 > local_real_sys_l40s/worker2_llama.log 2>&1 &

HELIX_LOCAL_IP=127.0.0.1 $HELIX_PY -u local_real_sys_l40s/run_host_fixed_random.py \
  --init-wait-s 60 --timeout-s 360 --num-requests 1 --input-len 8 --output-len 4 \
  > local_real_sys_l40s/host_llama.log 2>&1
```

### 3.3 Parse results

```bash
PYTHONPATH=/path/to/Helix-ASPLOS25 conda run -n helix python -c \
"import runpy; m=runpy.run_path('examples/real_sys/step4_parse_results.py'); m['parse_result']('local_real_sys_l40s/result_fixed_random/events.txt', warm_up_time=0, finish_time=1)"
```

## 4) Run `qwen3_30b_hacked` Dummy Workflow

`qwen3_30b_hacked` means: Qwen3-30B-like dimensions running through vLLM 0.4 Qwen2MoE code path.

### 4.1 Config and scripts

- Model config dir: `artifact_evaluation/single_cluster/models/qwen3_30b_hacked`
- Cluster config: `local_real_sys_l40s/real_sys_config_qwen3_30b_hacked.txt`
- Worker: `local_real_sys_l40s/run_worker_qwen3_30b_hacked_random.py`
- Host: `local_real_sys_l40s/run_host_fixed_qwen3_30b_hacked_random.py`

### 4.2 Launch workers + host

```bash
CUDA_VISIBLE_DEVICES=0 HELIX_LOCAL_IP=127.0.0.2 $HELIX_PY -u local_real_sys_l40s/run_worker_qwen3_30b_hacked_random.py --vram-usage 0.72 > local_real_sys_l40s/worker1_qwen3_30b_hacked.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 HELIX_LOCAL_IP=127.0.0.3 $HELIX_PY -u local_real_sys_l40s/run_worker_qwen3_30b_hacked_random.py --vram-usage 0.72 > local_real_sys_l40s/worker2_qwen3_30b_hacked.log 2>&1 &

HELIX_LOCAL_IP=127.0.0.1 $HELIX_PY -u local_real_sys_l40s/run_host_fixed_qwen3_30b_hacked_random.py \
  --init-wait-s 70 --timeout-s 360 --num-requests 1 --input-len 8 --output-len 4 \
  > local_real_sys_l40s/host_qwen3_30b_hacked.log 2>&1
```

### 4.3 Parse results

```bash
PYTHONPATH=/path/to/Helix-ASPLOS25 conda run -n helix python -c \
"import runpy; m=runpy.run_path('examples/real_sys/step4_parse_results.py'); m['parse_result']('local_real_sys_l40s/result_qwen3_30b_hacked_fixed_random/events.txt', warm_up_time=0, finish_time=1)"
```

## 5) Adapting to Other Machines (Not L40S-specific)

- **GPU type/count**: works on other NVIDIA GPUs if memory is enough for selected layer split and `--vram-usage`.
- **Layer partition**: edit `start_layer`/`end_layer` in real-system config to match your GPU count.
- **IPs and ports**:
  - Single-node multi-process: use distinct `127.0.0.x` addresses.
  - Multi-node: use real NIC IPs reachable between machines.
  - Change `HELIX_CONFIG_BROADCAST_ADDR` if port conflicts.
- **Throughput/latency**: depends heavily on GPU generation, interconnect, and scheduling params.

## 6) Troubleshooting

1. `model_type ... not recognized`
   - For `qwen3_30b_hacked`, keep config `architectures: ["Qwen2MoeForCausalLM"]` and compatibility `model_type: "llama"`.

2. `max_num_batched_tokens smaller than max_model_len`
   - Reduce `max_position_embeddings` in model config or increase scheduler batch-token cap in runtime.

3. Host starts but request never finishes
   - Check worker logs for crashes.
   - Check port conflicts and `HELIX_CONFIG_BROADCAST_ADDR`.
   - Confirm worker `HELIX_LOCAL_IP` matches config file IPs.

4. Low throughput
   - Increase `--vram-usage` carefully.
   - Install FlashAttention if supported.
   - Increase requests / tune scheduling for non-toy runs.
