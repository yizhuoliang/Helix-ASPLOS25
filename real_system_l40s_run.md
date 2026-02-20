# Helix ASPLOS'25 Artifact: Real-System Workflow Trial (2x L40S)

This log documents an end-to-end attempt to run the **real-system (prototype runtime)** portion of the Helix ASPLOS'25 artifact on a single node with **2x NVIDIA L40S** GPUs.

Constraints / intent
- CUDA is already installed on this node; I avoid reinstalling CUDA.
- Python dependencies are installed into the existing conda environment `helix`.
- This node is resource constrained; the goal is to validate the workflow with a small/contained setup rather than reproduce full paper-scale numbers.

---

## 0. Repo Snapshot

Workspace: `/home/yizhuoliang/Helix-ASPLOS25`

---

## 1. Environment & Hardware Check

GPU / driver
```bash
nvidia-smi
```

Observed
- Driver: 570.133.20
- Reported CUDA runtime: 12.8
- GPUs: 2x NVIDIA L40S, 46068 MiB each

CUDA toolkit
```bash
nvcc --version
```

Observed
- `nvcc` is not present on this node (toolkit not installed in PATH). I did not install it.

Build tools
```bash
gcc --version
g++ --version
cmake --version
make --version
```

Observed
- gcc/g++ 13.3.0, cmake 3.28.3, make 4.3

---

## 2. Python Environment (`conda env: helix`)

Conda env list (confirm `helix` exists)
```bash
conda info --envs
```

Observed
- `helix` located at `/home/yizhuoliang/miniconda3/envs/helix`
- Python: 3.10.19

Initial package state (before installing)
- `torch`, `vllm`, `ray`, `pybind11` (python), `zmq` were not installed in `helix`.

Installed conda packages (into `helix`)
```bash
conda install -n helix -c conda-forge -y libstdcxx-ng pybind11 zeromq cppzmq
```

Installed pip packages (into `helix`)
```bash
python -m pip install -U pip setuptools wheel
python -m pip install "numpy~=1.26"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.2.2+cu121"
python -m pip install "vllm==0.4.0.post1"
python -m pip install "transformers==4.39.3"  # pin for vllm 0.4.x compatibility
```

Verified versions
```bash
python -c "import transformers, torch, vllm; print(transformers.__version__); print(torch.__version__); print(vllm.__version__)"
```

Observed
- `transformers==4.39.3`
- `torch==2.1.2+cu121` with CUDA available and 2 visible GPUs
- `vllm==0.4.0.post1`

---

## 3. Build Helix Communication Framework (PyBind + ZeroMQ)

Fixes applied for this node
- `llm_sys/comm/CMakeLists.txt`: remove hardcoded Torch path; instead query `torch.utils.cmake_prefix_path` from the active Python and require `cppzmq`.
- `llm_sys/comm/src/poller.h`: replace `zmq::poller_t` usage with `zmq::poll` + `zmq::pollitem_t` for broader cppzmq compatibility.

Build command (run inside `helix` env)
```bash
cd llm_sys/comm
CUDACXX=/usr/local/cuda/bin/nvcc CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  bash build.sh
```

Notes
- I did not install CUDA. `nvcc` already existed at `/usr/local/cuda/bin/nvcc`; it just was not in `PATH`, so I pointed CMake to it via `CUDACXX`.

Import verification
```bash
python -c "import llm_host, llm_worker"
```

Unit test (message encoding/decoding)
```bash
./build/test_msg
```

Observed
- `Test Passed!`

---

## 4. Minimal Single-Node Real-System Config (Loopback IPs)

Goal
- Run host + 2 worker processes on the *same machine* (2 GPUs) while satisfying the real-system runtime's constraint that each worker must have a unique `ip_address` in the runtime config.

Approach
- Use Linux loopback IP aliases in `127.0.0.0/8`:
  - Host: `127.0.0.1`
  - Worker on GPU0: `127.0.0.2`
  - Worker on GPU1: `127.0.0.3`
- Use the `llama30b` dummy model config from `artifact_evaluation/single_cluster/models/llama30b` (60 layers) and split layers across the two workers.

Runtime config written
- `local_real_sys_l40s/real_sys_config.txt`

Broadcast port selection
- The artifact default broadcast port is `5000`, but on this node it is already occupied by Docker (`docker-proxy`).
- I used `tcp://127.0.0.1:5500` instead.

Code changes to support single-node loopback
- `llm_sys/utils.py`: allow overrides via env vars
  - `HELIX_CONFIG_BROADCAST_ADDR` (e.g. `tcp://127.0.0.1:5500`)
  - `HELIX_LOCAL_IP` (so each process can advertise a different loopback address)
- `llm_sys/worker.py`, `llm_sys/heuristic_host.py`, `llm_sys/maxflow_host.py`: remove hard-coded `10.x.x.x` assertions.

---

## 5. Run Host + 2 Workers (Random/Swarm)

I used **random** scheduling, but with a deterministic 2-stage pipeline because each node has only one `out_node` in the config:
- host(0) -> worker1(1) -> worker2(2) -> host(0)

Commands
```bash
# Use the helix conda env's python directly (to reliably capture PIDs)
HELIX_PY=/home/yizhuoliang/miniconda3/envs/helix/bin/python

export PYTHONPATH=/home/yizhuoliang/Helix-ASPLOS25
export HELIX_CONFIG_BROADCAST_ADDR=tcp://127.0.0.1:5500

# workers (background)
CUDA_VISIBLE_DEVICES=0 HELIX_LOCAL_IP=127.0.0.2 $HELIX_PY -u local_real_sys_l40s/run_worker_random.py --vram-usage 0.85 > local_real_sys_l40s/worker1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 HELIX_LOCAL_IP=127.0.0.3 $HELIX_PY -u local_real_sys_l40s/run_worker_random.py --vram-usage 0.85 > local_real_sys_l40s/worker2.log 2>&1 &

# host (foreground)
HELIX_LOCAL_IP=127.0.0.1 $HELIX_PY -u local_real_sys_l40s/run_host_fixed_random.py \
  --init-wait-s 60 --timeout-s 180 --num-requests 1 --input-len 8 --output-len 4 \
  > local_real_sys_l40s/host_fixed3.log 2>&1

# after host exits, terminate workers (they otherwise run forever)
pkill -f "python.*local_real_sys_l40s/run_worker_random.py" || true
```

Outputs
- Host log: `local_real_sys_l40s/host_fixed3.log`
- Worker logs: `local_real_sys_l40s/worker1.log`, `local_real_sys_l40s/worker2.log`
- Result files: `local_real_sys_l40s/result_fixed_random/events.txt`, `local_real_sys_l40s/result_fixed_random/query_route.txt`

Observed (from `local_real_sys_l40s/host_fixed3.log`)
- Request finished end-to-end; route reported by the system: `[1, 2, 0]`

Important note about host exit behavior
- The underlying `llm_host` binding starts detached C++ threads. In my first attempt using the stock host loop, the process often aborted at interpreter shutdown with `zmq::error_t: Context was terminated` (after logs were written).
- For this workflow trial, `local_real_sys_l40s/run_host_fixed_random.py` ends with `os._exit(0)` after writing logs to avoid that teardown abort and to return exit code 0.

---

## 6. Parse Results

Command
```bash
python -c "import runpy; m=runpy.run_path('examples/real_sys/step4_parse_results.py'); m['parse_result']('local_real_sys_l40s/result_fixed_random/events.txt', warm_up_time=0, finish_time=1)"
```

Observed
- Avg prompt latency: 0.121s
- Avg decode latency: 0.104s
- Throughput: 14.9 Tokens/s

---

## 5. Run Host + 2 Workers (Random/Swarm)

Commands and outputs will be appended here.

---

## 6. Parse Results

Commands and outputs will be appended here.
