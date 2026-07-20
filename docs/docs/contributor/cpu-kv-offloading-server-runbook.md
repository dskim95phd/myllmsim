---
title: Running CPU KV Offloading Experiments on a Server
---

# Running CPU KV Offloading Experiments on a Server

This runbook executes the staged CPU KV offloading experiment on a Linux
server. Start with load calibration and review its results before launching
the capacity screen. Do not launch the original 1,000-session, 135-run matrix
as the first server job.

## 1. Server requirements

The analytical simulator does not require a GPU. Use a Linux x86-64 server
with:

- Docker and permission to run containers;
- at least 8 CPU cores, 16 GiB host RAM, and 30 GiB free disk space;
- a persistent filesystem for the repository and `outputs/`;
- `git`, `bash`, and `timeout` on the host.

More CPU cores help only when independent runs execute concurrently. Simulated
NPU and CPU memory sizes do not allocate corresponding amounts of host memory.

## 2. Clone the exact branch and submodules

```bash
git clone --recurse-submodules \
  --branch agent/cpu-kv-offloading \
  https://github.com/dskim95phd/myllmsim.git
cd myllmsim

git submodule sync --recursive
git submodule update --init --recursive
git status --short
git submodule status --recursive
```

The first column of every `git submodule status` line must be blank. A leading
`-` means the submodule is not initialized, and `+` means it is checked out at
a different commit.

Record the exact revisions before running:

```bash
mkdir -p outputs/cpu_kv_experiment/server
git rev-parse HEAD \
  > outputs/cpu_kv_experiment/server/root_commit.txt
git submodule status --recursive \
  > outputs/cpu_kv_experiment/server/submodule_commits.txt
```

## 3. Create and build the simulator container

Launch the container from the repository root:

```bash
./scripts/docker-sim.sh
```

The command opens a shell inside `servingsim_docker` with the repository
mounted at `/app/LLMServingSim`. In that shell, build ASTRA-Sim and install the
checked-out Chakra converter:

```bash
cd /app/LLMServingSim
./scripts/compile.sh
```

This step is mandatory. A previously built image may contain an older Chakra
package that fails on migration-only traces. Rerun it whenever the ASTRA-Sim
or Chakra submodule commit changes.

For later logins, reattach without creating another container:

```bash
docker start servingsim_docker
docker exec -it servingsim_docker bash
```

## 4. Validate the environment

Run these checks inside the container:

```bash
cd /app/LLMServingSim

python3 -m unittest \
  tests.test_session_kv_generator \
  tests.test_session_kv_retention \
  tests.test_kv_offloading_foundation -q

python3 -m workloads.generators session-kv --help >/dev/null
test -x \
  astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
```

The current expected unit-test result is 72 passing tests.

## 5. Create a reproducible result directory

Use a separate directory for each server batch:

```bash
cd /app/LLMServingSim

export CPU_KV_RUN_ID="$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)"
export CPU_KV_RUN_ROOT="outputs/cpu_kv_experiment/server/${CPU_KV_RUN_ID}"
mkdir -p "${CPU_KV_RUN_ROOT}/workloads" "${CPU_KV_RUN_ROOT}/runs"

git rev-parse HEAD > "${CPU_KV_RUN_ROOT}/root_commit.txt"
git submodule status --recursive \
  > "${CPU_KV_RUN_ROOT}/submodule_commits.txt"
```

Do not reuse a result directory. The generated workload and its summary must
remain unchanged across the policies being compared.

## 6. Define a single-run helper

Paste this function into the container shell. It records the command, copied
configuration, process log, wall time, and exit status. The six-hour timeout
prevents an unattended stalled run from occupying the server indefinitely.

```bash
run_cpu_kv_case() {
  local label="$1"
  local config="$2"
  local workload="$3"
  local sessions="$4"
  local run_dir="${CPU_KV_RUN_ROOT}/runs/${label}"
  local start_epoch
  local end_epoch
  local status

  mkdir -p "${run_dir}"
  cp "${config}" "${run_dir}/cluster_config.json"
  cp "${workload}.summary.json" "${run_dir}/workload.summary.json"

  printf '%s\n' \
    "python3 -m serving --cluster-config ${config} --dtype bfloat16 --block-size 16 --dataset ${workload} --output ${run_dir}/requests.csv --num-reqs ${sessions} --log-level WARNING --log-interval 10" \
    > "${run_dir}/command.txt"

  start_epoch="$(date +%s)"
  set -o pipefail
  timeout --signal=TERM --kill-after=30s 6h \
    python3 -m serving \
      --cluster-config "${config}" \
      --dtype bfloat16 \
      --block-size 16 \
      --dataset "${workload}" \
      --output "${run_dir}/requests.csv" \
      --num-reqs "${sessions}" \
      --log-level WARNING \
      --log-interval 10 \
    2>&1 | tee "${run_dir}/run.log"
  status="${PIPESTATUS[0]}"
  set +o pipefail
  end_epoch="$(date +%s)"

  printf '%s\n' "$((end_epoch - start_epoch))" \
    > "${run_dir}/wall_seconds.txt"
  printf '%s\n' "${status}" > "${run_dir}/exit_code.txt"
  return "${status}"
}
```

Exit code `0`, a `requests.csv` file, and `All Request Has Been Exited` in the
log together indicate a completed run. Exit code `124` indicates timeout.

## 7. Gate 1: calibrate offered load

Generate one 20-session workload for each arrival rate. Use seed 7 for this
first calibration:

```bash
for rate in 0.5 1.0 1.5 2.0; do
  rate_label="${rate/./p}"
  python3 -m workloads.generators session-kv \
    --num-sessions 20 \
    --session-rate "${rate}" \
    --seed 7 \
    --gap-profile mixed \
    --output "${CPU_KV_RUN_ROOT}/workloads/rate_${rate_label}_seed7.jsonl"
done
```

Run Recompute and 16 GiB Session offload against each identical workload:

```bash
for rate in 0.5 1.0 1.5 2.0; do
  rate_label="${rate/./p}"
  workload="${CPU_KV_RUN_ROOT}/workloads/rate_${rate_label}_seed7.jsonl"

  run_cpu_kv_case \
    "calibrate_rate_${rate_label}_recompute" \
    configs/cluster/single_node_session_kv_experiment_recompute.json \
    "${workload}" 20 || true

  run_cpu_kv_case \
    "calibrate_rate_${rate_label}_offload_16gb" \
    configs/cluster/single_node_session_kv_experiment_offload_16gb.json \
    "${workload}" 20 || true
done
```

Treat the largest Recompute rate that completes without a persistent
no-progress interval as the provisional saturation rate. A known bad state is
zero running requests, a growing waiting queue, and NPU memory remaining near
100%. Do not use such a run as a stable-load result.

Select low and high rates near `0.6 * lambda_sat` and
`0.9 * lambda_sat`. Generate new 50-session workloads for those exact rates
instead of reusing a workload generated at a different rate.

## 8. Gate 2: screen CPU capacity

Start with 4, 16, 64, and 256 GiB. Generate capacity-specific configs under
the result directory so the tracked template remains unchanged:

```bash
for capacity in 4 16 64 256; do
  output_config="${CPU_KV_RUN_ROOT}/offload_${capacity}gb.json"
  python3 - "${capacity}" "${output_config}" <<'PY'
import json
import sys

capacity = int(sys.argv[1])
output_path = sys.argv[2]
template = "configs/cluster/single_node_session_kv_experiment_offload_16gb.json"

with open(template, encoding="utf-8") as input_file:
    config = json.load(input_file)
config["nodes"][0]["cpu_mem"]["mem_size"] = capacity
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=2)
    output_file.write("\n")
PY
done
```

For each selected load, generate one 50-session workload and run Recompute
once plus every capacity against that same file. Replace `LOW_RATE` and
`HIGH_RATE` below with the calibrated numeric values:

```bash
for load_spec in low:LOW_RATE high:HIGH_RATE; do
  load_label="${load_spec%%:*}"
  rate="${load_spec##*:}"
  workload="${CPU_KV_RUN_ROOT}/workloads/${load_label}_seed7.jsonl"

  python3 -m workloads.generators session-kv \
    --num-sessions 50 \
    --session-rate "${rate}" \
    --seed 7 \
    --gap-profile mixed \
    --output "${workload}"

  run_cpu_kv_case \
    "screen_${load_label}_recompute" \
    configs/cluster/single_node_session_kv_experiment_recompute.json \
    "${workload}" 50 || true

  for capacity in 4 16 64 256; do
    run_cpu_kv_case \
      "screen_${load_label}_offload_${capacity}gb" \
      "${CPU_KV_RUN_ROOT}/offload_${capacity}gb.json" \
      "${workload}" 50 || true
  done
done
```

Review this screen before adding 8, 32, and 128 GiB or more seeds. Expand only
around the observed capacity knee.

## 9. Run safely after disconnecting SSH

For a long batch, save the commands above in a script such as
`run_server_batch.sh` inside the result directory. Start it from the host with
the existing container running:

```bash
docker start servingsim_docker

nohup docker exec servingsim_docker bash -lc \
  'cd /app/LLMServingSim && bash outputs/cpu_kv_experiment/server/RUN_ID/run_server_batch.sh' \
  > outputs/cpu_kv_experiment/server/RUN_ID/launcher.log 2>&1 &

echo $! > outputs/cpu_kv_experiment/server/RUN_ID/launcher.pid
```

Replace `RUN_ID` with the created directory name. Monitor it from the host:

```bash
tail -f outputs/cpu_kv_experiment/server/RUN_ID/launcher.log
docker stats servingsim_docker
find outputs/cpu_kv_experiment/server/RUN_ID/runs \
  -name exit_code.txt -print -exec cat {} \;
```

Keep the calibration batch serial so wall-time measurements remain useful.
After calibration, at most two independent screening runs should execute in
parallel initially. Increase concurrency only if CPU utilization, memory, and
disk latency leave sufficient headroom. Parallel execution does not change
simulated time, but it can distort wall-time estimates.

## 10. Collect and transfer results

Preserve the entire run directory. At minimum, every completed case must have:

- `cluster_config.json` and `workload.summary.json`;
- `command.txt`, `run.log`, `wall_seconds.txt`, and `exit_code.txt`;
- `requests.csv` and any KV-offload metric sidecar produced by the simulator;
- root and recursive submodule commit IDs.

Archive the batch from the host:

```bash
CPU_KV_RUN_ID=REPLACE_WITH_THE_SERVER_RUN_DIRECTORY
tar -C outputs/cpu_kv_experiment/server \
  -czf "cpu_kv_${CPU_KV_RUN_ID}.tar.gz" "${CPU_KV_RUN_ID}"
sha256sum "cpu_kv_${CPU_KV_RUN_ID}.tar.gz" \
  > "cpu_kv_${CPU_KV_RUN_ID}.tar.gz.sha256"
```

Generated traces are deleted by default. Add `--no-cleanup-inputs` only for a
single failing case because retained Chakra graphs can consume substantial
disk space.

## 11. Stop conditions and next decision

Stop the batch and inspect the log when:

- ASTRA-Sim or Chakra exits nonzero;
- no request progress occurs for 30 simulated seconds while work is waiting;
- the same configuration times out twice;
- the server approaches its memory or disk limit.

After Gate 2, compare throughput, p95/p99 latency, CPU hit rate, recomputed
prompt tokens, migrated bytes, and capacity drops. Run longer workloads and
additional seeds only for Recompute, the apparent knee, one point below the
knee, and the maximum-capacity reference.
