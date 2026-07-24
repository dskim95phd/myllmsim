---
title: Running CPU KV Offloading Experiments on a Server
---

# Running CPU KV Offloading Experiments on a Server

The checked-in experiment runner generates every workload and derived cluster
configuration, runs the staged matrix, validates direct IPC against transport
oracle, and writes resumable records and summaries. Run it from the repository
root inside the simulator container.

## 1. Server requirements

The analytical simulator does not require a GPU. Use a Linux x86-64 server
with:

- Docker and permission to run containers;
- at least 8 CPU cores, 16 GiB host RAM, and 30 GiB free disk space;
- a persistent filesystem for the repository and `outputs/`;
- `git` and `bash` on the host.

More CPU cores help when independent simulator cases run concurrently. The
runner is serial by default so its wall-time measurements remain comparable.
It pins the common OpenMP and BLAS thread counts to one per child, so
`--workers N` means at most `N` independent Python + ASTRA-Sim process trees.
Simulated NPU and CPU capacities do not allocate the same amount of host
memory.

An Intel Xeon 6767P has 64 physical cores and 128 threads. Start this
experiment at `--workers 8`, observe host RAM, CPU use, and storage latency,
then try 12 or 16. Do not start at 64: graph conversion, process startup,
filesystem traffic, and shared memory bandwidth keep scaling from being
linear. Keep the timing stage serial even on this CPU. See the
[Intel Xeon 6767P specifications](https://www.intel.com/content/www/us/en/products/sku/241845/intel-xeon-6767p-processor-336m-cache-2-40-ghz/specifications.html).

## 2. Clone and build

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

The required submodules must have a blank first status column. A leading `+`
means a revision mismatch. The nested
`astra-sim/extern/network_backend/ns-3` entry is the exception: it has
`update = none`, so a leading `-` is expected and does not block this
analytical experiment.

Create the simulator container:

```bash
./scripts/docker-sim.sh
```

Inside the container, build ASTRA-Sim and install the checked-out Chakra fork:

```bash
cd /app/LLMServingSim
./scripts/compile.sh
```

Rerun `scripts/compile.sh` whenever the ASTRA-Sim, Chakra, or analytical-
network submodule revision changes.

## 3. Validate the checkout

Run inside the container:

```bash
cd /app/LLMServingSim

python3 -m unittest \
  tests.test_session_kv_generator \
  tests.test_session_kv_retention \
  tests.test_kv_offloading_foundation \
  tests.test_cpu_kv_experiment_runner -q

python3 scripts/run_cpu_kv_experiment.py --help
test -x \
  astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
```

## 4. Run the complete pilot

Choose one result directory and reuse it when resuming:

```bash
cd /app/LLMServingSim

RUN_ID="$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="outputs/cpu_kv_experiment/server/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" \
  pilot 2>&1 | tee "${RUN_ROOT}/launcher.log"
```

The pilot performs the following steps without manual config editing:

1. paired 10-session Recompute and 16 GiB Session-offload timing runs;
2. paired 20-session calibration runs at 0.5, 1.0, 1.5, and 2.0 sessions/s;
3. low/high 50-session screens for Recompute, Active offload, Session offload
   at 4/16/64/256 GiB, and Capacity oracle;
4. one high-load 16 GiB Session-offload transport-oracle run;
5. exact comparison of direct/oracle request CSV, KV sidecar, simulated clock,
   and completion-sequence digests.

Every case has a one-hour timeout by default. Change it with
`--case-timeout-seconds`, placed before `pilot`. A timed-out or interrupted
batch can be resumed using the exact same command. Completed cases are checked
against their config, workload, and output schema before being skipped.

### Split the pilot and use multiple cores

The pilot can be stopped between stages and resumed from the same `RUN_ROOT`.
Run the wall-time pair alone, then parallelize the independent calibration and
screen cases:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 1 \
  pilot --stages timing \
  2>&1 | tee "${RUN_ROOT}/pilot-timing.log"

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  pilot --stages calibration \
  2>&1 | tee "${RUN_ROOT}/pilot-calibration.log"

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  pilot --stages screen \
  2>&1 | tee "${RUN_ROOT}/pilot-screen.log"

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 1 \
  pilot --stages validation \
  2>&1 | tee "${RUN_ROOT}/pilot-validation.log"
```

Later stages read the rates selected by calibration from `pilot.json`.
Validation also reuses the screen's recorded session count, seed, and gap
profile, so they do not need to be repeated. Keep 16 in
`--screen-capacities`: the exact direct/oracle validation pair uses the 16 GiB
Session-offload case.

## 5. Review the pilot

```bash
cat "${RUN_ROOT}/pilot.json"
column -s, -t < "${RUN_ROOT}/summary.csv" | less -S
```

`pilot.json` records `lambda_sat`, low/high/overload rates, and the transport-
oracle validation result. `summary.csv` contains operational status, wall and
simulated time, request/session latency percentiles, session makespan,
throughput, migrations, Recompute preemptions/discarded KV/rebuilt tokens,
NPU/CPU hits, misses, session recomputed tokens, and capacity drops.

If `calibration_upper_bound_reached` is `true`, the highest tested rate also
completed. Create a new run directory and extend the range before treating
`lambda_sat` as a saturation estimate:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "outputs/cpu_kv_experiment/server/${RUN_ID}_extended" \
  pilot \
  --calibration-rates 0.5,1,1.5,2,3,4
```

Inspect any failed case under `runs/<case>/run.log`. Do not start confirmation
until direct/oracle validation passes and the selected low and high rates are
scientifically acceptable.

## 6. Run the confirmation matrix

The default confirmation matrix uses the rates selected by the pilot, 1,000
sessions, seeds 7/17/29/43/71, and CPU capacities
4/8/16/32/64/128/256 GiB:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  confirm 2>&1 | tee "${RUN_ROOT}/confirm.log"
```

Each `(load, seed)` workload is generated once and reused by every policy and
capacity. The matrix includes Recompute, Active offload, Session offload,
and Capacity oracle.

To use a smaller confirmation before the full matrix:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" \
  confirm \
  --sessions 200 \
  --seeds 7 \
  --capacities 4,16,64,256
```

The runner includes session count, seed, and capacity in confirmation case
identities, so a smaller confirmation and the full matrix can share the pilot
run root. Use a separate run root when changing the workload gap profile or
calibration design.

The confirmation matrix can also be divided by load without losing resume or
summary behavior:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  confirm --loads low

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  confirm --loads high

python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" --workers 8 \
  confirm --loads overload
```

For still smaller jobs, combine `--loads` with a subset of `--seeds` and
`--capacities`. Rerunning later with the full lists skips the already validated
cases. While tuning worker count, compare total elapsed time for the same small
matrix at 8, 12, and 16 workers; per-case `wall_seconds` is intentionally not a
clean performance measurement under contention.

## 7. Run after disconnecting SSH

From the host, start confirmation in the existing container:

```bash
RUN_ROOT="outputs/cpu_kv_experiment/server/REPLACE_WITH_RUN_ID"

docker start servingsim_docker
docker exec -d -w /app/LLMServingSim servingsim_docker \
  bash -lc "python3 scripts/run_cpu_kv_experiment.py \
    --run-root '${RUN_ROOT}' confirm \
    > '${RUN_ROOT}/confirm.log' 2>&1"
```

Monitor from the host:

```bash
tail -f "${RUN_ROOT}/confirm.log"
docker stats servingsim_docker
find "${RUN_ROOT}/runs" -name record.json | wc -l
```

If the server restarts, rerun the same `docker exec` command. Resume validation
prevents completed cases from being silently reused with different inputs.

## 8. Result layout

```text
<run-root>/
  manifest.json
  pilot.json
  summary.csv
  summary.json
  configs/
  workloads/
    *.jsonl
    *.summary.json
  runs/<case>/
    cluster_config.json
    workload.summary.json
    command.txt
    run.log
    record.json
    requests.csv
    requests_kv_offload.csv
    host_timing.json
```

The manifest records the root and recursive submodule revisions. Each record
contains input and output hashes, exit status, timeout state, wall time,
simulated time, and transport digests.

Rebuild summaries at any time without rerunning simulations:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root "${RUN_ROOT}" \
  summarize
```

Archive the complete run root:

```bash
tar -C "$(dirname "${RUN_ROOT}")" \
  -czf "cpu_kv_${RUN_ID}.tar.gz" "$(basename "${RUN_ROOT}")"
sha256sum "cpu_kv_${RUN_ID}.tar.gz" \
  > "cpu_kv_${RUN_ID}.tar.gz.sha256"
```
