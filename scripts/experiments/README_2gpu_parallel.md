# 2 GPU main_results launcher

These scripts run the formal `main_results` matrix on a 2 * RTX 4090 24GB machine without putting all rows into one unbounded local queue.

Do not use bare `N_JOBS` for this run. It controls only a single process pool and does not bind work to specific GPUs, so multiple children can pile onto one GPU while the other is underused. It also makes RAM and DataLoader I/O pressure harder to reason about.

Each child sets `CUDA_VISIBLE_DEVICES` before calling `05_run_one.sh`. That keeps every run isolated to one physical GPU while still reusing the existing single-index runner and `run_one.py` behavior. Row-level DataLoader settings are overridden with `ALLOW_ROW_OVERRIDE=1`.

Recommended settings:

- First formal test: `JOBS_PER_GPU=2 NUM_WORKERS_PER_RUN=2`
- If GPU utilization is still low and RAM/I/O are stable: `JOBS_PER_GPU=3 NUM_WORKERS_PER_RUN=1`
- For full trajectory or time-varying tasks that use more memory: `JOBS_PER_GPU=1`

One-command smoke test:

```bash
DATA_ROOT=/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
OUT_ROOT=outputs/main_results_20260622_1500 \
bash scripts/experiments/10_run_first8_2gpu_smoke.sh
```

One-command formal run:

```bash
DATA_ROOT=/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
OUT_ROOT=outputs/main_results_20260622_1500 \
bash scripts/experiments/09_run_main_results_2gpu.sh
```

Check progress:

```bash
bash scripts/experiments/11_progress_main_results.sh "$OUT_ROOT"
```

Clean stale marker files after confirming no intended run is active:

```bash
CONFIRM=1 bash scripts/experiments/12_clean_stale_run_locks.sh "$OUT_ROOT"
```

Monitor the machine:

```bash
watch -n 2 nvidia-smi
watch -n 2 free -h
```

