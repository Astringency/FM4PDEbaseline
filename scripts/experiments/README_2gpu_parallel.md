# 2 GPU main_results launcher

These scripts run the formal `main_results` matrix on a 2 * RTX 4090 24GB machine without putting all rows into one unbounded local queue.

Do not use bare `N_JOBS` for this run. It controls only a single process pool and does not bind work to specific GPUs, so multiple children can pile onto one GPU while the other is underused. It also makes RAM and DataLoader I/O pressure harder to reason about.

Each child sets `CUDA_VISIBLE_DEVICES` before calling `05_run_one.sh`. That keeps every run isolated to one physical GPU while still reusing the existing single-index runner and `run_one.py` behavior. Fingerprinted training and DataLoader settings are read unchanged from the matrix row; the launcher clears `ALLOW_ROW_OVERRIDE` and does not inject worker settings.

Recommended settings:

- First formal test: `JOBS_PER_GPU=2`
- If GPU utilization is still low and RAM/I/O are stable: `JOBS_PER_GPU=3`
- For full trajectory or time-varying tasks that use more memory: `JOBS_PER_GPU=1`
- Checkpoints: `SAVE_CHECKPOINT=amortized` is the default and saves reusable neural baselines (`fno`, `deeponet`, `ifno`, `recfno`, `senseiver`, `voronoicnn`). Formal amortized runs must retain their checkpoint and reject `SAVE_CHECKPOINT=0`; that override is limited to non-formal debug runs. Use `SAVE_CHECKPOINT=all` to save every baseline, but per-instance optimization methods are less reusable and use more disk.

To change `num_workers`, pinning, persistence, or prefetching, edit the experiment
config, run full data verification again, and rebuild the matrix. Runtime
overrides would change the effective command without changing the stored
fingerprint and are therefore rejected.

Verify the exact main-results config before its matrix is built:

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/main_results.yaml \
  --data-root /home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
  --full
```

One-command smoke test:

```bash
DATA_ROOT=/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
DATA_MANIFEST=outputs/data_protocol/main_results/full/data_protocol_report.json \
OUT_ROOT=outputs/main_results_20260622_1500 \
bash scripts/experiments/10_run_first8_2gpu_smoke.sh
```

One-command formal run:

```bash
DATA_ROOT=/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
DATA_MANIFEST=outputs/data_protocol/main_results/full/data_protocol_report.json \
OUT_ROOT=outputs/main_results_20260622_1500 \
SAVE_CHECKPOINT=amortized \
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
