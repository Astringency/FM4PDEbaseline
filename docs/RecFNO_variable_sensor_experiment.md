# RecFNO training with a sensor count sampled per batch

This opt-in experiment adds new files only. Existing model, data-adapter,
training-loop, and experiment configurations are unchanged. The entry point
temporarily installs its loader factory inside its own Python process and
restores it on exit.

## Comparison

- RecFNO only; Poisson, Helmholtz, Darcy and NS endpoints.
- Fixed control: 500 spatial locations in every training batch.
- Mixed: uniformly sample N from 50, 100, 250, 500, 1000 for each batch.
- All samples in a batch share N, with independent per-sample locations and
  a_only/u_only/both conditions. Both fields share the same locations.
- Same training seed, initialization, shuffled sample order, conditional
  sampling, normalization, model, loss, optimizer and learning-rate schedule.
- 50,000 requested training records: the original splitter reserves 5,000 for
  validation, leaving 45,000 for optimization. Batch size 16; maximum 200
  epochs with the existing early-stopping rule.
- Validation always uses 500 locations and all three conditions. This keeps
  checkpoint selection aligned with the primary 500-point comparison.
- Evaluate the first 1,000 distinct samples of each original, ID, Smooth and
  Rough test file, for all three conditions at all five N values. Match sample
  IDs, location hashes and input-file checksums across the two models.
- Report per-sample physical-field relative L2 for a, u, and the concatenated
  fields. For forward, the principal target is u; for inverse, it is a. Retain
  both individual errors for joint because raw joint L2 can be dominated by the
  field with the larger physical scale.

The count draw uses a separate stable hash of count seed, epoch and batch index,
so it consumes no model/shuffle random numbers. A batch sampler sends tuple
keys `(sample_index, count, epoch, step)` to workers, preserving the requested
epoch even across prefetch queues. DataLoader iteration audits include count
histograms, sample-order hashes and condition frequencies. Iteration zero is
normalization; later iterations are training epochs.

## Entry points

- `scripts/run_recfno_variable_sensors.py`: the original runner arguments plus
  `--train-sensor-counts 50,100,250,500,1000` and `--count-seed 1`. Use `500`
  alone for the matched fixed control.
- `scripts/recfno_sensor_count_study.py`: one sequential PDE queue for one
  regime, recording commands, logs and exit codes. `--pilot` runs a real-data
  memory smoke test; its results are excluded from the formal comparison.
- `scripts/eval_recfno_sensor_counts.py`: load a checkpoint verified against
  its training summary and sweep all evaluation conditions without retraining.
  Hide truth from model inputs, save errors for every sample, and retain three
  example predictions in each cell. Completed cells are checksum-validated
  before reuse. An interrupted incomplete cell is recomputed.
- `scripts/compare_recfno_sensor_counts.py STUDY_ROOT`: require all 240 paired
  cells, check training configuration/sample order and evaluation pairing, then
  save comparison JSON/CSV. `--allow-partial` explicitly marks incomplete output.

Run full data verification with the new experiment YAML before formal queues:

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/recfno_variable_sensors.yaml \
  --data-root "$DATA_ROOT" --output-dir "$STUDY_ROOT/data_protocol" --full
```

On server193 each regime has its own tmux session and GPU. New results belong
under the absolute shared FM4PDEbaseline output root, in a dated task directory.
The code is synchronized through Git into an independent checkout. Neither
historical outputs nor the original server checkout are changed. Training
resumes at completed PDE boundaries; an interrupted partial training run must
be inspected explicitly before starting another attempt.

Cooperating queues can use `--queue-tag parallel` to keep distinct status
records. Training and evaluation entry points lock their output directories;
an identical training request waits for the owner and then validates and reuses
the completed checkpoint. This lets different PDEs share a GPU when measured
resource headroom permits, without training the same output concurrently.
The original queue visits Poisson/Helmholtz/Darcy/NS; a cooperating queue can
visit Helmholtz/NS/Darcy to advance the remaining work. This affects scheduling,
not the batch size, seed, epoch budget, or sensor-count distribution.

Uncertainty intervals use paired bootstrap resampling of test samples (2,000
replicates). One training seed does not quantify variation across training
initializations. Mixed training has a mean requested budget of 380 locations;
this study tests robustness across densities, not equal average information
budget. It does not imply that a model trained at fixed 500 cannot accept other
counts: both checkpoints are evaluated across the same sweep.
