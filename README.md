# FM4PDE baselines

Baseline experiments used in the FM4PDE paper:

| Experiment | Methods | PDEs |
| --- | --- | --- |
| Full-field forward / inverse | FNO, DeepONet, iFNO (inverse: iFNO) | Poisson, Helmholtz, Darcy, Navier–Stokes |
| Sparse forward, inverse, and joint reconstruction | RecFNO, Senseiver, VoronoiCNN | Poisson, Helmholtz, Darcy, Navier–Stokes |
| Physics-based sparse forward / inverse | PINN-Sparse, PDE-Opt, PC-BNN | Poisson, Helmholtz, Darcy |
| Trajectory reconstruction: random / structured observations | RecFNO, Senseiver, VoronoiCNN, 4D-Var, VIVID | Burgers |
| Separate-task vs. shared-model reconstruction | RecFNO, Senseiver, VoronoiCNN | Poisson |

Spatial-frequency and timing comparisons reuse these models through the analysis
scripts in the companion FM4PDE repository. Upstream source code and licenses are
retained in `offical/`.

## Setup

Use the `fm4pdebaseline` environment specified in `environment.yml`.
Keep datasets outside the repository; filenames are listed in
`configs/data_files/formal_128.yaml`.

```bash
export DATA_ROOT=/path/to/PDEdata
export PYTHON_BIN=python
export OUT_ROOT=outputs/main_results  # use an absolute result path on a server
```

## Main experiments

```bash
DRY_RUN=1 bash scripts/training/run.sh
GPUS=0,1 JOBS_PER_GPU=1 bash scripts/training/run.sh
for distribution in id smooth rough; do
  bash scripts/sampling/main/run.sh --output-root "$OUT_ROOT" --distribution "$distribution"
done
```

`configs/experiments/main_results.yaml` defines the main comparisons. The workflow
validates data, trains models, evaluates, and collects results. Training uses
45,000 samples plus 5,000 validation samples; each evaluation uses 1,000 inputs.
Set `BASELINES` to a comma-separated method subset. The paper's physics-based
sparse comparisons use Smooth data; the other comparisons use all three distributions.

## Poisson multi-task ablation

```bash
ABLATION_ROOT=outputs/ablations/sparse_solution_multicondition/poisson
OUT_ROOT="$ABLATION_ROOT" GPUS=0,1 bash scripts/sampling/ablations/run.sh
for distribution in id smooth rough; do
  bash scripts/sampling/main/run.sh --output-root "$ABLATION_ROOT" \
    --matrix "$ABLATION_ROOT/matrices/sparse_solution_multicondition_ablation.jsonl" \
    --train-root "$ABLATION_ROOT/runs/sparse_solution_multicondition_ablation" \
    --task-groups "sparse_solution_multicondition_eval_a_only sparse_solution_multicondition_eval_u_only sparse_solution_multicondition_eval_both" \
    --distribution "$distribution"
done
```

`configs/experiments/sparse_solution_multicondition_ablation.yaml` trains one
model per method on an equal mixture of input-only, solution-only, and joint
observations, then evaluates all three modes using 500 shared sensor locations.
The separate-task models come from the main experiments.
