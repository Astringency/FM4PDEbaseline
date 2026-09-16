# FM4PDE Baseline Experiments

Baseline implementations and evaluation workflows for **Guided Flow Matching
for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory**.

| Comparison | Methods | PDEs |
| --- | --- | --- |
| Full-field forward / inverse | FNO, DeepONet, iFNO (inverse: iFNO) | Poisson, Helmholtz, Darcy, Navier–Stokes |
| Sparse forward, inverse, and joint reconstruction | RecFNO, Senseiver, VoronoiCNN | Poisson, Helmholtz, Darcy, Navier–Stokes |
| Physics-based sparse reconstruction | PINN-Sparse, PDE-Opt, PC-BNN | Poisson, Helmholtz, Darcy |
| Space–time trajectory reconstruction | RecFNO, Senseiver, VoronoiCNN, 4D-Var, VIVID | Burgers |

## Training

Run commands from the repository root in Bash (Linux or WSL), using
[environment.yml](environment.yml). External dataset filenames are listed in
[configs/data_files/formal_128.yaml](configs/data_files/formal_128.yaml).

```bash
conda env create -f environment.yml
conda activate fm4pdebaseline
export DATA_ROOT=/path/to/PDEdata
export OUT_ROOT=/path/to/results/main_results
export PYTHON_BIN=python
```

The Python entry is `main()` in [baselines/run.py](baselines/run.py).
It calls the selected method's `fit()` in [baselines/methods](baselines/methods)
(for example, `RecFNOBaseline.fit()` in
[recfno.py](baselines/methods/recfno.py)).
The formal experiment matrix supplies the task, model, data split, and sensor
settings from [configs/experiments/main_results.yaml](configs/experiments/main_results.yaml).

To prepare a formal matrix and launch one experiment directly with Python:

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/main_results.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/data_protocol/main_results/full" --full

python scripts/build_experiment_matrix.py \
  --config configs/experiments/main_results.yaml \
  --matrix-name main_results --output-root "$OUT_ROOT" \
  --data-manifest "$OUT_ROOT/data_protocol/main_results/full/data_protocol_report.json"

# Execute row 0; inspect the JSONL matrix to select a different row.
DEVICE=cuda:0 python scripts/experiments/run_one.py \
  "$OUT_ROOT/matrices/main_results.jsonl" 0
```

The Bash workflow performs data verification, matrix construction, training,
evaluation, and result collection, and supports resuming completed work:

```bash
DRY_RUN=1 bash scripts/training/run.sh
GPUS=0,1 JOBS_PER_GPU=1 bash scripts/training/run.sh

# Run only selected methods in a separate result directory.
BASELINES=recfno,senseiver,voronoicnn OUT_ROOT=/path/to/results/sparse_baselines \
  GPUS=0,1 JOBS_PER_GPU=1 bash scripts/training/run.sh
```

The main protocol uses 45,000 training inputs, 5,000 validation inputs, and
1,000 evaluation inputs per cell. Physics-based methods fit each test instance;
their execution is included in the same experiment workflow.

## Main Sampling

Evaluation is orchestrated by [scripts/run_eval.py](scripts/run_eval.py), which
uses the matrix and saved model checkpoints to invoke [baselines/run.py](baselines/run.py).
For supervised models, this is prediction from trained weights; physics-based
methods perform their configured optimization.

```bash
# Evaluate trained Poisson RecFNO models on Smooth inputs.
python scripts/run_eval.py --output-root "$OUT_ROOT" \
  --data-root "$DATA_ROOT" --pdes poisson --baselines recfno \
  --distribution smooth --device cuda:0

# Evaluate the main matrix on all three distributions.
for distribution in id smooth rough; do
  bash scripts/sampling/main/run.sh \
    --output-root "$OUT_ROOT" --distribution "$distribution"
done
```

Training must have produced the source checkpoints first. Use `--matrix` and
`--train-root` for a nondefault matrix, `--task-groups` to select comparisons,
and `--dry-run` to inspect evaluation commands. The paper reports the
physics-based sparse comparisons on Smooth; the other main comparisons cover
ID, Smooth, and Rough.

## Ablations

[scripts/sampling/ablations/run.sh](scripts/sampling/ablations/run.sh) uses
[configs/experiments/sparse_solution_multicondition_ablation.yaml](configs/experiments/sparse_solution_multicondition_ablation.yaml)
to train one shared model per method on an equal mixture of input-only,
solution-only, and joint observations, then evaluate each observation mode.
The paper's default is Poisson with 500 shared sensor locations.

```bash
ABLATION_ROOT=/path/to/results/poisson_multicondition

OUT_ROOT="$ABLATION_ROOT" DRY_RUN=1 \
  bash scripts/sampling/ablations/run.sh
OUT_ROOT="$ABLATION_ROOT" GPUS=0,1 \
  bash scripts/sampling/ablations/run.sh

for distribution in id smooth rough; do
  bash scripts/sampling/main/run.sh \
    --output-root "$ABLATION_ROOT" \
    --matrix "$ABLATION_ROOT/matrices/sparse_solution_multicondition_ablation.jsonl" \
    --train-root "$ABLATION_ROOT/runs/sparse_solution_multicondition_ablation" \
    --task-groups "sparse_solution_multicondition_eval_a_only sparse_solution_multicondition_eval_u_only sparse_solution_multicondition_eval_both" \
    --distribution "$distribution"
done
```

The separate-task reference models come from the main training workflow.

## Baseline and other info

Related repositories: [FM4PDE](https://github.com/Astringency/FM4PDEdebug.git),
[DiffusionPDE comparisons](https://github.com/Astringency/DiffusionPDE.git), and
[CoCoGen comparisons](https://github.com/Astringency/CoCoGen.git).
The FM4PDE repository also provides frequency and timing analyses of saved
baseline predictions.

Our implementations and adapters build on the following official codebases.
We adapt their data interfaces, observations, tasks, and evaluation to the
FM4PDE protocol. Some methods use local architecture reproductions; backend
provenance is recorded in the results and in
[baselines/methods/official.py](baselines/methods/official.py).
Bundled upstream sources and their notices are retained in [offical/](offical/).

| Method / component | Official code |
| --- | --- |
| FNO / NeuralOperator | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) |
| DeepONet / DeepXDE | [lululxvi/deepxde](https://github.com/lululxvi/deepxde) |
| iFNO | [BayesianAIGroup/iFNO](https://github.com/BayesianAIGroup/iFNO) |
| RecFNO | [zhaoxiaoyu1995/RecFNO](https://github.com/zhaoxiaoyu1995/RecFNO) |
| Senseiver | [OrchardLANL/Senseiver](https://github.com/OrchardLANL/Senseiver) |
| VoronoiCNN | [kfukami/Voronoi-CNN](https://github.com/kfukami/Voronoi-CNN) |
| PC-BNN | [Jianxun-Wang/Physics-constrained-Bayesian-deep-learning](https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning) |
| VIVID | [DL-WG/VIVID](https://github.com/DL-WG/VIVID) |
| Learned inverse-observation operator / data assimilation | [googleinterns/invobs-data-assimilation](https://github.com/googleinterns/invobs-data-assimilation) |
