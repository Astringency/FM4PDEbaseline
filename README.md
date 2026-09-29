# FM4PDE baseline experiment: RecFNO, Senseiver, VoronoiCNN, PINN-Sparse, PC-BNN, B-PINNs, 4D-Var, VIVID

Sparse reconstruction and physics-based baselines for the revised FM4PDE paper.

| Method | Approach and source |
| --- | --- |
| RecFNO | Voronoi embedding and a Fourier operator; [official implementation](https://github.com/zhaoxiaoyu1995/recfno). |
| Senseiver | Attention-based field reconstruction; [official implementation](https://github.com/OrchardLANL/Senseiver). |
| VoronoiCNN | Convolutions on a Voronoi representation; [official implementation](https://github.com/kfukami/Voronoi-CNN). |
| PINN-Sparse | Neural fields optimized against observations and PDE residuals; adapted from [PINNs](https://github.com/maziarraissi/PINNs) and [DeepXDE](https://github.com/lululxvi/deepxde). |
| PC-BNN | Physics-constrained Bayesian particles; [official implementation](https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning). |
| B-PINNs | HMC over neural-field parameters; local adaptation of the [public PyTorch reference](https://github.com/obok13/B-PINNs), which is a third-party implementation. |
| 4D-Var | Variational trajectory assimilation; local Burgers adaptation with [reference assimilation code](https://github.com/googleinterns/invobs-data-assimilation). |
| VIVID | Learned inverse observations and variational refinement; local Burgers adaptation of [VIVID](https://github.com/DL-WG/VIVID). |

## Structure

- `baselines/`: adapters, common data/sensor/physics code, metrics, and provenance.
- `configs/`: manuscript experiment matrix and five-shard data filenames.
- `offical/`: retained upstream sources and notices (historical directory spelling).
- `scripts/`: training, distribution evaluation, aggregation, and B-PINNs HMC.

## Setup and examples

Use `environment.yml`; keep datasets outside Git.

```bash
export DATA_ROOT=/path/to/PDEdata
export OUT_ROOT=/path/to/baseline-results
DRY_RUN=1 bash scripts/training/run.sh
BASELINES=recfno,senseiver,voronoicnn GPUS=0,1 JOBS_PER_GPU=1 bash scripts/training/run.sh
for split in id smooth rough; do
  bash scripts/sampling/main/run.sh --output-root "$OUT_ROOT" --distribution "$split"
done
```

[configs/experiments/main_results.yaml](configs/experiments/main_results.yaml)
uses 45,000 training / 5,000 validation samples and 100 evaluation inputs per
setting. Physics-based comparisons use Smooth; Burgers includes random points
and five complete physical-time levels, evaluated over the full trajectory.

B-PINNs uses the hashed physical inputs and training normalization exported by
`FunDPS_DDIS_ECI_OFM/adapters/prepare_shared_prior_assets.py`:

```bash
python scripts/run_bpinns.py --assets /path/to/shared_prior_assets \
  --output "$OUT_ROOT/bpinns" --pdes poisson helmholtz darcy --device cuda:0
```
