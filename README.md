# FM4PDE Baseline

这是一个面向 FM4PDE 论文对比实验的 external baseline 项目。项目目标不是把所有方法都改写成本地统一模型，而是提供一个 reviewer-facing 的 baseline framework：优先调用原始论文/官方仓库实现，明确区分 native、official adapter、adapted 和 unsupported，避免把本地简化实现伪装成官方 baseline。

## 项目定位

本仓库用于运行和审计 FM4PDE 外部 baseline 对比实验，覆盖：

- full-grid supervised operator learning
- sparse sensor field reconstruction
- static sparse inverse problems
- time-varying data assimilation

当前重点是外部 baseline 对比，不包含 FM4PDE 自身内部消融和 DiffusionPDE 对比。

## 目录结构

```text
baselines/                  baseline 框架核心代码
  capabilities.py           baseline/task/PDE/sensor capability registry
  experiment_matrix.py      capability-aware experiment matrix 和 skip 逻辑
  run.py                    单次 baseline runner
  aggregate_results.py      主表、补充表和 skip 结果聚合
  methods/                  各 baseline wrapper
  common/                   数据适配、传感器、物理残差和指标
  configs/                  paper/debug 配置
  README.md                 baseline 框架详细说明
  BASELINE_SOURCES.md       论文、官方代码和启用范围说明
  NATIVE_CAPABILITY_MATRIX.md

configs/experiments/        大规模实验矩阵配置
scripts/baselines/          paper/smoke baseline 运行脚本
scripts/experiments/        matrix 构建、单任务运行、聚合脚本
tests/                      pytest 测试
offical/                    vendored 官方代码快照（目录名沿用现有拼写）
outputs/                    实验输出，默认不应作为论文源码依赖
```

## Baseline 能力状态

每个 baseline/PDE/task/sensor_mode 组合都会通过 `baselines.capabilities.resolve_capability(...)` 判定。

| 状态 | 含义 | 是否进入主表 |
| --- | --- | --- |
| `native` | 属于原论文/标准方法能力范围 | 满足实现要求时进入 |
| `official_adapter` | 使用官方组件，并由本仓库做数据适配 | 满足实现要求时进入 |
| `adapted` | 本地改写、目标变换、style 实现或 surrogate | 只进入 supplement |
| `unsupported` | 不属于标准能力或缺少必要目标/轨迹/残差 | paper mode 跳过 |

paper mode 默认使用：

```yaml
method:
  implementation_mode: official_or_skip
```

也就是说，要求官方实现的 baseline 如果无法导入官方代码/组件，会失败或跳过，不会静默 fallback 到 local compact implementation。

## 数据要求

默认数据根目录：

```bash
/home/tat512/C01Python/PDEdata
```

可通过环境变量覆盖：

```bash
DATA_ROOT=/path/to/PDEdata
```

runner 会记录数据接口元信息，包括 raw input shape、native/official input shape、target shape、观测字段、预测字段、scalar PDE 参数是否进入输入、是否加载 full trajectory、sensor mode 等。

## 正式运行

运行 native/full-grid operator baseline：

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_native_full_operator.sh
```

运行 native sparse reconstruction baseline：

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_native_sparse_reconstruction.sh
```

运行 static sparse inverse baseline：

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_static_sparse_inverse.sh
```

运行 time-varying data assimilation baseline：

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_time_varying_da.sh
```

运行完整 native paper matrix：

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_all_native.sh
```

兼容入口：

```bash
bash scripts/baselines/run_paper_all.sh
```

该脚本会委托到 native paper matrix。旧的 per-task paper 脚本仍保留用于 supplement/debug，但 adapted/surrogate/local 结果不会混入主表。

## 聚合结果

```bash
OUT=outputs/baselines/paper \
  bash scripts/baselines/aggregate_paper_results.sh
```

聚合输出包括：

- `summary_main.csv/json`：只包含 `paper_table_eligible=true`
- `summary_supplement.csv/json`：adapted、surrogate、local 或 fallback 结果
- `skipped_combinations.csv/json`：paper mode 跳过组合
- `baseline_capability_matrix.csv/json`：完整能力矩阵
- `latex_table.csv/tex`：只从 main summary 生成

## 能力矩阵

导出完整 baseline capability matrix：

```bash
python -m baselines.experiment_matrix \
  --dump-matrix \
  --output outputs/baselines/capability_matrix
```

详细说明见：

- `baselines/README.md`
- `baselines/BASELINE_SOURCES.md`
- `baselines/NATIVE_CAPABILITY_MATRIX.md`

## Smoke 和测试

快速 smoke：

```bash
bash scripts/baselines/smoke_all.sh
```

推荐测试命令：

```bash
python -m py_compile \
  baselines/run.py baselines/aggregate_results.py baselines/experiment_matrix.py \
  baselines/capabilities.py baselines/common/*.py baselines/methods/*.py tests/test_*.py

bash -n scripts/baselines/*.sh

python -m pytest -q
```

如果系统里存在用户级 `pytest` shim，优先使用当前 conda 环境：

```bash
PATH=/home/tat512/.conda/envs/FM4PDEbaseline/bin:$PATH pytest -q
```

## 关键原则

- 不把 local compact implementation 写成 official baseline。
- 不把 adapted/surrogate 结果混入 paper main table。
- 不支持的任务必须 skip，并记录 `unsupported_reason`。
- FNO/DeepONet 的 sparse 或 inverse adaptation 只作为补充结果。
- iFNO sparse tasks 不作为 native baseline。
- 4D-Var/VIVID 只有 full trajectory 或 multi-time observations 才能进入 time-varying DA 主表。
- PDE-Opt 是 canonical mathematical baseline，不声称官方代码来源。
