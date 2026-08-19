# experiment_plan_v2 baseline 审计

> 结论：当前 87 行中 **0 行可直接发表**。本报告描述的是已确认的客观实现/协议问题，不对修改者的主观意图作判断。

## 结果处置

- 硬性排除（去重）：74/87 行。
- 仅条件性排除：13/87 行；仍须按修复后的协议重新运行。
  其中 12 行是固定传感器布局的 sparse-inverse，另 1 行是非传感器 full-forward 的单 seed 结果。
- 可发布：0/87 行。
- 原始工作簿只读校验：`4a103fcbd253cf8c53a170ecf1689e6df1a7497ac46533fb02e7ae1ba1e1a5b2`；87 行、38 列、公式=无。

五类硬性原因允许在同一行上叠加，不能把类别计数直接相加：

| 原因标签 | 行数 | 证据与影响 |
|---|---:|---|
| `stale_eval_only_artifact` | 33 | train_time 恰为 0：该行从旧 checkpoint 重新评估，并非由当前矩阵运行完成训练 |
| `sparse_forward_protocol_invalid` | 21 | 全部 sparse-forward 行使用了无效协议；细分标签记录已独立确认的归一化和隐藏目标机制 |
| `burger_inverse_target_leakage` | 4 | 审计版本的 Burger inverse 输入暴露了 t=0 目标；部分适配器训练时广播、评估时裁剪 |
| `sparse_solution_physics_hidden_truth` | 10 | PINN/PDEOpt sparse-solution 行读取完整隐藏物理场，违反 sensor-only 比较协议 |
| `legacy_summary_schema` | 14 | summary 早于强制 provenance schema，缺少 commit_hash |

legacy schema 的 14 行中有 8 行同时命中另一项硬性问题；五类原因的去重并集为 74 行。

sparse-forward 的 21 行进一步分解如下（子类可以重叠）：

| 细分原因标签 | 行数 | 证据与影响 |
|---|---:|---|
| `sparse_forward_wrong_input_normalization` | 15 | 15 个 amortized sparse-forward 行用 target/solution 而不是 input/source 统计量归一化输入观测 |
| `sparse_forward_physics_hidden_target` | 6 | 6 个 per-instance sparse-forward 行生成时，target fields 尚未与物理方法 metadata 隔离 |
| `sparse_forward_voronoicnn_hidden_target` | 2 | 2 个 VoronoiCNN sparse-forward 行由 commit 3340de0 生成，其 Voronoi 构造可读取稀疏目标解 |

## 为什么稀疏结果会过好

审计版本中的 `random` 并非逐样本随机：每个 split 都用相同 seed 构造一次 mask，训练、验证和测试因而共享同一传感器布局。这测量的是同布局插值，不是未知布局泛化。以下证据使用同一 checkpoint，只改变测试 mask seed：

| 方法 | test mask seed | relative L2 | 相对 seed=1 恶化倍数 | mask id |
|---|---:|---:|---:|---|
| recfno | 1 | 0.015016 | 1.00× | `907b3652363586fe` |
| recfno | 2 | 0.288793 | 19.23× | `72a859871eaf37e6` |
| recfno | 3 | 0.275686 | 18.36× | `5b15d323e5657662` |
| senseiver | 1 | 0.011557 | 1.00× | `907b3652363586fe` |
| senseiver | 2 | 0.065637 | 5.68× | `72a859871eaf37e6` |
| senseiver | 3 | 0.067196 | 5.81× | `5b15d323e5657662` |

此外，15 个 amortized sparse-forward 行把 input observation/Voronoi 元数据按 target solution 统计量归一化；6 个 physics 行存在隐藏 target 读取，15 个 amortized 行中的 2 个 VoronoiCNN 行也存在该问题。这些机制的去重并集覆盖全部 21 个 sparse-forward 行。Burger inverse 则把目标初态放进了输入，影响 4 行。

PINN/PDEOpt 的 `fit()` 是空操作，微秒级 `train_time` 不是实际优化成本；其优化发生在预测阶段，应使用 `inference_optimization_time_total/per_sample` 比较。

FNO 的极小 MSE 也不应单独解读为近乎完美：部分解场量纲很小，必须以无量纲 relative L2 和清晰的任务字段共同报告。当前指标公式本身未发现直接读取测试标签的问题。

样本内容方面，当前挂载的审计证据不包含原始 PDE 数据文件，因此尚未完成 train/val/test 的全量内容哈希与交集核验，不能排除物理文件重复或内容泄漏。修复后的正式运行必须先通过 `scripts/verify_data_protocol.py --full`，并把该报告的 SHA-256 绑定进矩阵、checkpoint 和 summary。

## 与 official 实现的一致性

`official` 在此项目中常表示导入官方网络组件，并不表示复现官方数据、训练器、损失、超参数和评估的完整实验流程。分类如下：
vendored source metadata 中的 upstream revision 和 local modifications 仍为 unknown；在固定 tree/commit 并完成差异核对前，不能声称 exact official reproduction。

| 方法 | 审计分类 | 完整官方复现 | 说明 |
|---|---|---:|---|
| fno | `official_component_reuse` | 否 | 复用了 NeuralOperator FNO 组件，但训练器、优化器、归一化、早停和评估均为本地协议。 |
| deeponet | `official_component_reuse` | 否 | 复用了 DeepXDE Cartesian-product 网络组件，但未使用官方 Model.compile/Model.train 实验协议。 |
| ifno | `official_training_aligned_adaptation` | 否 | 保留可逆耦合、官方 VAE 拓扑、posterior-mean 逆推理和三阶段训练；数据与任务由 FM4PDE adapter 提供。 |
| recfno | `adapted_reimplementation_major_input_divergence` | 否 | 审计适配器使用 zero-filled field + mask + coordinates，width/modes/loss/训练设置均不同于 vendored recipe。 |
| senseiver | `adapted_reimplementation_major_encoding_divergence` | 否 | 审计适配器使用 raw coordinates 和不同 latent/layer 设置，没有采用 vendored Fourier positional encoding recipe。 |
| voronoicnn | `architecture_reimplementation` | 否 | 复现了核心层类型，但 width、batch size、epochs 和统一任务协议不同于 vendored 实验。 |
| pinn_sparse | `deepxde_native_task_adapter` | 否 | 使用 DeepXDE PDE、PointSetBC、自动微分、FNN 与 Adam→L-BFGS；联合场输出适配 FM4PDE 稀疏任务。 |
| pde_opt | `project_custom_baseline` | 否 | 项目定义的 per-instance PDE optimization baseline。 |
| pc_bnn | `official_training_aligned_adaptation` | 否 | 保留三层 Swish、层次先验、Gamma 初始化、SVGD 修正梯度和逐粒子 Adam；PDE likelihood 与输出场适配静态 FM4PDE 任务。 |
| var4d | `project_canonical_math_baseline` | 否 | 项目本地 4D-Var/两层动力学实现；只有显式 time-varying trajectory 协议可进入统一比较，v2 表未运行。 |
| vivid | `official_component_or_aligned_adapter` | 否 | 需要显式 inverse-observation operator 与完整时变轨迹；统一训练/评估仍为本地适配，v2 表未运行。 |

## 本次实现已修复什么

当前代码已实现上述协议修复：`random_per_sample` 为 batch-aware mask，训练按 epoch 换布局；sparse-forward 改用 input-side normalization；loss/metric 禁止广播与裁剪；Burger full inverse 改为 `u(T)->u(0)` 并禁用 sparse-inverse；模型评估视图不再包含 target/full truth。

RecFNO 当前使用 Voronoi-filled field + mask + coordinates，Senseiver 当前恢复 Fourier positional encoding；FNO/DeepONet/RecFNO/Senseiver 均明确标成 official component + unified adapted training，iFNO/PINN-Sparse/PC-BNN 标成 official-training-aligned task adaptation。旧 checkpoint 因输入/架构、训练协议或 provenance 不兼容必须隔离重训，这些修复不会追溯性地使历史结果有效。

runner/checkpoint/export pipeline 现记录并校验 schema、execution mode、run/config fingerprint、protocol versions、checkpoint SHA-256 与 comparison track；legacy/mismatch summary 会进入 quarantine。

## 修复与重新验收

1. 使用真正的 `random_per_sample`：训练按样本/epoch 采样，验证与测试按 split 和样本 ID 确定性生成不同布局；保留 `fixed` 作为单独控制组。
2. sparse-forward 只用 input-side statistics；训练和评估严格要求 prediction/target shape 完全相同，禁止广播与静默裁剪。
3. Burger full inverse 定义为末态到初态，并移除 Burger sparse-inverse；sensor-only 主表不允许 PINN/PDEOpt 访问隐藏完整场。
4. 将方法标为 component reuse、architecture reimplementation 或 adapted protocol；只有固定 upstream revision 且端到端原生协议一致时才标 exact official。
5. 新运行必须保存并校验 commit、完整 config 指纹、checkpoint hash、数据 manifest、split 和 sensor policy；缺失 provenance 或 eval-only 复用不得进入主表。
6. 重新运行至少多个训练 seed，并分别报告 fixed-layout、unseen-layout 和 random-per-sample 面板。历史 87 行只保留为审计证据，不覆盖、不回填。

## 逐行证据

机器可读逐行原因和 provenance 位于 `outputs/experiment_plan_v2_audit/rows.json`、`rows.csv` 与 `rows.xlsx`；汇总位于同目录的 `summary.json`。`hard_reason_tags` 支持一行多个原因，`publishable` 对本历史 cohort 始终为 `false`。
