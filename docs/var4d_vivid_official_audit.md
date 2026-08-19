# Var4D / VIVID 官方实现对照审计

审计对象为仓库内的 `offical/VIVID`、`offical/invobs-data-assimilation` 与 Burgers 适配实现。vendored 源码没有可核验的上游 commit/tag，因此本审计只能对照当前仓库快照，不能证明与上游某个版本逐字一致。

## 结论

当前 Var4D 和 VIVID 均为 **Burgers 专用 adapted baseline**，不满足 `docs/baseline_exp.md` 第 7 条所要求的 official-core 保留条件，不得标记为 `canonical_math`、`official`、`official_aligned` 或 official-native。它们仍可进入统一结果汇总，但 capability 和结果 provenance 必须保持 `adapted`，且 `paper_table_eligible=false`、`official_native_eligible=false`。

## Var4D

仓库没有 vendored 的官方 Var4D 程序，只有 4D-Var 数学基线概念可供对照。标准强约束 4D-Var 优化初始状态或控制变量，通过动力学模型传播完整轨迹，并在多个时刻比较观测。

当前实现的差异：

| 项目 | 强约束 4D-Var | 当前 Burgers 适配 |
| --- | --- | --- |
| 优化变量 | 初始状态/控制变量 | 完整 $T\times X$ 轨迹的每个值 |
| 动力学约束 | 由数值模型传播，轨迹严格满足模型 | Burgers PDE 残差作为软惩罚 |
| 优化器 | 实现相关，常见 L-BFGS/增量法 | Adam |
| 初始背景 | 背景初值及其协方差 | 稀疏观测的 Voronoi 全轨迹填充 |
| 当前标签 | 可称 canonical 需实现标准目标 | `var4d_style_burgers_weak_constraint_adapted` |

因此当前算法更准确的名称是“weak-constraint Var4D-style trajectory optimization”，而不是 canonical 4D-Var。

## VIVID / invobs

### vendored VIVID 快照

`offical/VIVID/VCNN_training.py` 使用：

- Voronoi 填充后的单通道场作为 CNN 输入；
- 6 个 `48` 通道、`8×8`、ReLU、same-padding 卷积层，再接一个单通道 `8×8` 输出卷积；
- MSE、Adam、学习率 `1e-4`；
- 20 epochs、batch size 64、5% validation split；
- `offical/VIVID/VIVID.py` 中使用 ADAO 3DVAR 与 L-BFGS-B，最大 1000 steps，cost decrement tolerance 为 `1e-6`。

### vendored invobs 快照

`offical/invobs-data-assimilation` 是数据集专用的 JAX/Flax 流程：逆观测网络使用周期空间卷积、BatchNorm 和 SiLU；训练配置为 MSE、Adam、学习率 `1e-3`、500 epochs、batch size 8。数据同化优化初始状态，通过显式动力学积分生成轨迹，并使用 SciPy L-BFGS-B。

### 当前 Burgers 适配

当前实现保留了“Voronoi 逆映射初始化 + 变分细化”的高层思想，但核心流程不同：

- 逆观测网络复用本仓库 VoronoiCNN 适配器，不是 VIVID 的 8×8 Keras VCNN，也不是 invobs 的数据集专用 Flax 网络；
- 细化直接优化完整 Burgers 轨迹，而不是 ADAO 3DVAR 状态或由动力学传播的初值；
- 细化使用 Adam 和软 PDE 残差，不是 L-BFGS-B；
- 默认训练/细化预算为 1 epoch / 300 steps，而不是官方 VIVID 的 20 epochs / 1000 steps 或 invobs 的 500 epochs / 500–1000 steps；
- 没有直接 import 或执行 vendored 官方训练/同化代码。

因此该实现只能标记为 `vivid_style_burgers_adapted`。如需满足第 7 条，应另行实现并验证 VIVID 官方 VCNN+ADAO 流程，或为 Burgers 构建 invobs 风格的可微动力学、官方网络结构和 L-BFGS-B 两阶段流程，同时明确哪些变化仅属于数据接口适配。

## 已清理的旧入口

- Var4D/VIVID capability 仅允许 `burger + sparse_solution + full trajectory`；
- 删除 NS、Reaction-Diffusion、Shallow-Water 等旧方法入口和 NS 专用测试；
- 删除虚假的 VIVID `official_aligned` 本地适配器及 preflight 路径；
- 删除 sanity 配置中未使用的 Var4D/VIVID 资源块；
- 正式矩阵和两份消融配置继续只生成 Burgers 任务。
