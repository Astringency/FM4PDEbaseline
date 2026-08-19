# Var4D / VIVID 实现对照审计

审计对象为原文 *Efficient deep data assimilation with sparse observations and time-varying sensors*、仓库内 `offical/VIVID` 快照，以及 Burgers Sparse Solution Reconstruction 适配。vendored 源码没有记录上游 commit/tag，因此这里只能证明与当前仓库快照的实现对齐，不能证明与某个上游 revision 逐字一致。

## 当前结论

- Var4D 已改为 Burgers 强约束 4D-Var：只优化初始状态，由可微动力学传播完整轨迹，使用背景项和多时刻观测项，不再优化自由轨迹或加入软 PDE loss。它可标记为 `canonical_math`，但不是端到端 official-native 代码。
- VIVID 已恢复 vendored 官方 VCNN 架构、初始化、训练损失、优化器、学习率、训练预算、三项变分目标、协方差尺度和 L-BFGS-B 预算。它可标记为 `official_architecture` task adapter；Burgers 状态语义和规模适配仍使 `official_native_eligible=false`。
- 两者继续仅用于 `burger + sparse_solution + full trajectory`；旧的 NS、Reaction-Diffusion、Shallow-Water 入口保持删除状态。

## Var4D

仓库没有 vendored 的特定 Var4D 官方程序，因此核对标准是强约束 4D-Var 数学定义。

| 项目 | 当前实现 |
| --- | --- |
| 控制变量 | 去相关初值增量 $v$，通过 $u_0=u_b+Lv$、$B=LL^\top$ 映射到物理初值；每样本 $X$ 个自由度 |
| 状态轨迹 | 与数据生成约定对齐的周期 Fourier 伪谱空间离散，黏性项隐式、非线性项 predictor-corrector；每个保存间隔 1 个积分步 |
| 代价函数 | $\frac12\lVert u_0-u_b\rVert_{B^{-1}}^2+\frac12\lVert H(M(u_0))-y\rVert_{R^{-1}}^2$ |
| $B$ | 一维 Balgovind/Matérn-3/2；按 Burgers 数值尺度取 variance 0.01、长度 5 个网格点 |
| $R$ | 默认 $I$ |
| 优化器 | SciPy `L-BFGS-B` |
| 预算 | 500 iterations，cost-decrement tolerance `1e-6` |
| 背景 | 当前样本稀疏观测的 Voronoi 场在 $t=0$ 的切片；不读取真实初值 |
| 标签 | `canonical_strong_constraint_4dvar` / `canonical_math` |

结果 metadata 记录 requested/completed iterations、function/gradient evaluations、终止原因、优化变量形状、传播器和内步数。由于每次 objective evaluation 都执行完整 Burgers 传播，预算不再把隐藏的内部优化阶段记成一次。

## VIVID 原文与 vendored 官方实现

原文 VIVID 的核心是把 VCNN 逆算子输出 $x_v$ 直接加入 3D-Var 目标：

$$
J(x)=\tfrac12\lVert x-x_b\rVert_{B^{-1}}^2
+\tfrac12\lVert x-x_v\rVert_{P^{-1}}^2
+\tfrac12\lVert y-H(x)\rVert_{R^{-1}}^2.
$$

`offical/VIVID/VCNN_training.py` 的网络与训练为：

- 输入和输出都是单通道二维场；
- 6 个 `48` 通道、`8×8`、ReLU、Keras `same` 隐藏卷积；
- 1 个单通道、线性、`8×8`、Keras `same` 输出卷积；
- Keras 默认 Glorot-uniform kernel、zero bias；
- MSE、Adam、学习率 `1e-4`、20 epochs、batch size 64、`validation_split=0.05`。

`offical/VIVID/VIVID.py` 使用 ADAO 3DVAR 的 `LBFGSB`，最大 1000 steps，`CostDecrementTolerance=1e-6`。VIVID 示例使用 `1000 * Balgovind(50,5)` 作为背景误差协方差，逆算子误差方差为 100，稀疏观测误差方差为 1，并从背景状态而不是 VCNN 输出开始优化。

原文也明确将 spatial-temporal 4D-Var 作为未来扩展。因此本实现没有把 VIVID 改写成“优化初值并传播动力学”的 invobs/4D-Var 方法。

## 当前 VIVID port 保留项

| 官方核心 | 当前实现 |
| --- | --- |
| VCNN 层数、通道、kernel、激活、same padding | 完整保留；偶数 kernel 使用与 Keras 一致的前 3/后 4 非对称 padding |
| 初始化 | Glorot-uniform kernel、zero bias |
| 离线训练 | Adam/MSE，`1e-4`，20 epochs，有效 batch 64 |
| VIVID 目标 | 完整保留 $J_b+J_p+J_o$ |
| 优化初值 | 从 $x_b$ 开始，而不是从 $x_v$ 开始 |
| 优化器与预算 | SciPy L-BFGS-B，1000 iterations，tolerance `1e-6` |
| 协方差尺度 | $B=1000B_{\rm Balgovind}(L=5)$、$P=100I$、$R=I$ |
| 预算审计 | requested/completed iterations 与 function/gradient evaluations 均写入结果 metadata |
| checkpoint | VCNN 权重和“已训练”持久 buffer 一并保存/恢复 |

## 必须披露的 Burgers 适配

这些差异是为了执行统一 Burgers Sparse Solution Reconstruction，不应被描述为官方原生实验：

1. 官方浅水实验重建单时刻 50×50 空间场；当前任务把完整 128×128 的时间—空间解场视为二维 3D-Var 状态。
2. 官方浅水观测算子包含其物理变量变换；Burgers 使用直接稀疏采样 $H$。
3. 统一协议使用 45000 train + 5000 independent validation，而不是 Keras `validation_split=0.05`；训练 mask 随 epoch 变化。
4. 为维持 128×128 下的有效 batch 64，反向传播使用微批梯度累积。
5. 官方稠密 Balgovind 矩阵在 128×128 上超过 2 GiB；当前使用同一径向 kernel 的循环嵌入，通过 FFT 计算 $B^{-1}$ 二次型。
6. Keras/ADAO 被审计后的 PyTorch/SciPy port 取代，因为 vendored 脚本在 import 时执行全局实验，不能作为安全的统一 runner 组件直接调用。
7. 背景场由当前样本的稀疏观测 Voronoi 填充构造，避免官方 twin experiment 中“真值加噪背景”造成隐藏真值泄露。

因此 VIVID 的准确 provenance 是 `official_architecture_vivid_burgers_task_adapter`，而不是 `official_code` 或 `official_native`。

## 旧设置清理状态

- capability 仅允许 Burgers 完整轨迹重建；
- 正式矩阵和两份相关消融只生成 Burgers Var4D/VIVID；
- VIVID 旧的 1 epoch / 300-step / Adam trajectory refinement 配置已删除；
- Var4D 旧的 full-trajectory Adam + soft PDE residual 路径已删除；
- VIVID 不再依赖本仓库通用 VoronoiCNN，也不再声称使用 `invobs-data-assimilation` 的两阶段流程。
