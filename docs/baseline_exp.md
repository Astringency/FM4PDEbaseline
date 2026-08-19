# FM4PDE Baseline 实验方案

## 总体要求

设 $a$ 为源项（静态方程）或初始状态（时变方程 $t=0$），$u$ 为解场（静态方程）或终端状态（时变方程 $t = T$ 时的值，Burgers 除外，见下面具体设置），$O_{a}$ 和 $O_{u}$ 分别表示二者的观测值。

1. 实验涉及五个 PDE 上进行：Poisson、Helmholtz、Darcy、Burgers、Navier-Stokes
2. 基于训练的模型默认存储 checkpoint 以供训练之后的复用，不必每次评估都需重新训练
3. 训练数据 50000 个，其中 5000 个为验证数据（45000 train + 5000 val），测试数据 1000 个
4. 实验需要有早停逻辑，早停主要以损失下降平缓为判定依据，判定阈值为 1e-4
5. 稀疏任务的观测点个数默认取 500 个，Burgers 有特殊情况，见下面的具体设置
6. 需要确保各个任务不出现数据泄露，测试数据不可包含在训练数据当中
7. 各个 Baseline 方法应当在 Official 实现的基础上进行任务适配，保留官方网络架构、损失、优化器、学习率调度和分阶段训练等核心流程；统一的是 FM4PDE 的任务输入、数据划分、预算记录、checkpoint 与评估接口，不以“统一训练器”为由改写方法核心
8. 采样评估结果报告相对误差和 PDE 指标。主 PDE 指标统一指内部方程残差均方 `pde_residual`：静态 Forward 使用 $R(a_{true},u_{pred})$，静态 Inverse 使用 $R(a_{pred},u_{true})$，联合重建使用 $R(a_{pred},u_{pred})$。`bc_residual`、`ic_residual` 以及二者与内部残差的加权和 `physics_loss` 作为分项诊断，不代替主 PDE 指标。仅输出初末端点的 Navier-Stokes 任务报告 `endpoint_secant` 近似，并通过 `residual_mode` 与完整轨迹残差区分。

正式实验的数据文件由 `configs/data_files/formal_128.yaml` 逐个列出。矩阵、数据协议校验器和运行器必须传递同一份文件清单；配置清单存在时禁止回退到 glob 自动发现，缺失文件直接报错。清单中的相对路径以 `DATA_ROOT` 为根，因此迁移到其他机器时不需要修改绝对路径。

`train_size` 表示训练/验证共享数据池的总大小，而不是最终用于拟合的样本数。当没有单独的 `val` 文件时，验证集固定取该数据池尾部：训练区间为 `[0, train_size - val_size)`，验证区间为 `[train_size - val_size, train_size)`。当前统一使用 `train_size: 50000`、`val_size: 5000`，对应 45000 train + 5000 val；修改划分后需重新生成数据 manifest 和实验矩阵。
9. 每个实验只跑一次即可，不需要像现在这样有三个 seed
10. 评估得到的采样结果需要存储为可再次读入的文件，并绘制 pdf 图片
11. 所有已完成实验进入同一份 `summary` 与 LaTeX 表；不再按 main/supplement 等展示层级过滤结果。实现来源、是否适配及 official-native 资格仍作为审计字段保留，不参与结果丢弃

监督方法的训练适配保留以下官方核心设置：FNO 使用 NeuralOperator 的 H1 loss、AdamW、weight decay 与 StepLR；DeepONet 使用 DeepXDE 网络及 Adam/MSE；RecFNO 使用 L1、Adam 与 ExponentialLR；Senseiver 使用 sum-MSE、Adam 与 train-loss 早停；VoronoiCNN 使用七层 7x7 卷积栈、Adam/MSE 与 validation-loss 最佳 checkpoint。为减少无效训练，Senseiver 和 VoronoiCNN 的早停 patience 统一适配为 20。iFNO 明确记录 iFNO 预训练、VAE 预训练、联合训练三个预算，Darcy 顺序为 iFNO→VAE→joint，Navier-Stokes 顺序为 VAE→iFNO→joint，并保留 VAE 四倍几何增强。

PINN-sparse 对每个测试样本先执行 Adam（上限 1000 iterations），再执行 L-BFGS（上限 500 steps）。结果中必须分别记录 `adam_iterations`、`lbfgs_steps` 与两阶段上限之和 `total_optimization_steps`，不得只用单一 `steps` 字段代替。

---

## 具体实验设置

### Full Forward

给定 $a$，求解 $u$。主要涉及 FNO、DeepONet、iFNO 三个方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 四个 PDE

### Full Inverse

给定 $u$，求解 $a$。主要涉及 iFNO 方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 四个 PDE

### Sparse Solution Reconstruction

给定 $[O_{a}, O_{u}]$ 求解 $[a, u]$。RecFNO、Senseiver、VoronoiCNN 覆盖 Poisson、Helmholtz、Darcy、Burgers、Navier-Stokes 5 个 PDE；Var4D、VIVID 仅用于 Burgers，不再在 Navier-Stokes（`nsnonbounded`）上设置实验任务。

Burgers 是一维时变数据，重建目标为完整的 $128\times128$（时间 $\times$ 空间）轨迹，其中首个时间片对应 $a=u(t=0)$，其余时间片对应 $u$。五种方法均运行以下两种观测协议：

1. 在完整 $128\times128$ 时间—空间网格上，每个样本独立随机选择总计 500 个散布观测点；
2. 每个样本独立选择 5 个完整时间片，共 $5\times128$ 个观测点。

Var4D 不进行离线训练，对每个测试样本执行最多 500 次轨迹优化；VIVID 先用训练集拟合 1 epoch 的逆观测网络，再对每个测试样本执行最多 300 次变分细化。两者的初始背景场只能由当前样本的稀疏观测做 Voronoi 填充得到，禁止读取未观测的真实首时间片或完整轨迹。相应的传感器数量消融和运行预算消融也固定在 Burgers Sparse Solution Reconstruction 上。

经 vendored 官方源码对照，当前 Var4D 是直接优化完整轨迹并以 PDE 残差作软约束的 weak-constraint adaptation；当前 VIVID 保留 Voronoi 逆映射初始化和变分细化思想，但没有保留官方网络、L-BFGS-B 优化器及官方训练预算。因此两者暂不满足本方案第 7 条的 official-core 要求，结果必须标记为 `adapted`，不得标记为 canonical/official/official-aligned 或 official-native。详细差异见 [`docs/var4d_vivid_official_audit.md`](var4d_vivid_official_audit.md)。

其余方程为两通道二维数据，只取散布的稀疏观测值，注意是 $O_{a}$、$O_{u}$ 各 500 个。

### Sparse Forward

给定 $O_{a}$ 求解 $u$。主要涉及 RecFNO、Senseiver、VoronoiCNN、PINN-sparse、PDE-opt、PC-BNN 六种方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 4 个 PDE，其中 PINN-sparse、PDE-opt、PC-BNN 不需要对 Navier-Stokes 进行实验

### Sparse Inverse

给定 $O_{u}$ 求解 $a$。主要涉及 RecFNO、Senseiver、VoronoiCNN、PINN-sparse、PDE-opt、PC-BNN 六种方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 4 个 PDE，其中 PINN-sparse、PDE-opt、PC-BNN 不需要对 Navier-Stokes 进行实验

---

## 其他注意事项

1. RecFNO、Senseiver、VoronoiCNN 等在训练的过程中，不要完全使用固定的 mask，应当随 epoch 变动
2. 求解 $(a,u)$ 需要分别报告 $a$ 和 $u$ 的误差，不能只报告拼接后的整体误差
