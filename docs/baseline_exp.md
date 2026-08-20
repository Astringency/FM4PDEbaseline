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

监督方法的训练适配保留以下官方核心设置：FNO 使用 NeuralOperator 的 H1 loss、AdamW、weight decay 与 StepLR；DeepONet 使用 DeepXDE 网络及 Adam/MSE；RecFNO 使用 L1、Adam 与 ExponentialLR；Senseiver 使用 sum-MSE、Adam 与 train-loss 早停；VoronoiCNN 使用七层 7x7 卷积栈、Adam/MSE 与 validation-loss 最佳 checkpoint。为减少无效训练，Senseiver 和 VoronoiCNN 的早停 patience 统一适配为 20。iFNO 明确记录 iFNO 预训练、VAE 预训练、联合训练三个请求预算和实际完成预算，Darcy 顺序为 iFNO→VAE→joint，Navier-Stokes 顺序为 VAE→iFNO→joint，并保留 VAE 四倍几何增强。官方 iFNO 的三个阶段本身采用固定 epoch；本项目额外对三个阶段分别执行验证集早停并分别恢复最佳权重，属于 FM4PDE 的计算预算适配：iFNO 预训练使用 `min_epochs=50`、`patience=20`，VAE 预训练使用 `min_epochs=30`、`patience=20`，二者和联合训练的下降阈值均为 `1e-4`。VAE 的验证损失关闭几何增强和随机潜变量采样，避免随机性影响停止判定；联合训练继续采用 iFNO 配置中的 `patience=12`。

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

Var4D 不进行离线训练。它使用标准控制变量变换 $u_0=u_b+Lv$、$B=LL^\top$，以去相关初值增量 $v$ 为唯一优化变量，用可微的周期伪谱 Burgers 数值模型传播完整同化窗，最小化 Balgovind 背景项和多时刻稀疏观测项；逐样本使用 SciPy L-BFGS-B，最多 500 iterations，cost-decrement tolerance 为 `1e-6`。任何 $t>0$ 的轨迹值都由动力学传播得到，不再独立优化完整 $T\times X$ 网格，也不再以 PDE 残差作为软约束。结果标记为 `canonical_math`，同时明确 Burgers 离散传播器属于任务实现，不声称复用了某个端到端官方 Var4D 仓库。

VIVID 遵照 `offical/VIVID` 与原文实现：VCNN 为 6 个 `48` 通道、`8×8`、ReLU、Keras `same` 卷积层，接 1 个单通道线性 `8×8` 输出卷积；使用 Glorot-uniform 初始化、Adam/MSE、学习率 `1e-4`、20 epochs、有效 batch size 64。128×128 训练通过微批梯度累积维持有效 batch 64，微批大小仅控制显存，不改变一次 Adam 更新的目标。统一数据协议仍使用非重叠的 45000 train + 5000 validation，而不是官方脚本内部的 5% `validation_split`。

VIVID 原文是 3D-Var，而不是 4D-Var；论文把时空 4D-Var 列为未来扩展。Burgers 适配因此把完整 $T\times X$ 解场视为一个二维状态，保留原文

$$
J(x)=\frac12\lVert x-x_b\rVert_{B^{-1}}^2
+\frac12\lVert x-x_v\rVert_{P^{-1}}^2
+\frac12\lVert y-H(x)\rVert_{R^{-1}}^2,
$$

其中 $x_v$ 是 VCNN 输出。优化从背景场 $x_b$ 开始，使用 SciPy L-BFGS-B，最多 1000 iterations，cost-decrement tolerance 为 `1e-6`；协方差尺度遵照 vendored 脚本取 $B=1000\,B_{\mathrm{Balgovind}}(L=5)$、$P=100I$、$R=I$。为避免 128×128 状态对应的稠密 $B$ 超过 2 GiB，Balgovind 逆协方差作用采用矩阵自由循环嵌入，这是实现层面的规模适配。

Var4D 的背景初值和 VIVID 的背景解场都只能由当前样本的稀疏观测做 Voronoi 填充得到，禁止读取未观测的真实首时间片或完整轨迹。相应的传感器数量消融和运行预算消融固定在 Burgers Sparse Solution Reconstruction 上。VIVID 标记为 `official_architecture` task adapter：其官方架构、训练、目标、优化器和预算已保留，但由于 PDE、状态语义、观测算子和协方差存储方式发生了任务适配，`official_native_eligible=false`。详细审计见 [`docs/var4d_vivid_official_audit.md`](var4d_vivid_official_audit.md)。

其余方程为两通道二维数据，只取散布的稀疏观测值，注意是 $O_{a}$、$O_{u}$ 各 500 个。

### Sparse Forward

给定 $O_{a}$ 求解 $u$。主要涉及 RecFNO、Senseiver、VoronoiCNN、PINN-sparse、PDE-opt、PC-BNN 六种方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 4 个 PDE，其中 PINN-sparse、PDE-opt、PC-BNN 不需要对 Navier-Stokes 进行实验

### Sparse Inverse

给定 $O_{u}$ 求解 $a$。主要涉及 RecFNO、Senseiver、VoronoiCNN、PINN-sparse、PDE-opt、PC-BNN 六种方法，覆盖 Poisson、Helmholtz、Darcy、Navier-Stokes 4 个 PDE，其中 PINN-sparse、PDE-opt、PC-BNN 不需要对 Navier-Stokes 进行实验

---

## 其他注意事项

1. RecFNO、Senseiver、VoronoiCNN 等在训练的过程中，不要完全使用固定的 mask，应当随 epoch 变动
2. 求解 $(a,u)$ 需要分别报告 $a$ 和 $u$ 的误差，不能只报告拼接后的整体误差
