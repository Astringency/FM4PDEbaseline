# RecFNO / Senseiver / Voronoi-CNN 三个 baseline 实现总结

本文总结项目中三个稀疏场重建类 baseline（RecFNO、Senseiver、Voronoi-CNN）在不同任务上的实现，包括输入输出、方法实现原理，以及官方代码与本项目适配之间的关系。

## 一、整体定位

在 `baselines/experiment_matrix.py:22` 中，这三个方法被归为同一类：

```python
SPARSE_AMORTIZED_BASELINES = {"recfno", "senseiver", "voronoicnn"}
```

含义是：**稀疏传感器 → 全局场重建的「摊销式（amortized）监督学习」基线**，即训练一个网络，对每个样本的稀疏传感器布局一次性输出全场，而不是对每个样本重新做 PDE 求解或优化。正式训练使用 `random_per_sample`，mask 会随样本和 epoch 变化。

它们不参与 `forward` / `inverse` 这类全量算子学习任务（由 FNO / DeepONet / iFNO 承担），主要跑的任务（见 `baselines/capabilities.py` 和 `configs/experiments/*.yaml`）：

- `sparse_solution`（联合稀疏重建）：从 $[O_a,O_u]$ 恢复 $[a,u]$；Burgers 则恢复包含初值的完整时空解场
- `sparse_forward`：从输入场（系数/源项/初值）的稀疏观测映射到解场
- `sparse_inverse`：从解场的稀疏观测反演系数/源项场
- 额外：`senseiver` 还支持 `time_varying_da`（时变传感器数据同化）

## 二、公共数据接口（三个方法共享）

数据由 `baselines/common/data_adapter.py` 的 `make_task()` 统一构造，模型拿到的是 `PDEBatch`（`data_adapter.py:18`）。与这三个方法直接相关的字段：

- `input_fields`：模型主输入，shape `[B, C_in, *grid]`
- `target_fields`：监督目标，shape `[B, C_out, *grid]`
- `coords`：目标网格坐标 `[1|B, N_query, dim]`（静态 2D 为 `(y, x)`；时变为 `(t, y, x)`）
- `mask`：共享或按样本的传感器掩码 `[C, *grid]` 或 `[B, C, *grid]`
- `obs_values` / `obs_coords`：传感器处取值和坐标（Senseiver 用）
- `metadata["masked_grid"]`：观测源场与 mask 的乘积（只在传感器位置非零）
- `metadata["voronoi_grid"]`：对 `masked_grid` 做最近邻填充（`common/voronoi.py`，即 Voronoi 风格的最近传感器填充）

关键逻辑在 `data_adapter.py` 的 `make_task()`：

- `sparse_solution` / `sparse_reconstruction`：静态 PDE 和 Navier–Stokes 先把输入场与解场合并为联合 target，再以它作为 `observation_source`；Burgers target 本身是包含 $u(t=0)$ 的完整轨迹
- `sparse_inverse` / `sparse_forward`：`observation_source = input_fields`（`_split_task()` 已把 inverse 的输入/目标互换），再 `input_fields = masked_grid`

所以三种方法的「稀疏输入网格」含义随任务不同而变化（详见第四节）。

## 三、三个方法各自实现

### 1. RecFNO（`baselines/methods/recfno.py` + `offical/RecFNO`）

**原理**：把稀疏观测做 Voronoi 最近邻填充，再将「Voronoi 场 + mask + 坐标网格」喂给 FNO（Fourier Neural Operator，4 层谱卷积 + 1x1 卷积残差）进行全场重建。旧的 `embedding=mask|voronoi` 开关已移除，正式输入合同固定为 `input_representation: voronoi_mask_coords`。

**官方源码**：`offical/RecFNO/model/fno.py` 的 `VoronoiFNO2d`（`fno.py:340`）是实际用的骨干网：`fc0` 提升通道 → 4 个 `SpectralConv2d + Conv2d(1x1)` 残差块 → `fc1/fc2` 投影回输出通道。

**本项目的适配**（`recfno.py`）：

- `in_channels = input_channels * 2 + 2`
- `predict()` 拼接三部分：
  - `base`：`metadata["voronoi_grid"]`，不允许回退到 zero-masked input
  - `mask`（广播到 batch）
  - `grid_channels(base)`：归一化 `(y, x)` 坐标 `[B, 2, H, W]`（`shared.py:12`）
- 后端选择：`implementation_mode != adapted` 且后端非 local 时，优先用官方 `VoronoiFNO2d`（`OfficialRecFNOVoronoiFNO2dNet`，`shared.py:66`，转成 NCHW）；否则本地 `FNO2dNet`（`shared.py:41`）。paper 配置 `official_or_skip` + `auto`，所以会走官方骨干网。

### 2. Senseiver（`baselines/methods/senseiver.py` + `offical/Senseiver`）

**原理**：基于 Perceiver-IO 的注意力网络。它**不把观测放到网格上**，而是把每个传感器当作一个 token：`[传感器坐标, 传感器值]` 作为输入序列，先通过 cross-attention 压缩到固定数量的 latent，再对每个 query 坐标（目标网格点）做 cross-attention 解码，输出该点的场值。因此模型复杂度与空间分辨率解耦，天然支持任意/不规则传感器布局和 n-D 场。

**官方源码**：`offical/Senseiver/model.py` 的 `Encoder`（`model.py:147`）和 `Decoder`（`model.py:221`）。

**本项目的适配**（`senseiver.py`）：

- 优先直接导入官方 Encoder/Decoder，本地 fallback 是一个简化 Perceiver：`sensor_proj` → `enc_attn`(latent×tokens) → `self_attn` → `dec_attn`(query×latent) → `out`
- 训练时 `supervised_training_pair()` 在 decoder 之前随机抽取至多 `batch_pixels=2048` 个 query pixel，并用同一组索引切片 target；每个 field batch 执行与官方 loader 一致的 query-batch 更新次数
- 评估时 `predict()` 使用全部 `batch.coords`，因此输出仍是完整网格

### 3. Voronoi-CNN（`baselines/methods/voronoicnn.py` + `offical/Voronoi-CNN`）

**原理**：先用 Voronoi 镶嵌把稀疏传感器值扩散成一张连续的输入图像（最近邻填充），再叠一个传感器掩码通道，然后交给一个纯 CNN 全卷积堆栈学习「稀疏场 → 全场」的映射。

**官方源码**：`offical/Voronoi-CNN/Voronoi-CNN-cy.py:161-170`，Keras 模型是 7 层 `Conv2D(48, 7x7, ReLU)` + 1 层 `Conv2D(1, 3x3)`；输入是 `[Voronoi 填充场, 传感器掩码]` 两通道（`cy.py:104,108,161`）。

**本项目的适配**（`voronoicnn.py`）：

- 因为官方是 Keras/TF 脚本不可直接导入，paper 配置用 `implementation_mode: official_architecture` + `official_backend: voronoicnn`，走 `_VoronoiCNNArchitectureNet`（`voronoicnn.py:102`）：7 层 `Conv2d(7x7) + ReLU` + `Conv2d(3x3)`，与官方架构一致
- `architecture_in_channels = target_channels * 2`，`include_coords=False`（`voronoicnn.py:21,28-29`）
- `predict()`（`voronoicnn.py:88-99`）：拼接 `[voronoi_grid, mask]`
- 另有 `backend=recfno` 时用 RecFNO 的 UNet 作为 fallback，以及本地 `ConvReconNet` 兜底

## 四、不同任务上的输入/输出

下面 `C_in` / `C_out` 表示对应 PDE 的输入场通道数 / 目标场通道数（例如 Darcy/Poisson/Helmholtz 都是单通道的「系数或源项 → 解」）。

| 方法 | 任务 | 模型输入 | 模型输出（=target） |
|---|---|---|---|
| RecFNO | sparse_solution | `[voronoi([O_a,O_u]), mask, yx 坐标]` | 联合场 `[a,u]` |
| RecFNO | sparse_forward | `[masked_input(系数/源/初值), mask, yx]` | 完整解场 |
| RecFNO | sparse_inverse | `[masked_solution, mask, yx]` | 系数/源项场 |
| Senseiver | sparse_solution | 联合观测 `[coords, values(a,u)]` token + query 坐标 | 每个 query 点的 `[a,u]` |
| Senseiver | sparse_forward | 传感器 `[coords, values]`（观测的是输入场）+ query | 解场值 |
| Senseiver | sparse_inverse | 传感器 `[coords, values]`（观测的是解场）+ query | 系数/源项值 |
| Senseiver | time_varying_da | 时变传感器 `[t, y, x, value]` token + 时变 query | 整条轨迹场 |
| Voronoi-CNN | sparse_solution | `[voronoi([O_a,O_u]), mask]` | 联合场 `[a,u]` |
| Voronoi-CNN | sparse_forward | `[voronoi_input, mask]` | 完整解场 |
| Voronoi-CNN | sparse_inverse | `[voronoi_solution, mask]` | 系数/源项场 |

具体到「输入是从哪个场取稀疏观测」，规则在 `data_adapter.py`：

- `sparse_solution`：静态 PDE 和 Navier–Stokes 观测并 mask **联合场 `[a,u]`**；Burgers 观测并恢复包含初值的完整 $T\times X$ 解场
- `sparse_forward`：观测并 mask **输入场**（系数/源项/初值），目标仍是解场
- `sparse_inverse`：观测并 mask **解场**，目标变成系数/源项场（`_split_task` 做了输入/目标互换）

## 五、关键差异一句话总结

- **RecFNO**：把稀疏信息铺成网格 → FNO 谱域重建，强在全局/分辨率不变。
- **Senseiver**：把传感器当 token → 注意力编解码，强在任意不规则/时变传感器和 n-D 场。
- **Voronoi-CNN**：Voronoi 镶嵌 + 掩码 → 纯 CNN，最简单直接，但依赖固定网格和分辨率。

三个方法在 paper 模式下的后端策略也不同：

- RecFNO 和 Senseiver 走 `official_or_skip`；paper 预检会在 vendored 官方组件不可用时跳过该组合，不把本地 fallback 冒充为正式结果
- Voronoi-CNN 因为官方是 Keras 脚本，走 `official_architecture`（PyTorch 复刻官方 Conv2D 架构）

这些策略只记录在中央配置 `baselines/configs/paper.yaml` 的 `method_by_baseline` 中，不再保留分方法配方副本。
