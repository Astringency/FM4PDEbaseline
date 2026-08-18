# RecFNO / Senseiver / Voronoi-CNN 三个 baseline 实现总结

本文总结项目中三个稀疏场重建类 baseline（RecFNO、Senseiver、Voronoi-CNN）在不同任务上的实现，包括输入输出、方法实现原理，以及官方代码与本项目适配之间的关系。

## 一、整体定位

在 `baselines/experiment_matrix.py:22` 中，这三个方法被归为同一类：

```python
SPARSE_AMORTIZED_BASELINES = {"recfno", "senseiver", "voronoicnn"}
```

含义是：**稀疏传感器 → 全局场重建的「摊销式（amortized）监督学习」基线**，即训练一个网络，对任意（但固定的）传感器布局一次性输出全场，而不是对每个样本重新做 PDE 求解或优化。

它们不参与 `forward` / `inverse` 这类全量算子学习任务（由 FNO / DeepONet / iFNO 承担），主要跑的任务（见 `baselines/capabilities.py:308/344/380` 和 `configs/experiments/*.yaml`）：

- `sparse_solution`（稀疏重建）：从目标场自身的稀疏观测恢复全场
- `sparse_forward`：从输入场（系数/源项/初值）的稀疏观测映射到解场
- `sparse_inverse`：从解场的稀疏观测反演系数/源项场
- 额外：`senseiver` 还支持 `time_varying_da`（时变传感器数据同化）

## 二、公共数据接口（三个方法共享）

数据由 `baselines/common/data_adapter.py` 的 `make_task()` 统一构造，模型拿到的是 `PDEBatch`（`data_adapter.py:18`）。与这三个方法直接相关的字段：

- `input_fields`：模型主输入，shape `[B, C_in, H, W]`（时变任务会把时间轴展平成通道）
- `target_fields`：监督目标，shape `[B, C_out, H, W]`
- `coords`：目标网格坐标 `[B, N_query, dim]`（静态 2D 为 `(y, x)`；时变为 `(t, y, x)`）
- `mask`：传感器掩码 `[C, H, W]`
- `obs_values` / `obs_coords`：传感器处取值和坐标（Senseiver 用）
- `metadata["masked_grid"]`：`solution/input * mask`（只在传感器位置非零）
- `metadata["voronoi_grid"]`：对 `masked_grid` 做最近邻填充（`common/voronoi.py`，即 Voronoi 风格的最近传感器填充）

关键逻辑在 `make_task()` 的 `data_adapter.py:319-375`：

- `sparse_solution` / `sparse_reconstruction`：`observation_source = target_fields`，然后 `input_fields = masked_grid`
- `sparse_inverse` / `sparse_forward`：`observation_source = input_fields`（`_split_task` 在 `data_adapter.py:662-665` 已把 inverse 的输入/目标互换），再 `input_fields = masked_grid`

所以三种方法的「稀疏输入网格」含义随任务不同而变化（详见第四节）。

## 三、三个方法各自实现

### 1. RecFNO（`baselines/methods/recfno.py` + `offical/RecFNO`）

**原理**：把稀疏观测编码成一个 2D 网格，再喂给 FNO（Fourier Neural Operator，4 层谱卷积 + 1x1 卷积残差），用谱域全局算子完成分辨率不变的重建。稀疏信息通过「embedding + mask + 坐标网格」注入，有两种 embedding 可选：`mask`（掩码网格）或 `voronoi`（Voronoi 填充网格）。

**官方源码**：`offical/RecFNO/model/fno.py` 的 `VoronoiFNO2d`（`fno.py:340`）是实际用的骨干网：`fc0` 提升通道 → 4 个 `SpectralConv2d + Conv2d(1x1)` 残差块 → `fc1/fc2` 投影回输出通道。

**本项目的适配**（`recfno.py`）：

- `in_channels = target_channels * 2 + 2`（`recfno.py:20`）
- `predict()`（`recfno.py:76-87`）拼接三部分：
  - `base`：embedding 字段（`embedding=mask` 用 `masked_grid`，`voronoi` 用 `voronoi_grid`）
  - `mask`（广播到 batch）
  - `grid_channels(base)`：归一化 `(y, x)` 坐标 `[B, 2, H, W]`（`shared.py:12`）
- 后端选择：`implementation_mode != adapted` 且后端非 local 时，优先用官方 `VoronoiFNO2d`（`OfficialRecFNOVoronoiFNO2dNet`，`shared.py:66`，转成 NCHW）；否则本地 `FNO2dNet`（`shared.py:41`）。paper 配置 `official_or_skip` + `auto`，所以会走官方骨干网。

### 2. Senseiver（`baselines/methods/senseiver.py` + `offical/Senseiver`）

**原理**：基于 Perceiver-IO 的注意力网络。它**不把观测放到网格上**，而是把每个传感器当作一个 token：`[传感器坐标, 传感器值]` 作为输入序列，先通过 cross-attention 压缩到固定数量的 latent，再对每个 query 坐标（目标网格点）做 cross-attention 解码，输出该点的场值。因此模型复杂度与空间分辨率解耦，天然支持任意/不规则传感器布局和 n-D 场。

**官方源码**：`offical/Senseiver/model.py` 的 `Encoder`（`model.py:147`）和 `Decoder`（`model.py:221`）。

**本项目的适配**（`senseiver.py`）：

- 优先用官方 Encoder/Decoder（`senseiver.py:31-64`）
- 本地 fallback（`senseiver.py:72-78`）是一个简化 Perceiver：`sensor_proj` → `enc_attn`(latent×tokens) → `self_attn` → `dec_attn`(query×latent) → `out`
- `predict()`（`senseiver.py:95-121`）：输入 `obs_values/obs_coords`（或稠密回退用全部网格值当伪传感器），query 用 `batch.coords`

### 3. Voronoi-CNN（`baselines/methods/voronoicnn.py` + `offical/Voronoi-CNN`）

**原理**：先用 Voronoi 镶嵌把稀疏传感器值扩散成一张连续的输入图像（最近邻填充），再叠一个传感器掩码通道，然后交给一个纯 CNN 全卷积堆栈学习「稀疏场 → 全场」的映射。

**官方源码**：`offical/Voronoi-CNN/Voronoi-CNN-cy.py:161-170`，Keras 模型是 7 层 `Conv2D(48, 7x7, ReLU)` + 1 层 `Conv2D(1, 3x3)`；输入是 `[Voronoi 填充场, 传感器掩码]` 两通道（`cy.py:104,108,161`）。

**本项目的适配**（`voronoicnn.py`）：

- 因为官方是 Keras/TF 脚本不可直接导入，paper 配置用 `implementation_mode: official_architecture` + `official_backend: voronoicnn`，走 `_VoronoiCNNArchitectureNet`（`voronoicnn.py:102`）：7 层 `Conv2d(7x7) + ReLU` + `Conv2d(3x3)`，与官方架构一致
- `architecture_in_channels = target_channels * 2`，`include_coords=False`（`voronoicnn.py:21,28-29`）
- `predict()`（`voronoicnn.py:88-99`）：拼接 `[voronoi_grid, mask]`
- 另有 `backend=recfno` 时用 RecFNO 的 UNet 作为 fallback，以及本地 `ConvReconNet` 兜底

## 四、不同任务上的输入/输出

下面 `C_in` / `C_out` 表示对应 PDE 的输入场通道数 / 目标场通道数（例如 darcy/poisson/helmholtz 是 `源项 → 解` 单通道，shallow_water 是三通道 `h, hu, hv`）。

| 方法 | 任务 | 模型输入 | 模型输出（=target） |
|---|---|---|---|
| RecFNO | sparse_solution | `[masked_solution, mask, yx 坐标]`，`C_in = C_out` | 完整解场 `[B, C, H, W]` |
| RecFNO | sparse_forward | `[masked_input(系数/源/初值), mask, yx]` | 完整解场 |
| RecFNO | sparse_inverse | `[masked_solution, mask, yx]` | 系数/源项场 |
| Senseiver | sparse_solution | 传感器 `[coords, values]` token + query 坐标 | 每个 query 点的解场值 |
| Senseiver | sparse_forward | 传感器 `[coords, values]`（观测的是输入场）+ query | 解场值 |
| Senseiver | sparse_inverse | 传感器 `[coords, values]`（观测的是解场）+ query | 系数/源项值 |
| Senseiver | time_varying_da | 时变传感器 `[t, y, x, value]` token + 时变 query | 整条轨迹场 |
| Voronoi-CNN | sparse_solution | `[voronoi_solution, mask]` | 完整解场 |
| Voronoi-CNN | sparse_forward | `[voronoi_input, mask]` | 完整解场 |
| Voronoi-CNN | sparse_inverse | `[voronoi_solution, mask]` | 系数/源项场 |

具体到「输入是从哪个场取稀疏观测」，规则在 `data_adapter.py`：

- `sparse_solution`：观测并 mask **目标场**（解场），`input_fields = masked_grid`
- `sparse_forward`：观测并 mask **输入场**（系数/源项/初值），目标仍是解场
- `sparse_inverse`：观测并 mask **解场**，目标变成系数/源项场（`_split_task` 做了输入/目标互换）

## 五、关键差异一句话总结

- **RecFNO**：把稀疏信息铺成网格 → FNO 谱域重建，强在全局/分辨率不变。
- **Senseiver**：把传感器当 token → 注意力编解码，强在任意不规则/时变传感器和 n-D 场。
- **Voronoi-CNN**：Voronoi 镶嵌 + 掩码 → 纯 CNN，最简单直接，但固定网格、分辨率不不变。

三个方法在 paper 模式下的后端策略也不同：

- RecFNO 和 Senseiver 走 `official_or_skip`（优先导入 `offical/` 下的官方组件，否则本地适配）
- Voronoi-CNN 因为官方是 Keras 脚本，走 `official_architecture`（PyTorch 复刻官方 Conv2D 架构）

这些策略分别记录在 `baselines/configs/paper/recfno.yaml`、`senseiver.yaml`、`voronoicnn.yaml` 以及 `baselines/configs/paper.yaml` 的 `method_by_baseline`（`paper.yaml:54-71`）中。
