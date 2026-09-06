# 基于神经网络和自动微分的叶轮机械彻体力方法

## Neural Network-Based Body Force Method for Turbomachinery

**中文** | [English](README_EN.md)

---
<img width="2882" height="2603" alt="BFM_Procedure" src="https://github.com/user-attachments/assets/27d04fa3-2a7b-4362-a516-f9874742e8ee" />

<img width="3663" height="2316" alt="BFM_MESH" src="https://github.com/user-attachments/assets/2aa1b2cb-21f7-413f-94b0-289c3b238e74" />





## 1. 项目简介

本项目提出了一种**全新的彻体力计算流程**，首次将神经网络与自动微分技术引入叶轮机械彻体力建模领域，以 NASA Rotor 37 为例进行验证。本案例的NASA Rotor 37数据来自于https://github.com/Yashay03/Axial-Compressor-Rotor-37

**推荐使用AI协助搭建环境，运行和理解本项目内容！！！**

**核心思路**：
1. 从 CFX 单通道定常计算提取子午面通量数据
2. 使用神经网络拟合 λ（金属堵塞系数）×通量（$\rho UU$) 场，获得连续可微的表达式
3. 通过自动微分计算导数，代入动量方程反演彻体力分布
4. 将彻体力注入 OpenFOAM 全环网格，进行三维纯净流道的非定常计算

---

## 2. 研究背景

### 2.1 传统方法的局限性

| 方法 | 优点 | 缺点 |
|------|------|------|
| **全三维 CFD** | 精度高、捕捉细节 | 计算耗时极长，难以进行多工况/非定常分析 |
| **低维模型** | 计算快速 | 无法捕捉三维失速现象 |

### 2.2 本方法的定位

**彻体力方法（Body Force Method, BFM）** 是介于两者之间的折中方案：
- 用低维模型（单通道 CFD）生成彻体力源项
- 在三维纯净流道中进行全环非定常计算
- 兼顾计算效率与三维流动捕捉能力
- 可以用于任意稳定工况的粗略非定常计算，若要提升近失速或跨工况泛化时的可靠性，**必须**自行从多个工况中提取彻体力并制作在线查找表
---

## 3. 项目结构

本项目包含两个核心模块：

```
Arisa_Benneke_Method/
│
├── Bodyforce_Method/                 # 彻体力生成模块
│   ├── NASA_ROTOR_37/                # NASA ROTOR 37 示例案例
│   │   ├── ANN_TrainOneMLP.py       # Python 神经网络训练脚本
│   │   ├── ANN_Pre_Processing.C      # C++ 神经网络推理（LibTorch）
│   │   ├── Benneke_Pre_Processing.C  # 传统 IDW 插值方法（对比参考）
│   │   ├── BodyForceVisualizer.py    # 彻体力可视化工具
│   │   │
│   │   ├── optimization_results/    # RBF 插值超参数（计算堵塞因子子午分布）
│   │   │   └── best_parameters_separate.txt
│   │   │
│   │   ├── ANN_Output/               # 训练输出
│   │   │   ├── flux_mlp_traced.pt    # TorchScript 模型（ANN_Pre_Processing 读取）
│   │   │   ├── flux_mlp_best.pt      # 最优 checkpoint
│   │   │   └── normalization_params.csv
│   │   │
│   │   ├── CFX_Output_Benneke_Flux.csv  # CFX 导出的通量数据
│   │   ├── constant/                 # OpenFOAM 网格与物理属性
│   │   └── system/                   # 求解器配置
│   │
│   └── Initializer/                  # 全环计算初始化案例
│
├── ArisaSTALL/                       # 三维全环欧拉求解器
│   ├── ArisaSTALL.C                  # 主求解器
│   ├── ArisaSTALL.H                  # 头文件
│   ├── momentumPredictor.C           # 动量预测器
│   ├── thermophysicalPredictor.C     # 能量预测器
│   ├── correctPressure.C             # 压力修正
│   ├── Make/                         # 编译配置
│   └── Solver_README.md              # 求解器详细说明
│
└── README.md                         # 本文件
```

---

## 4. 核心创新点

### 4.1 首次引入神经网络 + 自动微分

传统彻体力方法（如 Benneke 方法）使用 **IDW 插值** 或 **多项式拟合** 来处理离散通量数据，存在以下问题：
- 插值结果不光滑，导数计算误差大
- 需要手工调参，泛化能力差
- 积分方法需要知道子午面上的网格信息，迁移性弱

**本方法的优势**：
- 神经网络天然提供**光滑、可微**的拟合结果
- 自动微分**精确计算导数**，无需手工推导
- 支持**任意叶轮机械**

### 4.2 网络介绍：直接学习 λ×通量
只需提供通量的散点数据即可（子午坐标+6个通量值+金属堵塞分布+密度）。

网络输出 **λ×通量**（而非单独的 λ 和通量），避免了推理时的乘积运算，减少了误差累积：

```
网络输出: [ρ, λ·Trr, λ·Trt, λ·Trz, λ·Ttt, λ·Tzt, λ·Tzz, λ]
         ↓ 自动微分
彻体力:   [f_r, f_θ, f_z]
```

### 4.3 批量推理

C++ 推理使用 **批量矩阵运算**，一次处理所有网格单元，该工况下的彻体力场，计算时由openFOAM的fvModels调用。

### 4.4 支持金属堵塞修正的欧拉求解器

配套可压缩**欧拉**求解器 **ArisaSTALL** 继承自 OpenFOAM v13 的 `fluid` 求解器，支持：
- 堵塞因子 λ 的动量/能量方程修正
- 三维全环非定常计算
- 与神经网络彻体力无缝对接

---

## 5. 方法局限性

- 准确性受**单通道 CFD 质量**影响
- 全环网格分布需与单通道数据**空间匹配**
- 训练参数（网络结构、学习率等）需**调优**

---

## 6. 数学原理

### 6.1 彻体力公式

从动量方程出发，彻体力定义为：

$$
\lambda \rho \mathbf{f} = \frac{1}{r} \frac{\partial (\lambda r \mathbf{T}_r)}{\partial r} + \frac{\partial (\lambda \mathbf{T}_z)}{\partial z} - \frac{\lambda \mathbf{T}_\theta}{r}
$$

其中 $\mathbf{T}$ 为通量张量：

$$
\mathbf{T} = \begin{bmatrix} T_{rr} & T_{r\theta} & T_{rz} \\ T_{\theta r} & T_{\theta\theta} & T_{\theta z} \\ T_{zr} & T_{z\theta} & T_{zz} \end{bmatrix}
$$

各分量定义：
- $T_{rr} = \rho V_r^2 + p$
- $T_{r\theta} = \rho V_r V_\theta$
- $T_{rz} = \rho V_r V_z$
- $T_{\theta\theta} = \rho V_\theta^2 + p$
- $T_{\theta z} = \rho V_\theta V_z$
- $T_{zz} = \rho V_z^2 + p$

### 6.2 自动微分链式法则

网络输入为归一化坐标 $(r_{norm}, z_{norm})$，物理导数为：

$$
\frac{\partial}{\partial r_{phys}} = \frac{\partial}{\partial r_{norm}} \cdot \frac{1}{r_{max} - r_{min}}
$$

$$
\frac{\partial}{\partial z_{phys}} = \frac{\partial}{\partial z_{norm}} \cdot \frac{1}{z_{max} - z_{min}}
$$

### 6.3 堵塞因子修正

堵塞因子 λ 表示流体体积分数（0 < λ ≤ 1），用于模拟金属叶片对流体流动的影响。

$\lambda$ 定义为 $N\frac{\theta_{PS}-\theta_{SS}}{2\pi}$，其中 $N$ 为叶片数量，$\theta$ 为任意为 $r$ , $z$ 处叶片的周向坐标（分别为相邻两个叶片的压力面和吸力面）。

修正后的动量方程：

$$
\frac{\partial (\lambda \rho \mathbf{U})}{\partial t} + \nabla \cdot (\lambda \rho \mathbf{U} \mathbf{U}) = -\nabla (\lambda p) + \lambda \rho \mathbf{g} + \lambda \mathbf{f}
$$

修正后的能量方程：

$$
\frac{\partial (\lambda \rho e)}{\partial t} + \nabla \cdot (\lambda \rho e \mathbf{U}) = -\nabla \cdot (\lambda p \mathbf{U}) + \lambda S_e
$$

---

## 7. 使用流程

> 建议在 Linux 虚拟机（OpenFOAM v13）中运行 C++ 工具，与主机共享文件夹；主机运行 Python 训练脚本。
>
> **本仓库自带了一套粗糙网格的 NASA Rotor 37 算例（`constant/polyMesh`、`0/` 等），已包含训练好的网络与彻体力场，可直接从 Step 7 的 `foamRun` 一键运行。** 若要为自己的叶轮机械建模，请从 Step 1 开始。

### Step 0: 准备网格

1. 画好**纯净流道全环网格**（叶片不参与几何建模，由彻体力 + 堵塞因子代替）。
2. 导入到算例文件夹（`constant/polyMesh`）。**本仓库已直接附带转换好的 `constant/polyMesh`**，无需任何网格文件即可运行；若自备网格，可用 `gmshToFoam` / `fluentMeshToFoam` 从 `.msh`/`.cas` 转换生成 `constant/polyMesh`。
3. **务必做交界面黏合**：用 `mergePairs` / `stitchMesh` 或在导入时把分片区域胶合（glue）为连续网格，并用 `createPatch` 设定好进口、出口、壁面等边界。
4. 定义 `ROTOR_FLUID` 单元区（cellZone）——彻体力只作用在该区域。

### Step 1: 从单通道 CFD 导出子午面数据

从单通道定常 CFD（如 CFX）导出以下子午面（周向平均）数据：

- 叶片表面（压力面/吸力面）坐标 → `CFX_Output_Blockage.csv`（用于算堵塞因子 λ 的子午分布）
- 子午面坐标 (R, Z)、密度 ρ、通量分量 Trr/Trt/Trz/Ttt/Tzt/Tzz、堵塞因子 λ → `CFX_Output_Benneke_Flux.csv`

### Step 2: 编译 ArisaSTALL 求解器

`ArisaSTALL` 是继承自 OpenFOAM v13 `fluid` 求解器的可压缩欧拉求解器，`wmake` 后生成 `libArisaSTALL.so`，由 `foamRun` 通过 `controlDict` 里的 `solver ArisaSTALL;` 加载。

```bash
cd ArisaSTALL
wmake          # 需要 OpenFOAM v13
cd ..
```

### Step 3: 编译预处理工具

彻体力场的生成有三种方法，均为算例文件夹下的单个 `.C` 文件工具，用 `wmake` 编译（共享同一份 `Make/files` 与 `Make/options`）：

| 工具 | 方法 | 依赖 |
|------|------|------|
| `ANN_Pre_Processing.C` | 神经网络 + 自动微分（**推荐，本项目主方法**） | LibTorch |
| `MLS_Pre_Processing.C` | 移动最小二乘（Moving Least Squares）局部多项式加权拟合 | 无 |
| `Benneke_Pre_Processing.C` | 传统 RBF/IDW 插值（对比参考） | 无 |
| `calculateBlockage.C` | 堵塞因子 λ 与训练数据生成 | 无 |

`wmake` 一次只能编译一个可执行程序：打开 `Make/files`，把要编译的那一组 `SOURCE += xxx.C` / `EXE = ...` 两行取消注释、其余注释掉，然后 `wmake`。仓库 `make/files` 已写好全部配置并默认激活 `MLS_Pre_Processing.C`。

> 注意：仓库目录名是小写 `make`，而 OpenFOAM 默认读取 `Make/files`；若 `wmake` 找不到文件，请把目录重命名为 `Make`（`mv make Make`）后再编译。

```bash
# 例：编译神经网络工具（在 Make/files 中激活 ANN_Pre_Processing.C 两行）
wmake
# 例：编译移动最小二乘工具（在 Make/files 中激活 MLS_Pre_Processing.C 两行）
wmake
# 例：编译堵塞因子工具（在 Make/files 中激活 calculateBlockage.C 两行）
wmake
```

> 提示：`Make/options` 里的 LibTorch 路径（如 `/home/dyfluid/libtorch`）需改成你自己的路径。只有 `ANN_Pre_Processing` 依赖 LibTorch；`MLS_Pre_Processing`、`Benneke_Pre_Processing`、`calculateBlockage` 均不依赖。

### Step 4: 计算堵塞因子并生成训练数据

```bash
# 读取 CFX_Output_Blockage.csv 与 CFX_Output_Benneke_Flux.csv
# 输出 constant/lambda（堵塞因子场）与 openFOAM_Input_Force.csv（带 λ 列的训练数据）
calculateBlockage
```

### Step 5: 训练神经网络（在主机上用 Python）

```bash
# 读取 openFOAM_Input_Force.csv，输出到 ANN_Output/
python ANN_TrainOneMLP.py
```

**输出文件**（`ANN_Output/`）：
- `flux_mlp_traced.pt` — TorchScript 模型（C++ LibTorch 直接加载）
- `flux_mlp_best.pt` — 最优 checkpoint
- `normalization_params.csv` — 归一化参数

### Step 6: 推理生成彻体力场

```bash
# 加载 ANN_Output/flux_mlp_traced.pt，在 ROTOR_FLUID 网格上做批量自动微分推理
ANN_Pre_Processing
```

**输出文件**：
- `constant/bodyForce` — 彻体力向量场（单位质量力，dimAcceleration）
- `constant/lambda` — 堵塞因子标量场（Step 4 已生成）

> 不使用神经网络时，可用两种传统插值方法替代 Step 5/6 直接生成 `constant/bodyForce`（均读 `openFOAM_Input_Force.csv`，无需训练、无需 LibTorch）：
> - **移动最小二乘（MLS）**：`MLS_Pre_Processing`，在邻域内做高斯加权局部多项式拟合，系数直接给出光滑函数值与解析梯度。
> - **RBF/IDW（Benneke 方法）**：`Benneke_Pre_Processing`，需自行提供 `optimization_results/best_flux_parameters.txt` 超参数。

### Step 7: 全环非定常计算

确认 `constant/bodyForce`、`constant/lambda` 已就位，`constant/fvModels` 会直接读取 `bodyForce` 彻体力场并把动量源项 `rho*bodyForce` 与叶片做功功率 `(rho*bodyForce)·U_blade` 注入求解器。然后：

```bash
# 串行
foamRun

# 或并行（参考 For_Nasa_Rotor_37_Body_Force.sh，内含 decomposePar + mpirun foamRun -parallel）
./For_Nasa_Rotor_37_Body_Force.sh
```

---
## 8. 依赖环境

| 软件 | 版本要求 |
|------|----------|
| **OpenFOAM** | v13 或更高 |
| **Python** | 3.8+ |
| **PyTorch** | 1.10+ |
| **LibTorch** | 与 PyTorch 版本匹配 |
| **NumPy** | 1.20+ |
| **Pandas** | 1.3+ |
| **Matplotlib** | 3.4+ |

### 8.1 LibTorch 安装

```bash
# 下载 LibTorch（CPU 版本）
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu.zip
unzip libtorch-shared-with-deps-1.10.0+cpu.zip

# 设置环境变量
export LibTorch_DIR=/path/to/libtorch
```

### 8.2 OpenFOAM 编译配置

在 `Make/options` 中添加：

```makefile
EXE_INC = \
    -I$(LIBTORCH_DIR)/include \
    -I$(LIBTORCH_DIR)/include/torch/csrc/api/include

EXE_LIBS = \
    -L$(LIBTORCH_DIR)/lib \
    -ltorch \
    -ltorch_cpu \
    -lc10
```

---

## 9. 神经网络架构

### 9.1 Random Fourier Features (RFF)

用于将低维坐标映射到高维特征空间，提高对高频信息的捕捉能力：

$$
\gamma(\mathbf{x}) = [\sin(\mathbf{B}\mathbf{x}), \cos(\mathbf{B}\mathbf{x})]
$$

其中 $\mathbf{B}$ 为随机投影矩阵，固定不参与训练。

### 9.2 FluxMLP 结构

```
Input: (r_norm, z_norm)  [2D]
   ↓
RFF Layer: sin/cos 映射  [128D]
   ↓
Hidden Layer 1: Linear + SiLU  [64D]
Hidden Layer 2: Linear + SiLU  [64D]
Hidden Layer 3: Linear + SiLU  [64D]
Hidden Layer 4: Linear + SiLU  [64D]
   ↓
Output Layer: Linear  [8D]
   ↓
Output: [ρ, λ·Trr, λ·Trt, λ·Trz, λ·Ttt, λ·Tzt, λ·Tzz, λ]
```

### 9.3 训练参数

| 参数 | 默认值 |
|------|--------|
| hidden_layers | 4 |
| hidden_dim | 64 |
| rff_dim | 128 |
| rff_sigma | 1.0 |
| epochs | 50000 |
| learning_rate | 1e-3 |
| weight_decay | 1e-3 |



## 10. 数据文件说明

### 10.1 输入数据格式 (CSV)

```csv
R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda
0.1234,1.225,101325.0,0.0,0.0,101325.0,0.0,101325.0,0.0567,0.95
...
```

**数据文件说明**：

| 文件 | 来源 | 用途 |
|------|------|------|
| `CFX_Output_Blockage.csv` | CFD 叶片表面（压力面/吸力面）坐标提取 | 计算堵塞因子子午分布 |
| `CFX_Output_Benneke_Flux.csv` | CFX 周向平均通量 | 神经网络训练输入（叶片平均通量） |
| `openFOAM_Input_Force.csv` | 金属堵塞计算程序输出 | 训练数据集 |
| `optimization_results/best_parameters_separate.txt` | RBF 插值优化结果 | 堵塞因子计算超参数 |

## 11. OpenFOAM 算例详细设置（NASA Rotor 37）

本章给出随仓库发布的 NASA Rotor 37 算例的完整数值设置，供论文撰写与算例复现使用。算例基于 **OpenFOAM v13**，采用自定义求解器 **ArisaSTALL** 在全环纯净流道网格上求解带金属堵塞修正的可压缩欧拉方程，彻体力场由 `fvModels` 以 coded 源项方式注入。

### 11.1 求解器特点（ArisaSTALL）

ArisaSTALL 继承自 OpenFOAM v13 的 `fluid` 基类，是一个**带金属堵塞因子 λ 修正的可压缩欧拉求解器**（详见 [ArisaSTALL/Solver_README.md](ArisaSTALL/Solver_README.md)）。其核心特点：

- **彻体力方法（BFM）专用**：网格为不含叶片的纯净流道，叶片效应由两部分等效——① 彻体力源项（动量/能量）；② 金属堵塞因子 λ（0 < λ ≤ 1，表示周向通流面积占比）对方程组的几何修正。
- **λ 修正贯穿全方程组**：
  - 动量预测：`λ·fvm::ddt(ρ,U) + fvm::div(λ_f·φ, U) == -fvc::grad(λ·p) + λ·源项`；
  - 压力修正（transonic + consistent PIMPLE）：压力方程含 `ψ·ddt(λ·p)`、`div(λ_f·φHbyA)` 与 `laplacian(λ·ρ·rAAtU_f, p)`，速度修正为 `U = HbyA - λ·rAAtU·grad(p)`，密度修正方程同样含 λ；
  - 能量预测（内能形式）：`λ·ddt(ρ,h_e) + div(λ_f·φ, h_e) + λ·ddt(ρ,K) + div(λ_f·φ, K)`，压力功项为 `div(λ_f·φ, p/ρ)`。
- **无湍流/无粘**：`momentumTransportPredictor/Corrector` 与 `thermophysicalTransportPredictor/Corrector` 均为空函数，求解器不求解任何湍流输运方程。
- λ 场从 `constant/lambda` 读取（`READ_IF_PRESENT`）；缺失时回退到 `fvSolution` 中的 `lambdaDefault` 值。
- 编译产物为 `$FOAM_USER_LIBBIN/libArisaSTALL.so`，由 `controlDict` 中 `solver ArisaSTALL;` 经 `foamRun` 加载。

### 11.2 彻体力源项注入（constant/fvModels）

两个 `coded` 源项均只作用于 `cellZone ROTOR_FLUID`（即转子叶片排区域）：

| 源项 | 作用方程 | 数学形式 | 说明 |
|------|----------|----------|------|
| `bodyForceSourceU` | 动量 U | `S_U = -ρ·bodyForce·V_cell` | 读取 `constant/bodyForce`（单位质量彻体力，dimAcceleration） |
| `bodyForceSourceE` | 能量 e | `S_e = -(ρ·bodyForce·U_blade)·V_cell` | 叶片对流体做功的功率源 |

能量源中叶片转速 **n = 17188.7 rpm**（ω = 2πn/60），叶片线速度按当地半径取周向分量 `U_blade = ω·r·e_θ`，彻体力功率为彻体力与叶片速度的内积。

### 11.3 时间步与运行控制（system/controlDict）

| 参数 | 值 | 说明 |
|------|----|------|
| `solver` | `ArisaSTALL` | 自定义求解器 |
| `endTime` | 0.02 s | 约 5.7 个转子转动周期（17188.7 rpm ≈ 286.5 Hz） |
| `deltaT` | 5×10⁻⁶ s | 固定时间步 |
| `adjustTimeStep` | `false` | 不自动调整步长（`maxCo 5.0`、`maxDeltaT 5×10⁻⁵` 仅为自适应预留） |
| `writeControl` | `adjustableRunTime` | 每 `writeInterval 1×10⁻⁴ s` 写一帧，`purgeWrite 6` 仅保留最近 6 帧 |
| `runTimeModifiable` | `true` | 运行中可修改字典 |

> 注意：随仓库发布的 `system/fvSchemes` 中 `ddtSchemes` 默认为 `steadyState`（伪时间/定常推进），`Euler` 方案被注释。做**非定常计算时需将默认方案改为 `Euler`**（一阶隐式），此时上述 `deltaT`、`endTime` 才具有真实物理时间意义。

### 11.4 空间离散格式（system/fvSchemes）

| 类别 | 设置 |
|------|------|
| 时间项 `ddtSchemes` | `default steadyState`（发布配置；非定常计算切换为 `Euler`） |
| 梯度 `gradSchemes` | `default Gauss linear`（二阶中心） |
| 对流项 `divSchemes` | `div(phi,U)`、`div(phid,p)`、`div(phi,e)`、`div(phi,(p\|rho))` 均为 `Gauss limitedLinear 0.7`；`div(phi,K)` 为 `Gauss limitedLinear 1`（二阶、有界，限制系数 0.7） |
| λ 加权对流项 | `div((interpolate(lambda)*phi), U/K/e/(p\|rho))` 同为 `Gauss limitedLinear 0.7`（λ 由单元内插至面心加权通量） |
| 粘性应力项 | `div(((rho*nuEff)*dev2(T(grad(U))))) Upwind`（无粘计算中 μ≈0，该项实际不贡献） |
| 拉普拉斯 `laplacianSchemes` | `default Gauss linear corrected`（面法向梯度显式非正交修正） |
| 面插值 `interpolationSchemes` | `default linear` |
| 面法向梯度 `snGradSchemes` | `default corrected` |

### 11.5 线性求解器与 PIMPLE 算法（system/fvSolution）

**线性求解器：**

| 场 | 求解器 | 光滑器/预条件 | tolerance | relTol |
|----|--------|---------------|-----------|--------|
| `p.*` | GAMG（几何代数多重网格，`cacheAgglomeration`，`nCellsInCoarsestLevel 20`） | DIC/Gauss-Seidel | 1×10⁻⁶ | 0 |
| `(U\|e\|h).*`、`rho.*` | smoothSolver | DILU/Gauss-Seidel | 1×10⁻⁶ ~ 1×10⁻⁸ | 0 ~ 0.01 |

**PIMPLE 算法：**

| 参数 | 值 | 说明 |
|------|----|------|
| `nOuterCorrectors` | 50 | 每个时间步外修正（伪时间迭代）次数 |
| `nCorrectors` | 2 | 压力方程内修正次数 |
| `nNonOrthogonalCorrectors` | 1 | 非正交修正次数 |
| `correctMeshPhi` | yes | |
| `consistent` | yes | consistent PIMPLE 形式（rAAtU 系数） |
| `transonic` | yes | 跨声速模式，压力方程含 `ψ·ddt(p)` 项 |
| 内迭代残差判据 | U/p/e = 1×10⁻⁶ | `residualControl` |
| 外修正残差判据 | U、p：tolerance 4×10⁻²，relTol 0.01 | `outerCorrectorResidualControl` |
| 松弛因子 | p = 0.4（`pFinal` = 1.0）；各方程 `".*"` = 0.4 | 定常/伪时间推进下的欠松弛 |

**场值限制（system/fvConstraints）：** `limitPressure`（minFactor 0.1，maxFactor 3）与 `limitTemperature`（200 K ≤ T ≤ 1000 K），用于抑制迭代初期非物理振荡。

### 11.6 湍流方法

**无粘（欧拉）设置**，不使用任何湍流模型：

- `constant/momentumTransport`：`simulationType laminar;`（层流框架下关闭湍流输运）；
- `constant/physicalProperties`：动力粘度 **μ = 1×10⁻¹⁰ Pa·s**（近零，等效无粘）；
- 固壁边界采用 **slip（滑移）** 条件，无壁面摩擦、无边界层求解。

彻体力模型本身已在源项中包含叶片排的耗散/转向效应，因此无需 RANS 闭合。

### 11.7 工质物性（constant/physicalProperties）

按**量热完全气体空气**处理，采用 OpenFOAM 热力学模板组合 `hePsiThermo / pureMixture / const / hConst / perfectGas / specie / sensibleInternalEnergy`：

| 属性 | 值 |
|------|----|
| 摩尔质量 | 28.9 g/mol |
| 定压比热 Cp | 1005 J/(kg·K)（常数） |
| 比热比 γ | 1.4（由 Cp、Pr 等推得） |
| 生成焓 hf | 0 |
| 动力粘度 μ | 1×10⁻¹⁰ Pa·s（无粘） |
| 普朗特数 Pr | 0.71 |

### 11.8 边界条件（0/ 目录）

| 边界 | p | U | T |
|------|-----|------|-----|
| INLET（进口） | `totalPressure`，p₀ = 101325 Pa | `pressureInletOutletVelocity` | `totalTemperature`，T₀ = 288.15 K，γ = 1.4 |
| OUTLET（出口） | `fixedMean`，平均静压 127892 Pa（反压，对应特定工况点） | `pressureInletOutletVelocity` | zeroGradient |
| 轮毂/机匣壁面（IN/OUT/ROTOR_HUB、_SHROUD） | zeroGradient | **slip**（滑移壁面） | hub/shroud 壁面 `fixedValue` 288.15 K |
| 转-静交界面 | cyclic（见 11.9） | cyclic | zeroGradient |

初始场：`p = 101325 Pa`，`U = (0 0 120) m/s`（轴向），`T = 288.15 K`。

### 11.9 转-静交界面处理

网格沿轴向分为进口段、转子段（ROTOR_FLUID）、出口段三个区域，区域间通过 **`nonConformalCyclic`（非共形循环/非匹配交界面）** 连接，共两对：

- `nonConformalCyclic_on_ROTOR_TO_IN` ↔ `nonConformalCyclic_on_IN_TO_ROTOR`（转子–进口段）；
- `nonConformalCyclic_on_ROTOR_TO_OUT` ↔ `nonConformalCyclic_on_OUT_TO_ROTOR`（转子–出口段）。

设置 `matchTolerance 0.0001`、`transformType none`（交界面两侧坐标一致，无需坐标变换），并配有 `nonConformalError` 补丁用于诊断非匹配面积。该交界面允许两侧网格节点不一一对应，通量通过面权重加权插值传递，适合彻体力模型中"叶片排独立建网、再拼接"的建模方式。由于叶片效应已全部由彻体力 + λ 等效，交界面两侧均在同一绝对坐标系下求解，**不使用 MRF/AMI 旋转参考系**。

### 11.10 网格、区域与并行

- **网格**：全环三维纯净流道六面体网格，共 **322,848 个单元**；
- **cellZone**：`ROTOR_FLUID`（彻体力作用区）、`IN_FLUID`、`OUT_FLUID` 三个区域；
- **堵塞因子场**：`constant/lambda` 为 322,848 个单元值的非均匀 volScalarField；
- **并行**：`system/decomposeParDict` 采用 **scotch** 方法自动分区，**8 核**；运行脚本 `For_Nasa_Rotor_37_Body_Force.sh` 自动执行 `decomposePar -constant`、`decomposePar -fields` 后 `mpirun -np 8 foamRun -parallel`。

### 11.11 运行监测（system/functions）

- `residuals`：各场残差输出；
- `patchFlowRate`：统计 OUTLET 补丁上 `phi` 的总流量（`operation sum`），用于监控流量收敛与工况点；
- `probes`：流道内多个测点坐标处的 `rho / U / p` 时历。

---

## 12. 参考文献

1. **Thollet, P., et al.** (2016). *Body-force modeling for aerodynamic analysis of air intake – fan interactions.* AIAA Journal.

2. **Benneke, J.** (2009). *A methodology for centrifugal compressor stability prediction.* ASME Turbo Expo.

3. **Xu, L.** (2003). *A computational fluid dynamics analysis of a three-dimensional transonic rotor.* NASA ROTOR 37.

---

## 13. 致谢

本项目基于以下开源平台开发：
- **OpenFOAM v13** — 开源 CFD 平台
- **PyTorch** — 深度学习框架
- **LibTorch** — PyTorch C++ API

感谢 OpenFOAM 基金会、PyTorch 团队提供优秀的开源工具。

---

## 14. License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 15. 联系方式
如有问题或建议，请通过705393357@qq.com反馈。
更新应该不会太更新了，本人拿来跑NASA ROTOR 37特性线误差在5%以内，但是超参数能调的太多，遂放弃，开源。欢迎来讨论。

## 16. 引用
老师让我整理成论文，非常烦，我应该会丢arXiv上，静候
