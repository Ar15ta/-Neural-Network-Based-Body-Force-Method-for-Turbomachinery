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
│   │   ├── ANN_Initial_Trainner.py   # Python 神经网络训练脚本
│   │   ├── ANN_Pre_Processing.C      # C++ 神经网络推理（LibTorch）
│   │   ├── Benneke_Pre_Processing.C  # 传统 IDW 插值方法（对比参考）
│   │   ├── BodyForceVisualizer.py    # 彻体力可视化工具
│   │   │
│   │   ├── optimization_results/    # RBF 插值超参数（计算堵塞因子子午分布）
│   │   │   └── best_parameters_separate.txt
│   │   │
│   │   ├── ANN_Output/               # 训练输出
│   │   │   ├── flux_mlp_traced.pt    # TorchScript 模型
│   │   │   ├── flux_mlp_weights.npz  # NPZ 权重
│   │   │   ├── weights_bin/          # 原始二进制权重
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
2. 导入到算例文件夹（`constant/polyMesh`）。本仓库提供了 `Rotor_37_Mesh.msh`，可用 `gmshToFoam` / `fluentMeshToFoam` 转换。
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

### Step 3: 编译两个预处理工具

`calculateBlockage` 和 `ANN_Pre_Processing` 是单个 `.C` 文件的独立工具，在算例文件夹下用 `wmake` 编译（每个工具需要自己的 `Make/files` 与 `Make/options`，`Make/options` 中指向你的 LibTorch 路径）：

```bash
# 编译堵塞因子计算工具（Make/files 中：SOURCE += calculateBlockage.C; EXE = $(FOAM_USER_APPBIN)/calculateBlockage）
wmake
# 编译神经网络推理工具（Make/files 中：SOURCE += ANN_Pre_Processing.C; EXE = $(FOAM_USER_APPBIN)/ANN_Pre_Processing）
wmake
```

> 提示：`make/options` 里的 LibTorch 路径（如 `/home/dyfluid/libtorch`）需改成你自己的路径。`ANN_Pre_Processing` 依赖 LibTorch；`calculateBlockage` 不依赖。

### Step 4: 计算堵塞因子并生成训练数据

```bash
# 读取 CFX_Output_Blockage.csv 与 CFX_Output_Benneke_Flux.csv
# 输出 constant/lambda（堵塞因子场）与 openFOAM_Input_Force.csv（带 λ 列的训练数据）
calculateBlockage
```

### Step 5: 训练神经网络（在主机上用 Python）

```bash
# 读取 openFOAM_Input_Force.csv，输出到 ANN_Output/
python ANN_Initial_Trainner.py
```

**输出文件**（`ANN_Output/`）：
- `flux_mlp_traced.pt` — TorchScript 模型（C++ LibTorch 直接加载）
- `flux_mlp_weights.npz` — NPZ 权重（可选）
- `weights_bin/` — 原始二进制权重（可选）
- `normalization_params.csv` — 归一化参数

### Step 6: 推理生成彻体力场

```bash
# 加载 ANN_Output/flux_mlp_traced.pt，在 ROTOR_FLUID 网格上做批量自动微分推理
ANN_Pre_Processing
```

**输出文件**：
- `constant/bodyForce` — 彻体力向量场（单位质量力，dimAcceleration）
- `constant/lambda` — 堵塞因子标量场（Step 4 已生成）

> 也可用传统 RBF+IDW 方法 `Benneke_Pre_Processing` 作对比参考（需补充超参数）。

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

## 11. 参考文献

1. **Thollet, P., et al.** (2016). *Body-force modeling for aerodynamic analysis of air intake – fan interactions.* AIAA Journal.

2. **Benneke, J.** (2009). *A methodology for centrifugal compressor stability prediction.* ASME Turbo Expo.

3. **Xu, L.** (2003). *A computational fluid dynamics analysis of a three-dimensional transonic rotor.* NASA ROTOR 37.

---

## 12. 致谢

本项目基于以下开源平台开发：
- **OpenFOAM v13** — 开源 CFD 平台
- **PyTorch** — 深度学习框架
- **LibTorch** — PyTorch C++ API

感谢 OpenFOAM 基金会、PyTorch 团队提供优秀的开源工具。

---

## 13. License

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

## 14. 联系方式
如有问题或建议，请通过705393357@qq.com反馈。
更新应该不会太更新了，本人拿来跑NASA ROTOR 37特性线误差在5%以内，但是超参数能调的太多，遂放弃，开源。欢迎来讨论。

## 15. 引用
老师让我整理成论文，非常烦，我应该会丢arXiv上，静候
