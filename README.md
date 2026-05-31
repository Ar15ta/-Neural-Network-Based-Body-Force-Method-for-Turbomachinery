# 基于神经网络和自动微分的叶轮机械彻体力方法

## Neural Network-Based Body Force Method for Turbomachinery

---

## 1. 项目简介

本项目提出了一种**全新的彻体力计算流程**，首次将神经网络与自动微分技术引入叶轮机械彻体力建模领域，以 NASA ROTOR 37 离心压缩机为例进行验证。

**推荐使用AI协助搭建环境，运行和理解本项目内容！！！**

**核心思路**：
1. 从 CFX 单通道定常计算提取子午面通量数据
2. 使用神经网络拟合 λ×通量 场，获得连续可微的表达式
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
│   └── Solver_README.md                     # 求解器详细说明
│
└── README.md                         # 本文件
```

---

## 4. 核心创新点

### 4.1 首次引入神经网络 + 自动微分

传统彻体力方法（如 Benneke 方法）使用 **IDW 插值** 或 **多项式拟合** 来处理离散通量数据，存在以下问题：
- 插值结果不光滑，导数计算误差大
- 需要手工调参，泛化能力差

**本方法的优势**：
- 神经网络天然提供**光滑、可微**的拟合结果
- 自动微分**精确计算导数**，无需手工推导
- 支持**任意叶轮机械**，只需提供通量数据即可

### 4.2 Method B：直接学习 λ×通量

网络输出 **λ×通量**（而非单独的 λ 和通量），避免了推理时的乘积运算，减少了误差累积：

```
网络输出: [ρ, λ·Trr, λ·Trt, λ·Trz, λ·Ttt, λ·Tzt, λ·Tzz, λ]
         ↓ 自动微分
彻体力:   [f_r, f_θ, f_z]
```

### 4.3 批量推理优化

C++ 推理使用 **批量矩阵运算**，一次处理所有网格单元，比逐单元计算快 **10-100 倍**。

### 4.4 支持金属堵塞修正的欧拉求解器

配套求解器 **ArisaSTALL** 继承自 OpenFOAM v13 的 `fluid` 求解器，支持：
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
建议在虚拟机用Linux环境与主机共享文件夹操作，这样可以主机运行Python而虚拟机用openFOAM环境运行下面的工具。
### Step 1: 从 CFX 导出通量数据

在 CFX 单通道定常计算完成后，导出以下数据：
- 子午面坐标 (R, Z)
- 密度 ρ
- 通量分量: Trr, Trt, Trz, Ttt, Tzt, Tzz
- 堵塞因子 λ

保存为 CSV 格式，例如 `CFX_Output_Benneke_Flux.csv`

### Step 2: 编译 OpenFOAM 工具
包括求解器和其他工具。具体地，见Make文件夹。
```bash
wmake
```
### Step 3: 数据准备\前处理
```bash
# 生成堵塞因子场和用于训练神经网络的.csv文件
calculateBlockage
```
### Step 4: 训练神经网络

```python
# 运行ANN_Initial_Trainner.py，生成神经网络模型。
python ANN_Initial_Trainner.py
```

**输出文件**：
- `flux_mlp_traced.pt` — TorchScript 模型（C++ 加载）
- `flux_mlp_weights.npz` — NPZ 权重（可选）
- `weights_bin/` — 原始二进制权重（可选）
- `normalization_params.csv` — 归一化参数
### Step 5: 生成彻体力场
```bash
# 使用 ANN 方法（推荐）
ANN_Pre_Processing

# 或使用传统 RBF+IDW方法（对比参考，需要补充超参数）
Benneke_Pre_Processing
```

**输出文件**：
- `constant/bodyForce` — 彻体力向量场
- `constant/lambda` — 堵塞因子标量场

### Step 6: 全环非定常计算
记得将生成的 `bodyForce` 和 `lambda` 拷贝到constant目录下，然后使用 ArisaSTALL 求解器进行计算：
```bash
foamRun
```
该求解器将用constant文件夹下的fvModels调用彻体力场计算。

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

## 15. 引用
