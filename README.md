# 基于神经网络与自动微分的叶轮机械彻体力工具

## Neural-Network-Based Body-Force Method for Turbomachinery

[English](README_EN.md)

这是一个面向研究和二次开发的彻体力（Body Force Method, BFM）原型。它从单通道 CFD 的周向平均通量出发，用神经网络构造连续通量场，再通过自动微分计算梯度并反演体积力。仓库附带 NASA Rotor 37 的 OpenFOAM v13 示例。

> 当前状态：代码可以作为单工况重构与数值实验的起点，但不是经过多构型、多工况验证的通用压气机模型。跨工况、近失速和畸变计算需要额外的标定与验证。

## 方法概览

输入为子午面坐标、密度、金属堵塞系数和六个动量通量分量：

```text
(r, z, rho, lambda, Trr, Trt, Trz, Ttt, Tzt, Tzz)
        -> neural flux reconstruction
        -> automatic differentiation
        -> (fr, ftheta, fz)
        -> OpenFOAM fvModels source
```

反演使用柱坐标下的守恒形式：

$$
\lambda\rho\mathbf f=
\frac{1}{r}\frac{\partial(\lambda r\mathbf T_r)}{\partial r}
+\frac{\partial(\lambda\mathbf T_z)}{\partial z}
-\frac{\lambda\mathbf T_\theta}{r}.
$$

神经网络直接拟合 $\lambda\mathbf T$，以便用自动微分计算空间导数。随仓库保留的 MLS 和 RBF/IDW 实现可用于对比。

## 主要文件

```text
ANN_TrainOneMLP.py          训练通量网络
ANN_Pre_Processing.C       LibTorch 推理与自动微分
MLS_Pre_Processing.C       移动最小二乘重构
Benneke_Pre_Processing.C   RBF/IDW 重构
calculateBlockage.C        生成 lambda 和训练数据
ArisaSTALL/                OpenFOAM v13 堵塞修正欧拉求解器
0/, constant/, system/     可运行的 Rotor 37 示例
ANN_Output/                示例网络与归一化参数
```

示例输入数据来自 [Yashay03/Axial-Compressor-Rotor-37](https://github.com/Yashay03/Axial-Compressor-Rotor-37)。

## 环境

- OpenFOAM v13
- Python 3.8+
- PyTorch、NumPy、Pandas
- 与 PyTorch 版本匹配的 LibTorch

`make/options` 中的 LibTorch 路径目前需要按本机安装位置修改。

## 最短运行路径

仓库已经包含 `constant/bodyForce`、`constant/lambda`、网格和初始场。若只想运行示例：

```bash
source /opt/openfoam13/etc/bashrc

cd ArisaSTALL
wmake
cd ..

foamRun
```

`system/controlDict` 通过 `libArisaSTALL.so` 加载求解器。建议先以串行、小终止时间运行，并检查质量流量、连续性误差和场值是否有限。

## 从 CFD 数据重新生成体积力

1. 准备两个 CSV：
   - `CFX_Output_Blockage.csv`：叶片压力面和吸力面坐标；
   - `CFX_Output_Benneke_Flux.csv`：周向平均子午面通量。
2. 在 `make/files` 中只启用 `calculateBlockage.C`，编译并运行 `calculateBlockage`。
3. 运行 `python ANN_TrainOneMLP.py`。
4. 在 `make/files` 中只启用 `ANN_Pre_Processing.C`，编译并运行 `ANN_Pre_Processing`。
5. 检查生成的 `constant/lambda` 和 `constant/bodyForce` 后运行 `foamRun`。

在当前目录结构下可使用 `wmake make` 指定小写的 `make/` 配置目录；也可以将其重命名为 OpenFOAM 常用的 `Make/` 后执行 `wmake`。

输入通量 CSV 的最小表头为：

```csv
R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda
```

其中 `Lamda` 是历史拼写，为兼容现有代码暂时保留。

## 已知局限

- 体积力由给定 CFD 工况反演，因此同工况复现不等于跨工况预测能力。
- 结果对训练超参数、边界外插、堵塞定义和源项离散敏感。
- 当前示例没有验证失速起始、进口畸变或跨压气机构型的泛化能力。
- 网格、区域名称和若干路径仍带有 Rotor 37 案例假设。
- 仓库中的示例模型和体积力场用于复现工作流，不应视为高精度基准数据。

## 欢迎贡献

优先欢迎以下改进：

- 可重复的一键构建和最小回归测试；
- 去除硬编码路径，统一 `Make/` 和运行脚本；
- 检查输入列、单位、网格区域和 NaN/Inf 的工具；
- 体积力积分、扭矩和轴向力闭合测试；
- 多工况标定、留一工况验证及不确定性评估；
- 更小、许可清晰、可公开分发的测试数据集。

提交问题时，请附上 OpenFOAM/PyTorch/LibTorch 版本、运行命令、输入数据表头和最短错误日志；不要上传完整计算时刻目录。

## 许可说明

`ArisaSTALL/` 包含 OpenFOAM 派生代码并保留其 GPL 许可声明。仓库其余代码的统一许可仍需整理；再分发或贡献前请同时检查各文件头和上游许可。
