# Neural Network-Based Body Force Method for Turbomachinery

[中文文档](README.md) | **English**

---
<img width="2882" height="2603" alt="BFM_Procedure" src="https://github.com/user-attachments/assets/27d04fa3-2a7b-4362-a516-f9874742e8ee" />

<img width="3663" height="2316" alt="BFM_MESH" src="https://github.com/user-attachments/assets/2aa1b2cb-21f7-413f-94b0-289c3b238e74" />




## 1. Introduction

This project proposes a **brand-new body force computation pipeline**, introducing — for the first time — neural networks and automatic differentiation into the field of turbomachinery body force modeling. It is validated on the NASA Rotor 37 case. The NASA Rotor 37 data used here come from https://github.com/Yashay03/Axial-Compressor-Rotor-37

**Using an AI assistant to set up the environment, run, and understand this project is highly recommended!!!**

**Core idea**:
1. Extract meridional-plane flux data from a steady single-passage CFX computation.
2. Use a neural network to fit the λ (metal blockage coefficient) × flux ($\rho UU$) field, obtaining a continuously differentiable expression.
3. Compute derivatives via automatic differentiation and substitute them into the momentum equation to recover the body force distribution.
4. Inject the body force into a full-annulus OpenFOAM mesh and run an unsteady 3D computation on a clean (blade-free) flow passage.

---

## 2. Background

### 2.1 Limitations of traditional methods

| Method | Pros | Cons |
|--------|------|------|
| **Full 3D CFD** | High accuracy, captures details | Extremely expensive; hard to run multi-condition / unsteady analyses |
| **Low-order models** | Very fast | Cannot capture 3D stall phenomena |

### 2.2 Position of this method

The **Body Force Method (BFM)** is a compromise between the two:
- Uses a low-order model (single-passage CFD) to generate the body force source terms.
- Runs full-annulus unsteady computations in a clean 3D passage.
- Balances computational efficiency with the ability to capture 3D flow.
- It can be used for coarse unsteady computations at any stable operating point. To improve reliability near stall or across operating conditions, you **must** extract body forces from multiple conditions yourself and build an online lookup table.

---

## 3. Project structure

The project contains two core modules:

```
Arisa_Benneke_Method/
│
├── Bodyforce_Method/                 # Body force generation module
│   ├── NASA_ROTOR_37/                # NASA ROTOR 37 example case
│   │   ├── ANN_TrainOneMLP.py       # Python neural network training script
│   │   ├── ANN_Pre_Processing.C      # C++ neural network inference (LibTorch)
│   │   ├── Benneke_Pre_Processing.C  # Traditional IDW interpolation (reference)
│   │   ├── BodyForceVisualizer.py    # Body force visualization tool
│   │   │
│   │   ├── optimization_results/     # RBF interpolation hyper-parameters (meridional blockage factor)
│   │   │   └── best_parameters_separate.txt
│   │   │
│   │   ├── ANN_Output/               # Training outputs
│   │   │   ├── flux_mlp_traced.pt    # TorchScript model (loaded by ANN_Pre_Processing)
│   │   │   ├── flux_mlp_best.pt      # Best checkpoint
│   │   │   └── normalization_params.csv
│   │   │
│   │   ├── CFX_Output_Benneke_Flux.csv  # Flux data exported from CFX
│   │   ├── constant/                 # OpenFOAM mesh and physical properties
│   │   └── system/                   # Solver configuration
│   │
│   └── Initializer/                  # Full-annulus initialization case
│
├── ArisaSTALL/                       # 3D full-annulus Euler solver
│   ├── ArisaSTALL.C                  # Main solver
│   ├── ArisaSTALL.H                  # Header
│   ├── momentumPredictor.C           # Momentum predictor
│   ├── thermophysicalPredictor.C     # Energy predictor
│   ├── correctPressure.C             # Pressure correction
│   ├── Make/                         # Build configuration
│   └── Solver_README.md              # Detailed solver documentation
│
└── README.md                         # This file
```

---

## 4. Key innovations

### 4.1 First use of neural networks + automatic differentiation

Traditional body force methods (e.g. the Benneke method) use **IDW interpolation** or **polynomial fitting** to process discrete flux data, which has the following problems:
- Interpolation results are not smooth; derivative computation has large errors.
- Requires manual tuning; poor generalization.
- Integration methods need meridional mesh information; weak portability.

**Advantages of this method**:
- Neural networks naturally provide **smooth, differentiable** fits.
- Automatic differentiation computes derivatives **exactly**, with no manual derivation.
- Supports **arbitrary turbomachinery**.

### 4.2 Network design: directly learning λ×flux
Only scatter data of the fluxes is needed (meridional coordinates + 6 flux values + metal blockage distribution + density).

The network outputs **λ×flux** (rather than λ and flux separately), avoiding a multiplication at inference time and reducing error accumulation:

```
Network output: [ρ, λ·Trr, λ·Trt, λ·Trz, λ·Ttt, λ·Tzt, λ·Tzz, λ]
         ↓ automatic differentiation
Body force:    [f_r, f_θ, f_z]
```

### 4.3 Batch inference

The C++ inference uses **batch matrix operations**, processing all mesh cells at once to produce the body force field for the operating condition, which is then invoked at runtime by OpenFOAM's fvModels.

### 4.4 Euler solver with metal blockage correction

The accompanying compressible **Euler** solver **ArisaSTALL** inherits from OpenFOAM v13's `fluid` solver and supports:
- Momentum/energy equation correction with the blockage factor λ.
- 3D full-annulus unsteady computation.
- Seamless coupling with the neural-network body force.

---

## 5. Limitations

- Accuracy depends on the **quality of the single-passage CFD**.
- The full-annulus mesh distribution must be **spatially consistent** with the single-passage data.
- Training parameters (network structure, learning rate, etc.) require **tuning**.

---

## 6. Mathematical principles

### 6.1 Body force equation

Starting from the momentum equation, the body force is defined as:

$$
\lambda \rho \mathbf{f} = \frac{1}{r} \frac{\partial (\lambda r \mathbf{T}_r)}{\partial r} + \frac{\partial (\lambda \mathbf{T}_z)}{\partial z} - \frac{\lambda \mathbf{T}_\theta}{r}
$$

where $\mathbf{T}$ is the flux tensor:

$$
\mathbf{T} = \begin{bmatrix} T_{rr} & T_{r\theta} & T_{rz} \\ T_{\theta r} & T_{\theta\theta} & T_{\theta z} \\ T_{zr} & T_{z\theta} & T_{zz} \end{bmatrix}
$$

Components:
- $T_{rr} = \rho V_r^2 + p$
- $T_{r\theta} = \rho V_r V_\theta$
- $T_{rz} = \rho V_r V_z$
- $T_{\theta\theta} = \rho V_\theta^2 + p$
- $T_{\theta z} = \rho V_\theta V_z$
- $T_{zz} = \rho V_z^2 + p$

### 6.2 Automatic differentiation chain rule

The network takes normalized coordinates $(r_{norm}, z_{norm})$; physical derivatives are:

$$
\frac{\partial}{\partial r_{phys}} = \frac{\partial}{\partial r_{norm}} \cdot \frac{1}{r_{max} - r_{min}}
$$

$$
\frac{\partial}{\partial z_{phys}} = \frac{\partial}{\partial z_{norm}} \cdot \frac{1}{z_{max} - z_{min}}
$$

### 6.3 Blockage factor correction

The blockage factor λ denotes the fluid volume fraction (0 < λ ≤ 1), used to model the effect of metal blades on the flow.

$\lambda$ is defined as $N\frac{\theta_{PS}-\theta_{SS}}{2\pi}$, where $N$ is the number of blades and $\theta$ is the circumferential coordinate of the blade at a given $r$, $z$ (the pressure side and suction side of two adjacent blades, respectively).

The corrected momentum equation:

$$
\frac{\partial (\lambda \rho \mathbf{U})}{\partial t} + \nabla \cdot (\lambda \rho \mathbf{U} \mathbf{U}) = -\nabla (\lambda p) + \lambda \rho \mathbf{g} + \lambda \mathbf{f}
$$

The corrected energy equation:

$$
\frac{\partial (\lambda \rho e)}{\partial t} + \nabla \cdot (\lambda \rho e \mathbf{U}) = -\nabla \cdot (\lambda p \mathbf{U}) + \lambda S_e
$$

---

## 7. Workflow

> It is recommended to run the C++ tools in a Linux virtual machine (OpenFOAM v13) sharing a folder with the host, and to run the Python training script on the host.
>
> **This repository ships with a coarse-mesh NASA Rotor 37 case (`constant/polyMesh`, `0/`, etc.), including a trained network and the body force field. You can run it directly with `foamRun` in Step 7.** To model your own turbomachine, start from Step 1.

### Step 0: Prepare the mesh

1. Build a **clean full-annulus passage mesh** (blades are not part of the geometry; they are represented by the body force + blockage factor).
2. Import it into the case folder (`constant/polyMesh`). **This repository already ships the converted `constant/polyMesh`**, so no mesh file is needed to run; for your own mesh, use `gmshToFoam` / `fluentMeshToFoam` to convert a `.msh`/`.cas` file into `constant/polyMesh`.
3. **Make sure to glue the interfaces**: use `mergePairs` / `stitchMesh`, or glue the partitioned regions into a contiguous mesh during import, and use `createPatch` to set up the inlet, outlet, and wall boundaries.
4. Define the `ROTOR_FLUID` cellZone — the body force acts only on this region.

### Step 1: Export meridional data from single-passage CFD

Export the following meridional (circumferentially averaged) data from a steady single-passage CFD run (e.g. CFX):

- Blade surface (pressure side / suction side) coordinates → `CFX_Output_Blockage.csv` (used to compute the meridional distribution of the blockage factor λ).
- Meridional coordinates (R, Z), density ρ, flux components Trr/Trt/Trz/Ttt/Tzt/Tzz, blockage factor λ → `CFX_Output_Benneke_Flux.csv`.

### Step 2: Compile the ArisaSTALL solver

`ArisaSTALL` is a compressible Euler solver inheriting from OpenFOAM v13's `fluid` solver. After `wmake` it produces `libArisaSTALL.so`, loaded by `foamRun` via `solver ArisaSTALL;` in controlDict.

```bash
cd ArisaSTALL
wmake          # requires OpenFOAM v13
cd ..
```

### Step 3: Compile the pre-processing tools

The body force field can be generated by three methods, each a single-`.C`-file tool compiled with `wmake` in the case folder (they share one `Make/files` and `Make/options`):

| Tool | Method | Dependency |
|------|--------|------------|
| `ANN_Pre_Processing.C` | Neural network + automatic differentiation (**recommended, the project's main method**) | LibTorch |
| `MLS_Pre_Processing.C` | Moving Least Squares local weighted polynomial fit | none |
| `Benneke_Pre_Processing.C` | Traditional RBF/IDW interpolation (reference comparison) | none |
| `calculateBlockage.C` | Blockage factor λ and training data generation | none |

`wmake` can only build one executable at a time: open `Make/files`, uncomment the `SOURCE += xxx.C` / `EXE = ...` pair for the tool you want (comment the others), then run `wmake`. The repository's `make/files` already contains all configurations and activates `MLS_Pre_Processing.C` by default.

> Note: the repository folder is named lowercase `make`, while OpenFOAM reads `Make/files` by default; if `wmake` cannot find the files, rename the folder to `Make` (`mv make Make`) before compiling.

```bash
# e.g. build the neural network tool (activate the ANN_Pre_Processing.C pair in Make/files)
wmake
# e.g. build the Moving Least Squares tool (activate the MLS_Pre_Processing.C pair in Make/files)
wmake
# e.g. build the blockage factor tool (activate the calculateBlockage.C pair in Make/files)
wmake
```

> Note: the LibTorch path in `Make/options` (e.g. `/home/dyfluid/libtorch`) must be changed to your own path. Only `ANN_Pre_Processing` depends on LibTorch; `MLS_Pre_Processing`, `Benneke_Pre_Processing`, and `calculateBlockage` do not.

### Step 4: Compute the blockage factor and generate training data

```bash
# Reads CFX_Output_Blockage.csv and CFX_Output_Benneke_Flux.csv
# Outputs constant/lambda (blockage factor field) and openFOAM_Input_Force.csv (training data with a λ column)
calculateBlockage
```

### Step 5: Train the neural network (in Python on the host)

```bash
# Reads openFOAM_Input_Force.csv, outputs to ANN_Output/
python ANN_TrainOneMLP.py
```

**Output files** (`ANN_Output/`):
- `flux_mlp_traced.pt` — TorchScript model (loaded directly by C++ LibTorch).
- `flux_mlp_best.pt` — best checkpoint.
- `normalization_params.csv` — normalization parameters.

### Step 6: Run inference to generate the body force field

```bash
# Loads ANN_Output/flux_mlp_traced.pt and runs batch automatic-differentiation inference on the ROTOR_FLUID mesh
ANN_Pre_Processing
```

**Output files**:
- `constant/bodyForce` — body force vector field (force per unit mass, dimAcceleration).
- `constant/lambda` — blockage factor scalar field (generated in Step 4).

> Without using the neural network, two traditional interpolation methods can replace Steps 5/6 and directly produce `constant/bodyForce` (both read `openFOAM_Input_Force.csv`; no training, no LibTorch):
> - **Moving Least Squares (MLS)**: `MLS_Pre_Processing`, a Gaussian-weighted local polynomial fit in the neighborhood whose coefficients directly give a smooth value and analytical gradient.
> - **RBF/IDW (Benneke method)**: `Benneke_Pre_Processing`, requires you to supply `optimization_results/best_flux_parameters.txt` hyper-parameters.

### Step 7: Full-annulus unsteady computation

Make sure `constant/bodyForce` and `constant/lambda` are in place. `constant/fvModels` directly reads the `bodyForce` field and injects the momentum source `rho*bodyForce` and the blade power `(rho*bodyForce)·U_blade` into the solver. Then:

```bash
# Serial
foamRun

# Or parallel (see For_Nasa_Rotor_37_Body_Force.sh, which includes decomposePar + mpirun foamRun -parallel)
./For_Nasa_Rotor_37_Body_Force.sh
```

---
## 8. Dependencies

| Software | Version |
|----------|---------|
| **OpenFOAM** | v13 or later |
| **Python** | 3.8+ |
| **PyTorch** | 1.10+ |
| **LibTorch** | matching the PyTorch version |
| **NumPy** | 1.20+ |
| **Pandas** | 1.3+ |
| **Matplotlib** | 3.4+ |

### 8.1 Installing LibTorch

```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu.zip
unzip libtorch-shared-with-deps-1.10.0+cpu.zip

# Set the environment variable
export LibTorch_DIR=/path/to/libtorch
```

### 8.2 OpenFOAM build configuration

Add the following to `Make/options`:

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

## 9. Neural network architecture

### 9.1 Random Fourier Features (RFF)

Maps low-dimensional coordinates into a high-dimensional feature space to improve the capture of high-frequency information:

$$
\gamma(\mathbf{x}) = [\sin(\mathbf{B}\mathbf{x}), \cos(\mathbf{B}\mathbf{x})]
$$

where $\mathbf{B}$ is a random projection matrix, fixed and not trained.

### 9.2 FluxMLP structure

```
Input: (r_norm, z_norm)  [2D]
   ↓
RFF Layer: sin/cos mapping  [128D]
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

### 9.3 Training parameters

| Parameter | Default |
|-----------|---------|
| hidden_layers | 4 |
| hidden_dim | 64 |
| rff_dim | 128 |
| rff_sigma | 1.0 |
| epochs | 50000 |
| learning_rate | 1e-3 |
| weight_decay | 1e-3 |



## 10. Data files

### 10.1 Input data format (CSV)

```csv
R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda
0.1234,1.225,101325.0,0.0,0.0,101325.0,0.0,101325.0,0.0567,0.95
...
```

**Data file descriptions**:

| File | Source | Purpose |
|------|--------|---------|
| `CFX_Output_Blockage.csv` | CFD blade surface (PS/SS) coordinate extraction | Compute meridional blockage factor distribution |
| `CFX_Output_Benneke_Flux.csv` | CFX circumferentially averaged fluxes | Neural network training input (blade-averaged fluxes) |
| `openFOAM_Input_Force.csv` | Output of the metal blockage computation program | Training dataset |
| `optimization_results/best_parameters_separate.txt` | RBF interpolation optimization result | Hyper-parameters for blockage factor computation |

## 11. References

1. **Thollet, P., et al.** (2016). *Body-force modeling for aerodynamic analysis of air intake – fan interactions.* AIAA Journal.

2. **Benneke, J.** (2009). *A methodology for centrifugal compressor stability prediction.* ASME Turbo Expo.

3. **Xu, L.** (2003). *A computational fluid dynamics analysis of a three-dimensional transonic rotor.* NASA ROTOR 37.

---

## 12. Acknowledgements

This project is built on the following open-source platforms:
- **OpenFOAM v13** — open-source CFD platform.
- **PyTorch** — deep learning framework.
- **LibTorch** — PyTorch C++ API.

Thanks to the OpenFOAM Foundation and the PyTorch team for their excellent open-source tools.

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

## 14. Contact
For questions or suggestions, please contact 705393357@qq.com.
There probably won't be many further updates. For the NASA ROTOR 37 characteristic line the error is within 5%, but there are too many tunable hyper-parameters, so I gave up and open-sourced it. Discussions are welcome.

## 15. Citation
My advisor asked me to write it up as a paper, which is annoying. I'll probably put it on arXiv — stay tuned.
