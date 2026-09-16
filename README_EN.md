# Neural-Network-Based Body-Force Method for Turbomachinery

[中文](README.md)

This repository is a research prototype for reconstructing turbomachinery body-force fields from pitchwise-averaged single-passage CFD data. A neural network represents a continuous momentum-flux field, automatic differentiation evaluates its spatial derivatives, and an OpenFOAM source term applies the reconstructed force. A NASA Rotor 37 case for OpenFOAM v13 is included.

> Status: the code is a starting point for single-operating-point reconstruction and numerical experiments. It is not a generally validated compressor model. Off-design, near-stall, and distorted-inflow applications require additional calibration and validation.

## Pipeline

```text
(r, z, rho, lambda, Trr, Trt, Trz, Ttt, Tzt, Tzz)
        -> neural flux reconstruction
        -> automatic differentiation
        -> (fr, ftheta, fz)
        -> OpenFOAM fvModels source
```

The cylindrical conservative relation used by the reconstruction is

$$
\lambda\rho\mathbf f=
\frac{1}{r}\frac{\partial(\lambda r\mathbf T_r)}{\partial r}
+\frac{\partial(\lambda\mathbf T_z)}{\partial z}
-\frac{\lambda\mathbf T_\theta}{r}.
$$

The network jointly fits $\rho$, $\lambda$, and the six raw flux components $\mathbf T$. During inversion, $\lambda\mathbf T$ is formed in the computation graph and differentiated by automatic differentiation. MLS and RBF/IDW implementations are retained for comparison.

## Main files

```text
ANN_TrainOneMLP.py          flux-network training
ANN_Pre_Processing.C       LibTorch inference and automatic differentiation
MLS_Pre_Processing.C       moving-least-squares reconstruction
Benneke_Pre_Processing.C   RBF/IDW reconstruction
calculateBlockage.C        lambda field and training-data generation
ArisaSTALL/                OpenFOAM v13 blockage-aware Euler solver
0/, constant/, system/     runnable Rotor 37 example
ANN_Output/                example network and normalization parameters
```

The example input is based on [Yashay03/Axial-Compressor-Rotor-37](https://github.com/Yashay03/Axial-Compressor-Rotor-37).

## Requirements

- OpenFOAM v13
- Python 3.8+
- PyTorch, NumPy, and Pandas
- LibTorch matching the installed PyTorch version

Update the LibTorch paths in `make/options` for your installation.

## Minimal run

The repository includes a mesh, initial fields, `constant/bodyForce`, and `constant/lambda`:

```bash
source /opt/openfoam13/etc/bashrc

cd ArisaSTALL
wmake
cd ..

foamRun
```

`system/controlDict` loads `libArisaSTALL.so`. Start with a short serial run and verify mass flow, continuity errors, and finite field values.

## Rebuilding the body-force field

1. Prepare `CFX_Output_Blockage.csv` from the blade surfaces and `CFX_Output_Benneke_Flux.csv` from pitchwise-averaged meridional fluxes.
2. Enable only `calculateBlockage.C` in `make/files`, compile it, and run `calculateBlockage`.
3. Run `python ANN_TrainOneMLP.py`.
4. Enable only `ANN_Pre_Processing.C` in `make/files`, compile it, and run `ANN_Pre_Processing`.
5. Inspect `constant/lambda` and `constant/bodyForce`, then run `foamRun`.

Use `wmake make` with the current lowercase directory, or rename it to the conventional `Make/` and run `wmake`.

Minimum flux CSV header:

```csv
R,Rho,Trr,Trt,Trz,Ttt,Tzt,Tzz,Z,Lamda
```

`Lamda` is a historical spelling retained for compatibility.

## Known limitations

- A force reconstructed from one CFD operating point is not, by itself, an off-design predictive model.
- Results are sensitive to training hyperparameters, extrapolation, blockage definitions, and source discretization.
- Stall inception, inlet distortion, and cross-configuration generalization are not validated.
- Several paths, zone names, and assumptions remain Rotor-37-specific.
- Bundled models and fields demonstrate the workflow and are not reference-quality benchmark data.

## Contributing

Useful contributions include:

- reproducible build scripts and a minimal regression test;
- removal of hard-coded paths and consistent `Make/` layouts;
- validation of input columns, units, zones, and NaN/Inf values;
- integrated force, torque, and axial-load closure tests;
- multi-operating-point calibration and leave-one-point-out evaluation;
- a smaller, clearly licensed public test dataset.

When opening an issue, include OpenFOAM/PyTorch/LibTorch versions, the exact command, the input header, and the shortest relevant log. Do not upload full time directories.

## Licensing note

`ArisaSTALL/` contains OpenFOAM-derived code and retains its GPL notices. Repository-wide licensing for the remaining original files still needs to be normalized; consult individual file headers and upstream licenses before redistribution.
