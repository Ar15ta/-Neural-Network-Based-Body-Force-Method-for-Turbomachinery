#!/bin/bash
# OpenFOAM Case Runner with Monitoring for NASA_Rotor_37_Body_Force



# 求解控制区域
solver="ArisaSTALL"  # 选择求解器
num_procs=8  # 并行进程数 - 根据虚拟机分配的核心数调整，建议不要超过分配的物理核心数

# 求解相关-----------------------------------------------------------------
echo "=== Operator:Arista ==="
echo "OpenFOAM Case Runner"
echo "Case directory: $(pwd)"
echo "Solver: $solver"
echo "Number of processes: $num_procs"
echo "Started at: $(date)"
# 确认使用的OpenFOAM版本
echo "Using OpenFOAM version: $WM_PROJECT_VERSION"

# 预处理检查
#echo "Running checkMesh..."
#checkMesh

# 启动求解器
echo "Starting solver: $solver"
# 串行
# $solver &
# 并行
if [ -d "processor0" ]; then
    echo "Using existing decomposed mesh..."
    # 即使网格已经分解，也要重新分解初始场（可能更新了自定义场）
    echo "Re-decomposing initial fields in constant directory..."
    for field in constant/*;
    do
        # 只处理文件，跳过目录和网格文件
        if [ -f "$field" ] && [ "$(basename "$field")" != "polyMesh" ] && [ "$(basename "$field")" != "blockMeshDict" ] && [ "$(basename "$field")" != "decomposeParDict" ]; then
            field_name=$(basename "$field")
            echo "  Decomposing field: $field_name"
            decomposePar -fields -constant $field_name
        fi
    done
else
    echo "Decomposing mesh..."
    decomposePar -constant
    
    # 分解constant目录下的初始场（包括自定义彻体力场）
    echo "Decomposing initial fields in constant directory..."
    for field in constant/*;
    do
        # 只处理文件，跳过目录和网格文件
        if [ -f "$field" ] && [ "$(basename "$field")" != "polyMesh" ] && [ "$(basename "$field")" != "blockMeshDict" ] && [ "$(basename "$field")" != "decomposeParDict" ]; then
            field_name=$(basename "$field")
            echo "  Decomposing field: $field_name"
            decomposePar -fields -constant $field_name
        fi
    done
fi

# 分解完成后，无论哪种情况都启动求解器
echo "Starting parallel solver with $num_procs processes..."
echo "VMware tip: If parallel efficiency is low, try reducing num_procs to match allocated physical cores"
echo "Checking available cores..."
# 打印CPU信息供参考
if [ -f /proc/cpuinfo ]; then
    echo "Available CPUs: $(grep -c ^processor /proc/cpuinfo)"
fi
# 使用核心绑定优化，在虚拟机上提高并行效率
# --bind-to core: 绑定进程到不同核心，避免迁移开销
# --map-by core: 按核心映射
# 前台直接运行，输出到终端，减少IO
mpirun -np $num_procs --bind-to core:overload-allowed --map-by core --report-bindings foamRun -parallel