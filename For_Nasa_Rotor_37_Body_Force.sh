#!/usr/bin/env bash

set -euo pipefail

# Run after sourcing the OpenFOAM v13 environment.
if ! command -v wmake >/dev/null 2>&1 || ! command -v foamRun >/dev/null 2>&1; then
    echo "OpenFOAM is not initialized. Source OpenFOAM v13 etc/bashrc first." >&2
    exit 2
fi

case_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$case_dir"

build_jobs="${BUILD_JOBS:-2}"
(cd ArisaSTALL && wmake -j "$build_jobs")

for required in 0 constant/polyMesh constant/bodyForce constant/lambda system/controlDict; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required case item: $required" >&2
        exit 2
    fi
done

echo "Running NASA Rotor 37 BFM example with foamRun"
foamRun 2>&1 | tee log.foamRun
