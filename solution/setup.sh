#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Prerequisites ==="
echo "Make sure you have:"
echo "  1. conda env activated (Python 3.11)"
echo "  2. CUDA toolkit installed: conda install -c nvidia cuda-toolkit"
echo "  3. CUDA_HOME set to your conda env"
echo ""

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[WARN] No conda env detected. Activate one: conda activate <env>"
fi

if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME="$CONDA_PREFIX"
    echo "[INFO] Setting CUDA_HOME=$CUDA_HOME"
fi

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Setting up 3D Gaussian Splatting ==="
if [ ! -d "third_party/gaussian-splatting" ]; then
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
    cd third_party/gaussian-splatting
    git submodule update --init --recursive
    cd ../..
    echo "Applying patches..."
    cd third_party/gaussian-splatting
    for p in ../../third_party_patches/*.patch; do
        patch -p1 < "$p"
        echo "  Applied: $p"
    done
    cd ../..
fi

cd third_party/gaussian-splatting

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

if [ ! -d "submodules/diff-gaussian-rasterization/build" ]; then
    echo "Building diff-gaussian-rasterization..."
    CUDA_HOME="$CUDA_HOME" pip install submodules/diff-gaussian-rasterization
fi

if [ ! -d "submodules/simple-knn/build" ]; then
    echo "Building simple-knn..."
    CUDA_HOME="$CUDA_HOME" pip install submodules/simple-knn
fi

cd ../../

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Extract competition data:"
echo "     unzip ../VAI_NVS_DATA.zip -d ../data/raw/"
echo "     (or copy existing data/raw/ from another machine)"
echo ""
echo "  2. Run the pipeline:"
echo "     python -m src.run_pipeline --scene hcm0031"
echo "     python -m src.run_pipeline --split public"
echo "     python -m src.run_pipeline --split private"
echo "     python -m src.run_pipeline --split all"
