#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Setting up 3D Gaussian Splatting ==="
if [ ! -d "third_party/gaussian-splatting" ]; then
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
    cd third_party/gaussian-splatting
    git submodule update --init --recursive
    cd ../..
    echo "Applying patches..."
    for p in third_party_patches/*.patch; do
        patch -p1 < "$p"
        echo "  Applied: $p"
    done
fi

cd third_party/gaussian-splatting

if [ ! -d "submodules/diff-gaussian-rasterization/build" ]; then
    echo "Building diff-gaussian-rasterization..."
    pip install submodules/diff-gaussian-rasterization
fi

if [ ! -d "submodules/simple-knn/build" ]; then
    echo "Building simple-knn..."
    pip install submodules/simple-knn
fi

cd ../../

echo ""
echo "=== Setup complete ==="
echo "To run the pipeline: python -m src.run_pipeline"
echo "To process all public scenes: python -m src.run_pipeline --split public"
