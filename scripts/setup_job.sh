#!/usr/bin/env bash

echo "============================================================"
echo "Setting up environment!"
echo "============================================================"
echo

echo "Running on node: $(hostname)"
echo "Date/time: $(date)"
echo
echo

# display cpu and memory info
echo "CPU model: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPU cores: $(nproc)"
echo "Total memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Available memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo

echo "Loading modules..."
module load python/3.11.6-gcc-13.2.0
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
echo "Modules loaded:"
module list
echo

echo "Loading virtual environment..."
source .venv/bin/activate
echo

echo "Loading environment variables from .env..."
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found! Create one with HF_HOME, HF_TOKEN, etc."
fi
echo

# use expandable memory segments to prevent pytorch cuda allocator fragmentation,
# which causes oom errors on long-sequence training even when total memory is available
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment setup, checking CUDA versions..."

nvidia-debugdump -l
cat /proc/driver/nvidia/version
echo
echo "CUDA version:"
nvcc --version

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo

echo "Now test python CUDA access..."

test_cuda  # from llm_cgr

echo "CUDA tested, starting job..."

echo
echo
