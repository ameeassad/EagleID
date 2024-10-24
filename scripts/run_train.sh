#!/bin/bash
#SBATCH -A hpc2n2024-083
#SBATCH --job-name=train_Age_Classifier
#SBATCH --error=./jobs/job.%J.err
#SBATCH --output=./jobs/job.%J.out
#SBATCH --time=02:30:00
#SBATCH --gpus=1  # Request one GPU
#SBATCH -C 'zen4&GPU_ML'

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load necessary modules
module load GCC/12.3.0 OpenMPI/4.1.5
# module load PyTorch/2.1.2-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-CUDA-12.1.1
#module load libjpeg-turbo/3.0.1
#module load libpng/1.6.40
module load Mesa

source /proj/nobackup/aiforeagles/myenv/bin/activate

export PATH=/proj/nobackup/aiforeagles/myenv/bin:$PATH
export WANDB_API_KEY=ed39b65704eb2b673e6207ceb358c5f81710f29a

wandb login

# Set the PYTHONSTARTUP environment variable to the preload script
# export PYTHONSTARTUP=/proj/nobackup/aiforeagles/hrnet/lib/nms/load_custom_ops.py

# Check if torch and torchvision are installed, if not, install them
#python -c "import torch" &> /dev/null || pip install --no-deps torch==2.1.0+cu121 torchvision==0.14.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html


#export PYTHONNOUSERSITE=True


# Check if nvidia-smi exists and run it if found
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found in PATH"
fi

# Check nvcc version
nvcc --version

# Python script to check CUDA availability in PyTorch
python - <<END
import torch

print("PyTorch version:", torch.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available")
END

cd $PROJECT/EagleID/

echo "Starting training script with config-hpc.yaml"
srun python train.py  --config ./configs/config-hpc.yaml

echo "complete"
